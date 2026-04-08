# backend/app.py
import io
import traceback
from pathlib import Path
import os
import random
import string
import logging
import time
import asyncio

import threading
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import torch
import torchaudio
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_distances
from pydantic import ValidationError

from param_graph.graph import ParameterGraph
from param_graph.elements.models.base_model_element import Model
from param_graph.elements.artifacts.audio_element import Audio
from param_graph.elements.collections.batch_element import Batch
from param_graph.utils import extract_graph_elements, save_artifact_asset, resolve_element
from engine.engine_provider import EngineProvider
from engine.encoders.clap_encoder import CLAPEncoder
from utils.audio import load_audio
from utils.form import create_dynamic_model
from utils.uid import XXH3_64

app = Flask(__name__)
CORS(app)

# Filter out health check logs
class NoHealthCheckLogFilter(logging.Filter):
    def filter(self, record):
        return record.getMessage().find('/health') == -1

# Add the filter to the Werkzeug logger (used by Flask's dev server)
logging.getLogger('werkzeug').addFilter(NoHealthCheckLogFilter())

# Add a unique ID for the server instance
SERVER_INSTANCE_ID = ''.join(random.choices(string.ascii_letters + string.digits, k=16))

# Global state
device_type_accelerator = "cpu"
device_accelerator = torch.device(device_type_accelerator)
# A default sample rate for processing and playback
APP_SAMPLE_RATE = 48000
SIMILARITY_GROUPS = {
    "audio": "clap"
}

param_graph: ParameterGraph = None
execution_url = os.environ.get("ENGINE_URL")
uid_generator = XXH3_64()
# EngineProvider is now initialized after a project is loaded
engine_provider: EngineProvider = None

# Context storage for background jobs
active_jobs = {}


# --------------------
#  Health Check
# --------------------

@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint"""
    global param_graph
    return jsonify({
        "status": "healthy",
        "device": str(device_accelerator),
        "project_loaded": param_graph is not None,
        "server_instance_id": SERVER_INSTANCE_ID,
    })


# --------------------
#  Project Management
# --------------------

def initialize_engine(project_path: str):
    """Initializes the engine provider for a given project."""
    global engine_provider, execution_url
    # Reset the singleton instance
    EngineProvider._engine_instance = None
    engine_provider = EngineProvider(remote_url=execution_url, data_root=project_path)
    if execution_url:
        print(f"Engine: Using remote engine at {execution_url}")
    else:
        print(f"Engine: Using local engine with data root {project_path}")


@app.route("/load_project", methods=["POST"])
def load_project():
    """
    Load a project from an absolute path.
    """
    global param_graph
    
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Request body is required"}), 400

        project_path_str = data.get("project_path")
        if not project_path_str:
            return jsonify({"error": "project_path is required"}), 400
        
        project_path = Path(project_path_str)
        if not project_path.is_dir():
            return jsonify({"error": f"Project path '{project_path_str}' does not exist or is not a directory."}), 404

        param_graph = ParameterGraph(str(project_path))
        initialize_engine(str(project_path))

        if param_graph.load():
            return jsonify({
                "message": f"Project '{project_path.name}' loaded successfully.",
                "project_name": project_path.name,
                "project_path": str(project_path),
                "success": True
            })
        
        # If load fails, it might be a directory without a project file yet.
        # We can still "load" it to create one.
        param_graph.save()

        return jsonify({
            "message": f"New project '{project_path.name}' created and loaded.",
            "project_name": project_path.name,
            "project_path": str(project_path),
            "success": True
        })
        
    except Exception as e:
        print(f"Failed to load project: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
    
@app.route("/create_project", methods=["POST"])
def create_project():
    """
    Create a new project at a given absolute path.
    """
    global param_graph
    
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Request body is required"}), 400
            
        project_path_str = data.get("project_path")
        project_name = data.get("project_name") # Optional, can derive from path
        if not project_path_str:
            return jsonify({"error": "project_path is required"}), 400
        
        project_path = Path(project_path_str)
        project_path.mkdir(parents=True, exist_ok=True)

        final_project_name = project_name or project_path.name

        param_graph = ParameterGraph(str(project_path))
        param_graph.project_name = final_project_name
        initialize_engine(str(project_path))
        param_graph.save()

        return jsonify({
            "message": f"Project '{final_project_name}' created successfully.",
            "project_name": final_project_name,
            "project_path": str(project_path),
            "success": True
        })
        
    except Exception as e:
        print(f"Failed to create project: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/project", methods=["GET"])
def get_project():
    """Get current project info"""
    global param_graph
    if param_graph is not None:
        return jsonify({
            "message": "success",
            "project_name": Path(param_graph.root).name,
            "project_path": str(param_graph.root),
            "success": True
        })
    else:
        return jsonify({
            "message": "no project selected",
            "success": False
        })


# --------------------
#  Graph Data
# --------------------

@app.route("/graph", methods=["GET"])
def get_graph():
    """Get graph data in batch mode"""
    if param_graph is None:
        return jsonify({"error": "No project loaded"}), 400
    
    try:
        graph_data = param_graph.to_json(mode='batch')
        return jsonify({
            "message": "success",
            "graph_data": graph_data,
            "success": True
        })
    except Exception as e:
        print(f"Failed to get graph data: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/graph/batch", methods=["POST"])
def batch_elements():
    """Create a batch from a selection of nodes."""
    if param_graph is None:
        return jsonify({"error": "No project loaded"}), 400

    try:
        data = request.get_json()
        member_ids = data.get("member_ids")
        if not member_ids:
            return jsonify({"error": "member_ids is required"}), 400

        # 1. Create a new collection element
        batch_id = uid_generator.from_uids(member_ids)

        member_type = None
        for member_id in member_ids:
            member = param_graph.get_element(member_id)
            if member_type is None:
                member_type = member.type
            elif member_type != member.type:
                return jsonify({"error": "All members must be of the same type"}), 400
    
        batch = Batch(id=batch_id, member_ids=member_ids, member_type=member_type)
        param_graph.add_element(batch)

        # Update parents
        for member_id in member_ids:
            param_graph.update_element(member_id, {"parent": batch_id})
        
        param_graph.save()

        return jsonify({
            "message": "Batch created successfully",
            "collection": batch.to_dict(),
            "success": True
        })

    except Exception as e:
        print(f"Failed to create batch: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# --------------------
#  Model Operations
# --------------------

@app.route("/register_model", methods=["POST"])
async def import_model():
    """Register a model by providing absolute paths to its files."""
    if param_graph is None:
        return jsonify({"error": "No project loaded"}), 400
    
    try:
        data = request.get_json()
        adapter_name = data.pop("adapter")

        config_path = data.get("config_path")
        checkpoint_path = data.get("checkpoint_path")

        if not config_path or not checkpoint_path:
            return jsonify({"error": "config_path and checkpoint_path are required"}), 400

        # Paths are now expected to be absolute and valid on the server's filesystem.
        data["config_path"] = str(Path(config_path))
        data["checkpoint_path"] = str(Path(checkpoint_path))

        engine = engine_provider.get_engine()
        model_artifact = await engine.register_model(adapter_name, **data)

        param_graph.add_element(model_artifact)
        param_graph.save()
        
        return jsonify({
            "message": "Model registered successfully",
            "success": True
        })
        
    except Exception as e:
        print(f"Failed to import model: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/adapter_config/<adapter_name>", methods=["GET"])
async def get_adapter_config(adapter_name):
    """Get the form configuration for a specific adapter."""
    try:
        engine = engine_provider.get_engine()
        config = await engine.get_adapter_config(adapter_name)
        return jsonify(config)

    except Exception as e:
        print(f"Failed to get adapter config: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/generate", methods=["POST"])
async def generate():
    """
    Handles the full audio generation lifecycle: validation, queuing,
    polling, and final artifact processing.
    """
    if param_graph is None or engine_provider is None:
        return jsonify({"error": "No project loaded"}), 400
    
    try:
        json_data = request.get_json()
        print(f"app.py: Received /generate payload: {json_data}")
        model_id = json_data.get("model_id")
        job_id = json_data.get("job_id")
        
        if not job_id:
            return jsonify({"error": "'job_id' is required."}), 400
        if not model_id:
            return jsonify({"error": "'model_id' is required."}), 400

        model_element = param_graph.get_element(model_id)
        if not isinstance(model_element, Model):
            return jsonify({"error": f"Node '{model_id}' is not a valid model."}), 400
        
        engine = engine_provider.get_engine()
        form_config = await engine.get_adapter_config(model_element.adapter)
        form_config = form_config.get("generate")
        if not form_config:
            return jsonify({"error": f"Adapter '{model_element.adapter}' has no 'generate' configuration"}), 404

        DynamicArgsModel = create_dynamic_model(form_config)
        validated_params = DynamicArgsModel.model_validate(json_data)
        dumped_params = validated_params.model_dump()
        
        node_engine_args, resolved_elements = extract_graph_elements(
            form_config, dumped_params, param_graph
        )

        for arg_name, element in node_engine_args.items():
            if not isinstance(element, (Audio, Model)):
                field_name = arg_name.removesuffix("_element")
                return jsonify({"error": f"Node '{element.id}' for field '{field_name}' is not a valid artifact."}), 400

        engine_args = {"model_element": model_element, **node_engine_args}
        linked_elements = [model_element, *resolved_elements]
        
        # --- Batching Logic ---
        batch_id = json_data.get("batch_id")
        if batch_id:
            if not param_graph.G.has_node(batch_id):
                # Create the batch element if it's the first time we see this ID
                batch_element = Batch(id=batch_id, member_type="audio")
                param_graph.add_element(batch_element)
                print(f"Created new batch element {batch_id}")

        # --- Execute, Poll, and Process ---
        print(f"Submitting generation job {job_id} to engine...")
        returned_job_id = await engine.execute("generate", job_id=job_id, **engine_args, **dumped_params)
        
        if returned_job_id != job_id:
             print(f"Warning: Engine returned different job_id {returned_job_id} than requested {job_id}")
             job_id = returned_job_id
             
        print(f"app.py: Engine returned job_id: {job_id}")
        
        # Store job context to process the artifact later when the frontend polls /job_status
        active_jobs[job_id] = {
            "batch_id": batch_id,
            "linked_elements": linked_elements,
            "validated_params": validated_params.model_dump()
        }

        return jsonify({
            "message": "Job started successfully.",
            "job_id": job_id,
            "status": "running"
        }), 202

    except (ValidationError, ValueError) as e:
        print(f"Invalid request: {e}")
        traceback.print_exc()
        return jsonify({"error": "Invalid request", "details": str(e)}), 400
    except Exception as e:
        print(f"Generation failed: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/job_status/<job_id>", methods=["GET"])
async def get_job_status(job_id):
    """Gets the status of a generation job and handles final artifact processing."""
    if engine_provider is None or param_graph is None:
        return jsonify({"error": "No project loaded"}), 400

    try:
        engine = engine_provider.get_engine()
        status_info = await engine.get_job_status(job_id)
        status = status_info.get("status")

        if status == "completed":
            print(f"Job {job_id} completed. Processing artifact...")
            job_context = active_jobs.pop(job_id, {})
            
            result_dict = status_info.get("result", {})
            artifact_data = result_dict.get('artifact', result_dict)

            if not artifact_data:
                raise Exception("Completed job did not return a valid artifact.")

            temp_artifact = resolve_element(artifact_data) if isinstance(artifact_data, dict) else artifact_data

            output_dir = param_graph.root / "generate"
            final_artifact = save_artifact_asset(temp_artifact, output_dir, asset_name="file")
            
            param_graph.add_element(final_artifact)

            batch_id = job_context.get("batch_id")
            if batch_id:
                param_graph.update_element(final_artifact.id, {"parent": batch_id})
                batch_node_attrs = param_graph.G.nodes[batch_id]
                if 'member_ids' not in batch_node_attrs or not isinstance(batch_node_attrs['member_ids'], list):
                    batch_node_attrs['member_ids'] = []
                batch_node_attrs['member_ids'].append(final_artifact.id)

            for element in job_context.get("linked_elements", []):
                param_graph.link(element, final_artifact)
            param_graph.save()
            
            trigger_embedding_update()
            
            print("Artifact processed and saved to graph successfully.")
            return jsonify({
                "status": "completed",
                "message": "Audio generated and registered successfully.",
                "artifact": final_artifact.to_dict(),
                "validated_params": job_context.get("validated_params")
            }), 200

        elif status == "failed":
            error_msg = status_info.get("error", "Unknown error during generation.")
            active_jobs.pop(job_id, None)
            return jsonify({"status": "failed", "error": error_msg}), 500

        elif status == "not_found":
            active_jobs.pop(job_id, None)
            return jsonify({"status": "not_found", "error": f"Job {job_id} was lost."}), 404

        # For 'pending' or 'running', just return the status info
        return jsonify(status_info), 200

    except Exception as e:
        print(f"Failed to get job status: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/jobs/<job_id>/cancel", methods=["POST"])
async def cancel_job(job_id):
    """
    Requests cancellation of a specific job.
    """
    if engine_provider is None:
        return jsonify({"error": "No project loaded"}), 400

    try:
        engine = engine_provider.get_engine()
        await engine.cancel_job(job_id)
        return jsonify({
            "message": f"Cancellation requested for job {job_id}.",
            "success": True
        }), 202
    except Exception as e:
        print(f"Failed to cancel job: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# --------------------
#  External Sources
# --------------------

@app.route("/add_external_source", methods=["POST"])
def add_external_source():
    """
    Add external audio source.
    """
    if param_graph is None:
        return jsonify({"error": "No project loaded"}), 400
    
    try:
        data = request.get_json()
        source_path = data.get('source_path')
        
        if not source_path:
            return jsonify({"error": "source_path is required"}), 400
        
        param_graph.add_external_source(source_path)
        param_graph.save()
        
        # Trigger an incremental embedding update
        trigger_embedding_update()
        
        return jsonify({
            "message": "External source added successfully",
            "success": True
        })
        
    except Exception as e:
        print(f"Failed to add external source: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/rescan_source", methods=["POST"])
def rescan_source():
    """Rescan external source"""
    if param_graph is None:
        return jsonify({"error": "No project loaded"}), 400
    
    try:
        data = request.get_json()
        source_name = data.get('source_name')
        
        if not source_name:
            return jsonify({"error": "source_name is required"}), 400
        
        param_graph.scan_external_source(source_name)
        param_graph.save()
        
        # Trigger an incremental embedding update
        trigger_embedding_update()
        
        return jsonify({
            "message": "Source rescanned successfully",
            "success": True
        })
        
    except Exception as e:
        print(f"Failed to rescan source: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


def trigger_embedding_update(force_recalculate=False, background=True):
    """
    Background task to compute missing embeddings and recalculate similarity edges.
    If force_recalculate is True, it will recompute embeddings for all nodes.
    """
    if param_graph is None:
        return

    def update_embeddings_task():
        try:
            # Instantiate encoder only once per task run
            clap_encoder = CLAPEncoder()
            
            for group_type, embedding_type in SIMILARITY_GROUPS.items():
                # 1. Compute embeddings ONLY for nodes that need them
                for node, data in param_graph.G.nodes(data=True):
                    if data.get('type') != group_type:
                        continue
                        
                    embeddings = data.get('embeddings', {})
                    
                    # Optimization: Skip if we already have the embedding and aren't forcing a recalculation
                    if not force_recalculate and embedding_type in embeddings:
                        continue
                        
                    try:
                        audio_path = param_graph.get_path_from_id(node, relative=False)
                        if not audio_path:
                            continue
                            
                        embedding = clap_encoder.get_embedding(audio_path)
                        embeddings[embedding_type] = embedding.tolist()
                        
                        param_graph.update_element(node, {'embeddings': embeddings})
                        print(f"Updated {embedding_type} embedding for node {node}")
                    except Exception as e:
                        print(f"Could not update {embedding_type} embedding for node {node}. Error: {e}")

                # 2. Fast similarity edge rebuild based on cached embeddings
                edges_to_remove = [
                    (u, v) for u, v, d in param_graph.G.edges(data=True) 
                    if d.get('group') == group_type
                ]
                param_graph.G.remove_edges_from(edges_to_remove)

                # Get all nodes of the current group type with the required cached embeddings
                group_nodes = {
                    node: data['embeddings'][embedding_type]
                    for node, data in param_graph.G.nodes(data=True)
                    if data.get('type') == group_type and embedding_type in data.get('embeddings', {})
                }

                if len(group_nodes) <= 1:
                    continue

                node_ids = list(group_nodes.keys())
                all_latents = np.array(list(group_nodes.values()))
                n_nodes = len(node_ids)
                
                # Dynamically scale k based on graph size (logarithmic scaling)
                # e.g., 10 nodes -> k_near=2, 100 nodes -> k_near=5, 1000+ nodes -> k_near=7
                safe_n = max(n_nodes, 1)
                k_near = max(2, min(7, int(np.log10(safe_n) * 2.5)))
                k_far = max(1, min(4, int(np.log10(safe_n) * 1.5)))

                # Build the nearest neighbors model
                nn = NearestNeighbors(n_neighbors=min(n_nodes, k_near), metric='cosine', algorithm='brute')
                nn.fit(all_latents)

                # Find neighbors for each node
                distances, indices = nn.kneighbors(all_latents)
                
                # Calculate full distance matrix for distant neighbors
                full_distances = cosine_distances(all_latents)
                k_furthest = min(n_nodes - 1, k_far)

                for i, node_id in enumerate(node_ids):
                    near_indices = set(indices[i])
                    
                    # 1. Add nearest neighbors (similar nodes)
                    for j, neighbor_idx in enumerate(indices[i]):
                        if i == neighbor_idx:
                            continue
                            
                        param_graph.G.add_edge(
                            node_id, 
                            node_ids[neighbor_idx], 
                            type='spring', 
                            spring_type='near',
                            weight=1 - distances[i][j],
                            group=group_type
                        )
                        
                    # 2. Add furthest neighbors (ghost edges for separation)
                    furthest_indices = np.argsort(full_distances[i])[-k_furthest:]
                    
                    for neighbor_idx in furthest_indices:
                        if i == neighbor_idx or neighbor_idx in near_indices:
                            continue  # Prevent overlap on small graphs
                        
                        param_graph.G.add_edge(
                            node_id, 
                            node_ids[neighbor_idx], 
                            type='spring', 
                            spring_type='distant',
                            weight=1 - full_distances[i][neighbor_idx],
                            group=group_type
                        )

            param_graph.save()
            print("Embeddings updated and similarity edges created successfully")

        except Exception as e:
            print(f"Failed to update embeddings: {e}")
            traceback.print_exc()
            
    if background:
        thread = threading.Thread(target=update_embeddings_task)
        thread.start()
    else:
        update_embeddings_task()

@app.route("/update_embeddings", methods=["POST"])
def update_embeddings():
    """Update all embeddings and create similarity edges."""
    if param_graph is None:
        return jsonify({"error": "No project loaded"}), 400

    data = request.get_json(silent=True) or {}
    force = data.get("force", False)
    
    # Run synchronously so the frontend's HTTP request waits for completion
    trigger_embedding_update(force_recalculate=force, background=False)

    return jsonify({
        "message": "Embeddings updated successfully.",
        "success": True
    }), 200

# --------------------
#  Audio Operations
# --------------------

@app.route("/audio_data/<string:audio_id>", methods=["GET"])
def serve_audio_data(audio_id):
    """
    Loads an audio file, processes it, and returns the raw audio data.
    This ensures a consistent format (stereo WAV) for playback.
    """
    if param_graph is None:
        return jsonify({"error": "No project loaded"}), 400
    
    try:
        audio_path = param_graph.get_path_from_id(audio_id, relative=False)
        if not audio_path or not Path(audio_path).exists():
            return jsonify({"error": "Audio file not found"}), 404

        # Load audio using the robust loader (converts to stereo, resamples)
        audio_tensor = load_audio(device_accelerator, str(audio_path), APP_SAMPLE_RATE)
        
        # Save the tensor to an in-memory buffer
        buffer = io.BytesIO()
        torchaudio.save(buffer, audio_tensor, APP_SAMPLE_RATE, format="wav")
        buffer.seek(0)
        
        return send_file(buffer, mimetype="audio/wav")
        
    except Exception as e:
        print(f"Failed to serve audio data: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/audio_path/<string:audio_id>", methods=["GET"])
def get_audio_path(audio_id):
    """Get the absolute path of an audio file"""
    print(f"get_audio_path called with id: {audio_id}")
    if param_graph is None:
        print("get_audio_path: param_graph is None. No project loaded.")
        return jsonify({"error": "No project loaded"}), 400
    
    try:
        audio_path = param_graph.get_path_from_id(audio_id, relative=False)
        print(f"Path retrieved from graph: {audio_path}")

        if not audio_path or not Path(audio_path).exists():
            print(f"Audio file not found at path: {audio_path}")
            return jsonify({"error": "Audio file not found"}), 404
        
        print(f"Returning audio path: {audio_path}")
        return jsonify({"path": str(audio_path)})
        
    except Exception as e:
        print(f"Failed to get audio path: {e}")
        traceback.print_exc()
        return jsonify({"error": f"An unexpected error occurred in get_audio_path: {str(e)}"}), 500

@app.route("/audio/<string:audio_id>", methods=["GET"])
def serve_audio(audio_id):
    """Serve audio files"""
    if param_graph is None:
        return jsonify({"error": "No project loaded"}), 400
    
    try:
        audio_path = param_graph.get_path_from_id(audio_id, relative=False)
        if not audio_path or not Path(audio_path).exists():
            return jsonify({"error": "Audio file not found"}), 404
        
        return send_file(str(audio_path))
        
    except Exception as e:
        print(f"Failed to serve audio: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/export", methods=["POST"])
def export_audio():
    """Export audio files"""
    if param_graph is None:
        return jsonify({"error": "No project loaded"}), 400
    
    try:
        data = request.get_json()
        names = data.get('names', [])
        export_name = data.get('export_name', 'export')
        
        if not names:
            return jsonify({"error": "names list is required"}), 400
        
        if len(names) == 1:
            export_path = param_graph.export_single(names[0], export_name)
        else:
            export_path = param_graph.export_batch(names, export_name)
        
        return jsonify({
            "message": "Export completed",
            "export_path": str(export_path),
            "success": True
        })
        
    except Exception as e:
        print(f"Export failed: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# --------------------
#  Element Updates
# --------------------

@app.route("/update_element", methods=["POST"])
def update_element():
    """Update element attributes"""
    if param_graph is None:
        return jsonify({"error": "No project loaded"}), 400
    
    try:
        data = request.get_json()
        element_name = data.get('name')
        attributes = data.get('attributes', {})
        
        if not element_name:
            return jsonify({"error": "name is required"}), 400
        
        param_graph.update_element(element_name, attributes)
        param_graph.save()
        
        return jsonify({
            "message": "Element updated successfully",
            "success": True
        })
        
    except Exception as e:
        print(f"Failed to update element: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/save_node_positions", methods=["POST"])
def save_node_positions():
    """Save the spatial positions of nodes."""
    if param_graph is None:
        return jsonify({"error": "No project loaded"}), 400
    
    try:
        data = request.get_json()
        positions = data.get('positions', {})
        
        if not positions:
            return jsonify({"error": "No positions provided"}), 400
        
        for node_id, pos in positions.items():
            param_graph.update_element(node_id, {"position": pos})
            
        param_graph.save()
        
        return jsonify({
            "message": "Node positions saved successfully",
            "success": True
        })
        
    except Exception as e:
        print(f"Failed to save node positions: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/element/<element_id>", methods=["DELETE"])
def remove_element(element_id):
    """Remove an element from the graph"""
    if param_graph is None:
        return jsonify({"error": "No project loaded"}), 400

    try:
        param_graph.remove_element(element_id)
        param_graph.save()

        return jsonify({
            "message": "Element removed successfully",
            "success": True
        })

    except Exception as e:
        print(f"Failed to remove element: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# --------------------
#  Error Handlers
# --------------------

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500

# --------------------
#  Main
# --------------------

if __name__ == "__main__":
    print(f"Starting StemmA2A backend on device: {device_accelerator}")
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )