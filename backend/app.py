# backend/app.py
import io
import traceback
from pathlib import Path
import os
import random
import string
import logging
import shutil
from dataclasses import replace

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
from param_graph.elements.base_elements import Asset
from param_graph.elements.collections.batch_element import Batch
from param_graph.elements.collections.directory_element import Directory
from param_graph.elements.local_path import LocalPath
from param_graph.utils import save_artifact_asset, resolve_element
from engine.engine_provider import EngineProvider
from engine.encoders.clap_encoder import CLAPEncoder
from utils.audio import load_audio
from utils.form import create_dynamic_model
from utils.uid import XXH3_64
from utils.semantic_interrogation import SemanticInterrogator

app = Flask(__name__)
CORS(app)

# Filter out health check and job status logs
class QuietLogFilter(logging.Filter):
    def filter(self, record):
        msg = record.getMessage()
        return msg.find('/health') == -1 and msg.find('/job_status') == -1

# Add the filter to the Werkzeug logger (used by Flask's dev server)
logging.getLogger('werkzeug').addFilter(QuietLogFilter())

# Add a unique ID for the server instance
SERVER_INSTANCE_ID = ''.join(random.choices(string.ascii_letters + string.digits, k=16))

is_container = os.environ.get("RUNNING_IN_CONTAINER") == "true"

if is_container:
    data_path_str = os.environ.get("CONTAINER_DATA_PATH")
    if not data_path_str:
        print("Warning: CONTAINER_DATA_PATH not set. Defaulting to /data")
        data_path_str = "/data"
else:
    data_path_str = os.environ.get("LOCAL_DATA_PATH")
    if not data_path_str:
        print("Warning: LOCAL_DATA_PATH not set. Defaulting to ~/.stemma2a_data")
        data_path_str = "~/.stemma2a_data"

data_cache_root = Path(data_path_str).expanduser()
data_cache_root.mkdir(parents=True, exist_ok=True)

# Set Hugging Face cache directory to a persistent location within the data cache.
hf_cache_dir = data_cache_root / "huggingface"
hf_cache_dir.mkdir(parents=True, exist_ok=True)
os.environ["HF_HOME"] = str(hf_cache_dir)
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"

# Global state
device_type_accelerator = "cuda" if torch.cuda.is_available() else "cpu"
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
local_jobs = {}

graph_lock = threading.Lock()


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

        with graph_lock:
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

        with graph_lock:
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
        with graph_lock:
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


@app.route("/graph/create_batch", methods=["POST"])
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
    
        with graph_lock:
            batch = Batch(id=batch_id, member_ids=member_ids, member_type=member_type)
            param_graph.add_element(batch)

            # Update parents
            for member_id in member_ids:
                param_graph.update_element(member_id, {"parent": batch_id})
            
            update_batch_labels(batch_id)
            param_graph.save()
            
            updated_batch = param_graph.get_element(batch_id).to_dict()

        return jsonify({
            "message": "Batch created successfully",
            "collection": updated_batch,
            "success": True
        })

    except Exception as e:
        print(f"Failed to create batch: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/graph/update_batch", methods=["PUT", "POST"])
@app.route("/graph/update_batch/<batch_id>", methods=["PUT", "POST"])
def update_batch_endpoint(batch_id=None):
    """Update an existing batch with new members."""
    if param_graph is None:
        return jsonify({"error": "No project loaded"}), 400

    try:
        data = request.get_json()
        
        # Handle alternative route where batch_id is in the body
        if batch_id is None:
            batch_id = data.get("batch_id")
            
        if not batch_id:
            return jsonify({"error": "batch_id is required"}), 400
            
        new_member_ids = data.get("member_ids")
        if not new_member_ids:
            return jsonify({"error": "member_ids is required"}), 400

        with graph_lock:
            if not param_graph.G.has_node(batch_id):
                return jsonify({"error": f"Batch {batch_id} not found"}), 404

            batch_node_attrs = param_graph.G.nodes[batch_id]
            if batch_node_attrs.get('type') != 'batch':
                return jsonify({"error": f"Node {batch_id} is not a batch"}), 400

            old_member_ids = batch_node_attrs.get('member_ids', [])

            # Unlink removed members
            for m_id in old_member_ids:
                if m_id not in new_member_ids:
                    param_graph.update_element(m_id, {"parent": None, "alias": None})

            # Link new members
            for m_id in new_member_ids:
                param_graph.update_element(m_id, {"parent": batch_id})

            # Update batch properties
            batch_node_attrs['member_ids'] = new_member_ids
            
            update_batch_labels(batch_id)
            param_graph.save()
            
            updated_batch = param_graph.get_element(batch_id).to_dict()

        return jsonify({
            "message": "Batch updated successfully",
            "collection": updated_batch,
            "success": True
        })

    except Exception as e:
        print(f"Failed to update batch: {e}")
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

        with graph_lock:
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
        
        node_engine_args = {}
        for field in form_config:
            if field.get("type") == "node":
                field_name = field.get("name")
                node_id = dumped_params.pop(field_name, None)
                if node_id:
                    element = param_graph.get_element(node_id)
                    if element:
                        node_engine_args[f"{field_name}_element"] = element

        resolved_elements = list(node_engine_args.values())

        # Cache external audio files used in this step, and resolve to cache if missing
        for arg_name, element in node_engine_args.items():
            if isinstance(element, Audio):
                cache_used_audio(element.id)
                valid_path = resolve_audio_path(element.id)
                if valid_path and str(valid_path) != element.file.path:
                    element.file = replace(element.file, path=str(valid_path))
                    node_engine_args[arg_name] = element
            
            if not isinstance(element, (Audio, Model)):
                field_name = arg_name.removesuffix("_element")
                return jsonify({"error": f"Node '{element.id}' for field '{field_name}' is not a valid artifact."}), 400

        engine_args = {"model_element": model_element, **node_engine_args}
        linked_elements = [model_element, *resolved_elements]
        
        # --- Batching Logic ---
        batch_id = json_data.get("batch_id")
        if batch_id:
            with graph_lock:
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
            "status": "pending"
        }), 202

    except (ValidationError, ValueError) as e:
        print(f"Invalid request: {e}")
        traceback.print_exc()
        return jsonify({"error": "Invalid request", "details": str(e)}), 400
    except Exception as e:
        print(f"Generation failed: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500


@app.route("/job_status/<job_id>", methods=["GET"])
async def get_job_status(job_id):
    """Gets the status of a generation job and handles final artifact processing."""
    if engine_provider is None or param_graph is None:
        return jsonify({"error": "No project loaded"}), 400

    # Check for local backend jobs first
    if job_id in local_jobs:
        job_info = local_jobs[job_id]
        status = job_info.get("status")
        
        if status in ["completed", "failed"]:
            # Pop the job from memory once we send the final completed/failed state
            return jsonify(local_jobs.pop(job_id)), 200 if status == "completed" else 500
        return jsonify(job_info), 200

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
            
            # Ensure context is populated for labelling
            if hasattr(final_artifact, 'context') and not final_artifact.context:
                final_artifact = replace(final_artifact, context=job_context.get("validated_params", {}))
            
            with graph_lock:
                param_graph.add_element(final_artifact)

                batch_id = job_context.get("batch_id")
                if batch_id:
                    param_graph.update_element(final_artifact.id, {"parent": batch_id})
                    batch_node_attrs = param_graph.G.nodes[batch_id]
                    if 'member_ids' not in batch_node_attrs or not isinstance(batch_node_attrs['member_ids'], list):
                        batch_node_attrs['member_ids'] = []
                    if final_artifact.id not in batch_node_attrs['member_ids']:
                        batch_node_attrs['member_ids'].append(final_artifact.id)
                        
                    update_batch_labels(batch_id)

                for element in job_context.get("linked_elements", []):
                    print(f"Linking {element.id} to {final_artifact.id}")
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
            traceback_msg = status_info.get("traceback")
            active_jobs.pop(job_id, None)
            return jsonify({"status": "failed", "error": error_msg, "traceback": traceback_msg}), 500

        elif status == "not_found":
            active_jobs.pop(job_id, None)
            return jsonify({"status": "not_found", "error": f"Job {job_id} was lost."}), 404

        # For 'pending' or 'running', just return the status info
        return jsonify(status_info), 200

    except Exception as e:
        print(f"Failed to get job status: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500


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
            
        path_obj = Path(source_path)
        if not path_obj.exists():
            return jsonify({"error": f"Path '{source_path}' does not exist."}), 400
            
        element_id = uid_generator.from_string(str(path_obj.resolve()))
        path_node = LocalPath(id=element_id, name=path_obj.name, path=str(path_obj))
        
        with graph_lock:
            param_graph.add_element(path_node)
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
        
        with graph_lock:
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

@app.route("/expand_path", methods=["POST"])
def expand_path():
    """Reads a PathNode, creates a Directory compound node if needed, and populates it."""
    if param_graph is None:
        return jsonify({"error": "No project loaded"}), 400

    try:
        data = request.get_json()
        path_node_id = data.get("path_node_id")
        
        if not path_node_id:
            return jsonify({"error": "path_node_id is required"}), 400
            
        path_element = param_graph.get_element(path_node_id)
        if not path_element or not hasattr(path_element, "path"):
            return jsonify({"error": "Invalid path element"}), 400
            
        dir_path = Path(path_element.path)
        if not dir_path.is_dir():
            return jsonify({"error": f"Path {dir_path} is not a valid directory"}), 400
            
        # 1. Check if we already have an expanded Directory for this PathNode
        existing_dir_id = None
        with graph_lock:
            for u, v, edge_data in param_graph.G.out_edges(path_node_id, data=True):
                target_node = param_graph.get_element(v)
                if getattr(target_node, "type", None) == "directory":
                    existing_dir_id = v
                    break
            
            new_dir_created = False
            if not existing_dir_id:
                existing_dir_id = f"{path_node_id}.dir"
                new_dir = Directory(id=existing_dir_id, name=path_element.name, path=path_element.path)
                
                param_graph.add_element(new_dir)
                new_edge = param_graph.link(path_element, new_dir, action='expands')
                new_dir_created = True

        # 2. Gather paths of existing children to prevent duplicates during sync
        existing_paths = set()
        with graph_lock:
            for node_id, data in param_graph.G.nodes(data=True):
                if data.get("parent") == existing_dir_id:
                    el = param_graph.get_element(node_id)
                    if hasattr(el, "path"):
                        existing_paths.add(el.path)
                    elif hasattr(el, "file") and hasattr(el.file, "path"):
                        existing_paths.add(el.file.path)

        audio_exts = {'.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aiff'}
        new_elements = []
        
        for entry in dir_path.iterdir():
            if entry.name.startswith('.'):
                continue
                
            entry_str = str(entry)
            if entry_str in existing_paths:
                continue  # This file/folder is already in the graph

            
            if entry.is_dir():
                element_id = uid_generator.from_string(str(entry.resolve()))
                new_elements.append(LocalPath(id=element_id, name=entry.name, path=entry_str))
            elif entry.is_file() and entry.suffix.lower() in audio_exts:
                try:
                    audio_tensor = load_audio(device_accelerator, entry_str, APP_SAMPLE_RATE)
                    element_id = uid_generator.from_tensor(audio_tensor)
                    asset = Asset(path=entry_str, uid=element_id, extension=entry.suffix.lower())
                    new_elements.append(Audio(id=element_id, name=entry.name, context={}, file=asset))
                except Exception as e:
                    print(f"Failed to load audio for {entry.name} to generate UID: {e}")
                
        with graph_lock:
            for el in new_elements:
                param_graph.add_element(el)
                param_graph.update_element(el.id, {"parent": existing_dir_id})
            param_graph.save()
            
        # Trigger embedding calculation for the newly added audio files
        trigger_embedding_update(background=False)
        
        response_data = {
            "message": f"Successfully expanded path {path_element.name}",
            "directory_id": existing_dir_id,
            "elements": [el.to_dict() for el in new_elements],
            "success": True
        }
        if new_dir_created:
            response_data["new_directory"] = new_dir.to_dict()
            if new_edge:
                response_data["new_edge"] = new_edge
            
        return jsonify(response_data)

    except Exception as e:
        print(f"Failed to expand path: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

def resolve_audio_path(audio_id):
    """Gets the audio path, falling back to the project cache if the original file was deleted."""
    if param_graph is None:
        return None
        
    path_str = param_graph.get_path_from_id(audio_id, relative=False)
    if not path_str:
        return None
        
    original_path = Path(path_str)
    if original_path.exists():
        return original_path
        
    # Check fallback cache
    cache_dir = Path(param_graph.root) / "cache"
    if cache_dir.exists():
        cached_files = list(cache_dir.glob(f"{audio_id}.*"))
        if cached_files:
            return cached_files[0]
            
    return None

def cache_used_audio(audio_id):
    """Copies an external audio file to the local project cache if it's not already there."""
    if param_graph is None:
        return
        
    path_str = param_graph.get_path_from_id(audio_id, relative=False)
    if not path_str:
        return
        
    original_path = Path(path_str)
    if not original_path.exists():
        return  # Can't cache what doesn't exist
        
    # If the file is already inside the project directory, no need to cache
    if original_path.is_relative_to(Path(param_graph.root)):
        return
        
    cache_dir = Path(param_graph.root) / "cache"
    cache_dir.mkdir(exist_ok=True)
    
    cached_path = cache_dir / f"{audio_id}{original_path.suffix}"
    if not cached_path.exists():
        try:
            shutil.copy2(original_path, cached_path)
            print(f"Cached external audio file {audio_id} to {cached_path}")
        except Exception as e:
            print(f"Failed to cache audio {audio_id}: {e}")

def update_batch_labels(batch_id: str):
    """
    Recalculates the shared parameters of a batch and updates the batch's alias,
    as well as the aliases of all its children (based on their unique parameters).
    Assumes caller holds `graph_lock`.
    """
    if param_graph is None or not param_graph.G.has_node(batch_id):
        return

    batch_node_attrs = param_graph.G.nodes[batch_id]
    if batch_node_attrs.get('type') != 'batch':
        return
        
    member_ids = batch_node_attrs.get('member_ids', [])
    if not member_ids:
        return

    shared_context = {}
    member_diffs = {m_id: {} for m_id in member_ids}
    
    # Extract contexts
    contexts = []
    for m_id in member_ids:
        try:
            el = param_graph.get_element(m_id)
            ctx = getattr(el, 'context', {})
            contexts.append(ctx if isinstance(ctx, dict) else {})
        except Exception:
            contexts.append({})
    
    if contexts:
        # Find intersection of all keys and values
        common_keys = set.intersection(*(set(c.keys()) for c in contexts))
        for k in common_keys:
            if all(c[k] == contexts[0][k] for c in contexts):
                shared_context[k] = contexts[0][k]
        
        # Calculate what makes each member unique
        for m_id, ctx in zip(member_ids, contexts):
            diff = {k: v for k, v in ctx.items() if k not in shared_context or shared_context[k] != v}
            member_diffs[m_id] = diff

    print(f"Shared context: {shared_context}")
    print(f"Member diffs: {member_diffs}")

    # Generate a label for the batch based on the shared prompt/context
    member_type = batch_node_attrs.get('member_type', 'element')
    batch_alias = shared_context.get('prompt', f"{member_type.capitalize()} Batch")
    if len(str(batch_alias)) > 30:
        batch_alias = str(batch_alias)[:27] + "..."

    param_graph.update_element(batch_id, {"shared_context": shared_context, "alias": batch_alias})

    # Update member aliases
    for member_id in member_ids:
        diff = member_diffs.get(member_id, {})
        diff_items = [f"{k}: {v}" for k, v in diff.items()]
        diff_label = "\n".join(diff_items) if diff_items else "Base"
        
        if len(diff_label) > 40:
            diff_label = diff_label[:37] + "..."
            
        param_graph.update_element(member_id, {"alias": diff_label})

def trigger_labeling_update():
    """
    Forces a graph-wide recalculation of all batch and node labels.
    """
    if param_graph is None:
        return

    with graph_lock:
        batch_ids = [
            node for node, data in param_graph.G.nodes(data=True)
            if data.get('type') == 'batch'
        ]
        for batch_id in batch_ids:
            update_batch_labels(batch_id)
            
        param_graph.save()
    print("Labeling update completed successfully.")

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
            
            # Initialize interrogator for semantic edge labels
            interrogator = SemanticInterrogator(device_accelerator)
            bank_path = Path(__file__).parent / "data" / "semantic_transitions_bank.pt"
            has_label_bank = bank_path.exists()
            if has_label_bank:
                interrogator.load_bank_from_disk(str(bank_path))
            
            for group_type, embedding_type in SIMILARITY_GROUPS.items():
                # 1. Compute embeddings ONLY for nodes that need them
                with graph_lock:
                    nodes_to_process = []
                    for node in param_graph.G.nodes():
                        el = param_graph.get_element(node)
                        if el and getattr(el, 'type', None) == group_type:
                            emb_copy = getattr(el, 'embeddings', {}).copy()
                            nodes_to_process.append((node, emb_copy))
                        
                for node, embeddings in nodes_to_process:
                    if not force_recalculate and embedding_type in embeddings:
                        continue

                    try:
                        audio_path = resolve_audio_path(node)
                        if not audio_path:
                            continue
                            
                        embedding = clap_encoder.get_embedding(str(audio_path))
                        
                        with graph_lock:
                            current_data = param_graph.G.nodes[node]
                            current_embeddings = current_data.get('embeddings', {})
                            current_embeddings[embedding_type] = embedding.tolist()
                            param_graph.update_element(node, {'embeddings': current_embeddings})
                            
                        print(f"Updated {embedding_type} embedding for node {node}")
                    except Exception as e:
                        print(f"Could not update {embedding_type} embedding for node {node}. Error: {e}")

                with graph_lock:
                    # 2. Fast similarity edge rebuild based on cached embeddings
                    edges_to_remove = [
                        (u, v) for u, v, d in param_graph.G.edges(data=True) 
                        if d.get('group') == group_type
                    ]
                    param_graph.G.remove_edges_from(edges_to_remove)

                    group_nodes = {}
                    for node in param_graph.G.nodes():
                        el = param_graph.get_element(node)
                        if el and getattr(el, 'type', None) == group_type:
                            if embedding_type in getattr(el, 'embeddings', {}):
                                group_nodes[node] = el.embeddings[embedding_type]

                if len(group_nodes) <= 1:
                    with graph_lock:
                        param_graph.save()
                    continue

                node_ids = list(group_nodes.keys())
                all_latents = np.array(list(group_nodes.values()))
                
                # L2 Normalize all latents to project them onto the unit hypersphere
                norms = np.linalg.norm(all_latents, axis=1, keepdims=True)
                all_latents = all_latents / np.where(norms == 0, 1e-10, norms)
                
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

                with graph_lock:
                    for i, node_id in enumerate(node_ids):
                        near_indices = set(indices[i])
                        emb_A = all_latents[i]
                        
                        # 1. Add nearest neighbors (similar nodes)
                        for j, neighbor_idx in enumerate(indices[i]):
                            if i == neighbor_idx:
                                continue
                                
                            source_label = ""
                            if has_label_bank:
                                emb_B = all_latents[neighbor_idx]
                                diff_A_to_B = torch.tensor(emb_B - emb_A, dtype=torch.float32)
                                
                                res_A = interrogator.interrogate(diff_A_to_B, k=1)
                                
                                source_label = res_A[0][0] if res_A else ""
                                
                            param_graph.G.add_edge(
                                node_id, 
                                node_ids[neighbor_idx], 
                                id=f"edge-{node_id}-near-{node_ids[neighbor_idx]}",
                                type='spring', 
                                spring_type='near',
                                weight=float(1 - distances[i][j]),
                                group=group_type,
                                source_label=source_label
                            )
                            
                        # 2. Add furthest neighbors (ghost edges for separation)
                        furthest_indices = np.argsort(full_distances[i])[-k_furthest:]
                        
                        for neighbor_idx in furthest_indices:
                            if i == neighbor_idx or neighbor_idx in near_indices:
                                continue  # Prevent overlap on small graphs
                            
                            source_label = ""
                            if has_label_bank:
                                emb_B = all_latents[neighbor_idx]
                                diff_A_to_B = torch.tensor(emb_B - emb_A, dtype=torch.float32)
                                
                                res_A = interrogator.interrogate(diff_A_to_B, k=1)
                                
                                source_label = res_A[0][0] if res_A else ""
                            
                            param_graph.G.add_edge(
                                node_id, 
                                node_ids[neighbor_idx], 
                                id=f"edge-{node_id}-dist-{node_ids[neighbor_idx]}",
                                type='spring', 
                                spring_type='distant',
                                weight=float(1 - full_distances[i][neighbor_idx]),
                                group=group_type,
                                source_label=source_label
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

@app.route("/update_labels", methods=["POST"])
def update_labels():
    """Update all batch and node labels in the graph."""
    if param_graph is None:
        return jsonify({"error": "No project loaded"}), 400

    try:
        trigger_labeling_update()
        return jsonify({
            "message": "Labels updated successfully.",
            "success": True
        }), 200
    except Exception as e:
        print(f"Failed to update labels: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

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
        audio_path = resolve_audio_path(audio_id)
        if not audio_path:
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
        audio_path = resolve_audio_path(audio_id)
        print(f"Path resolved for graph: {audio_path}")

        if not audio_path:
            print(f"Audio file not found for id: {audio_id}")
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
        audio_path = resolve_audio_path(audio_id)
        if not audio_path:
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
        
        with graph_lock:
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
        
        with graph_lock:
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
        with graph_lock:
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