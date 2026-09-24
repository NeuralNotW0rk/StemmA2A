# backend/app.py
import sys
from pathlib import Path

# Add sibling NeutralSelection directory to sys.path dynamically
sibling_neutral_selection = Path(__file__).parent.parent.parent / "NeutralSelection"
if sibling_neutral_selection.exists() and str(sibling_neutral_selection.resolve()) not in sys.path:
    sys.path.insert(0, str(sibling_neutral_selection.resolve()))

import io
import traceback
import os
import json
import random
import string
import logging
import shutil
from dataclasses import replace

from dotenv import load_dotenv
# Load .env file from the project root (parent directory)
load_dotenv(Path(__file__).parent.parent / ".env")

import time
import threading
import asyncio
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
from param_graph.elements.models.stylegan_element import StyleGANModel
from param_graph.elements.artifacts.audio_element import Audio
from param_graph.elements.artifacts.image_element import Image
from param_graph.elements.artifacts.grating_element import Grating
from param_graph.elements.artifacts.latent_element import Latent
from param_graph.elements.base_elements import Asset, Artifact
from param_graph.elements.collections.group_element import Group
from param_graph.elements.artifacts.bundle_element import Bundle
from param_graph.elements.collections.directory_element import Directory
from param_graph.elements.local_path import LocalPath
from param_graph.utils import save_artifact_asset, resolve_element
from engine.engine_provider import EngineProvider
from engine.encoders.clap_encoder import CLAPEncoder
from engine.encoders.clip_encoder import CLIPEncoder
from utils.audio import load_audio, save_audio_to_buffer, save_audio
from utils.form import create_dynamic_model
from utils.uid import XXH3_64, path_from_uid
from utils.migrations import run_global_migrations, run_project_migrations
from utils.semantic_interrogation import SemanticInterrogator

from operations.registry import SyncRegistry
from diffracture import Actant
from diffracture.topology.grating import Grating as DiffractureGrating

from param_graph.elements.artifacts.individual_element import Individual
from param_graph.elements.artifacts.bundle_element import Bundle
from param_graph.elements.collections.group_element import Group

from operations.evolution import (
    recombine_offspring,
    RecombinedOffspring,
    breed_offspring,
    ReproducedOffspring,
    build_selection_strategy,
    build_crossover_strategy,
    get_evolution_operations,
    wrap_artifact_as_individual,
)
from evolution.lora_genome import (
    LoRAGenome,
    express_to_grating,
    get_lora_mutation_strategy,
    get_lora_crossover_strategy,
)
from neutral_selection.representation.individual import Individual as NSIndividual
import copy
import uuid
from coolname import generate_slug

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
run_global_migrations(data_cache_root)

# Set Hugging Face cache directory to a persistent location within the data cache.
hf_cache_dir = data_cache_root / "huggingface"
hf_cache_dir.mkdir(parents=True, exist_ok=True)
os.environ["HF_HOME"] = str(hf_cache_dir)
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"

# Set Triton and PyTorch extensions cache directories to a persistent location within the data cache.
triton_cache_dir = data_cache_root / "triton"
triton_cache_dir.mkdir(parents=True, exist_ok=True)
os.environ["TRITON_CACHE_DIR"] = str(triton_cache_dir)

torch_ext_dir = data_cache_root / "torch_extensions"
torch_ext_dir.mkdir(parents=True, exist_ok=True)
os.environ["TORCH_EXTENSIONS_DIR"] = str(torch_ext_dir)

# Global state
device_type_accelerator = "cuda" if torch.cuda.is_available() else "cpu"
device_accelerator = torch.device(device_type_accelerator)
# A default sample rate for processing and playback
APP_SAMPLE_RATE = 48000
SIMILARITY_GROUPS = {
    "audio": "clap",
    "image": "clip"
}

param_graph: ParameterGraph = None
execution_url = os.environ.get("ENGINE_URL")
uid_generator = XXH3_64()
sync_registry = SyncRegistry()
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
            run_project_migrations(project_path)
            param_graph = ParameterGraph(str(project_path))
            initialize_engine(str(project_path))

            if param_graph.load():
                return jsonify({
                    "message": f"Project '{project_path.name}' loaded successfully.",
                    "project_name": project_path.name,
                    "project_path": str(project_path),
                    "success": True
                })
            
            # If load fails, it is a new/uninitialized directory.
            # Initialize project_name and save initial graph.
            param_graph.project_name = project_path.name
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


@app.route("/graph/create_group", methods=["POST"])
def group_elements():
    """Create a group from a selection of nodes."""
    if param_graph is None:
        return jsonify({"error": "No project loaded"}), 400

    try:
        data = request.get_json() or {}
        member_ids = data.get("member_ids", [])

        # 1. Create a new collection element
        if member_ids:
            group_id = uid_generator.from_uids(member_ids)
            member_type = None
            for member_id in member_ids:
                member = param_graph.get_element(member_id)
                if member_type is None:
                    member_type = member.type
                elif member_type != member.type:
                    return jsonify({"error": "All members must be of the same type"}), 400
        else:
            import uuid
            group_id = uid_generator.from_string(str(uuid.uuid4()))
            member_type = None

        with graph_lock:
            group = Group(id=group_id, member_ids=member_ids, member_type=member_type)
            param_graph.add_element(group)

            # Update parents
            for member_id in member_ids:
                param_graph.update_element(member_id, {"parent": group_id})
            
            update_group_labels(group_id)
            param_graph.save()
            
            updated_group = param_graph.get_element(group_id).to_dict()

        return jsonify({
            "message": "Group created successfully",
            "collection": updated_group,
            "success": True
        })

    except Exception as e:
        print(f"Failed to create group: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/graph/update_group", methods=["PUT", "POST"])
@app.route("/graph/update_group/<group_id>", methods=["PUT", "POST"])
def update_group_endpoint(group_id=None):
    """Update an existing group with new members."""
    if param_graph is None:
        return jsonify({"error": "No project loaded"}), 400

    try:
        data = request.get_json()
        
        # Handle alternative route where group_id is in the body
        if group_id is None:
            group_id = data.get("group_id") or data.get("batch_id")
            
        if not group_id:
            return jsonify({"error": "group_id is required"}), 400
            
        new_member_ids = data.get("member_ids", [])

        with graph_lock:
            if not param_graph.G.has_node(group_id):
                return jsonify({"error": f"Group {group_id} not found"}), 404

            group_node_attrs = param_graph.G.nodes[group_id]
            if group_node_attrs.get('type') != 'group':
                return jsonify({"error": f"Node {group_id} is not a group"}), 400

            # Validate types of new members
            member_type = None
            for m_id in new_member_ids:
                member = param_graph.get_element(m_id)
                if member_type is None:
                    member_type = member.type
                elif member_type != member.type:
                    return jsonify({"error": "All members must be of the same type"}), 400

            old_member_ids = group_node_attrs.get('member_ids', [])

            # Unlink removed members
            for m_id in old_member_ids:
                if m_id not in new_member_ids:
                    param_graph.update_element(m_id, {"parent": None, "alias": None})

            # Link new members
            for m_id in new_member_ids:
                param_graph.update_element(m_id, {"parent": group_id})

            # Update group properties
            group_node_attrs['member_ids'] = new_member_ids
            group_node_attrs['member_type'] = member_type
            
            update_group_labels(group_id)
            param_graph.save()
            
            updated_group = param_graph.get_element(group_id).to_dict()

        return jsonify({
            "message": "Group updated successfully",
            "collection": updated_group,
            "success": True
        })

    except Exception as e:
        print(f"Failed to update group: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# --------------------
#  Model Operations
# --------------------

@app.route("/shared_models", methods=["GET"])
async def get_shared_models():
    """Exposes the shared models from the engine provider."""
    if engine_provider is None:
        return jsonify({"shared_models": [], "success": True})
    try:
        engine = engine_provider.get_engine()
        models = await engine.get_shared_models()
        return jsonify({"shared_models": models, "success": True})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

async def _import_model_task(job_id: str, data: dict) -> None:
    try:
        local_jobs[job_id]["status"] = "running"
        local_jobs[job_id]["progress"] = {
            "value": 10,
            "total": 100,
            "description": "Validating model configuration..."
        }

        # Check if registering pre-constructed model element (e.g. shared model)
        if "model_element" in data:
            model_dict = data["model_element"]
            model_artifact = resolve_element(model_dict)

            local_jobs[job_id]["progress"] = {
                "value": 40,
                "total": 100,
                "description": "Extracting model layers..."
            }
            if not getattr(model_artifact, 'layers', None):
                try:
                    engine = engine_provider.get_engine()
                    layers = await engine.get_model_layers(model_artifact)
                    model_artifact.layers = layers
                except Exception as layer_err:
                    print(f"Warning: Failed to extract layers for shared model during registration: {layer_err}")

            local_jobs[job_id]["progress"] = {
                "value": 80,
                "total": 100,
                "description": "Registering model in graph..."
            }
            with graph_lock:
                param_graph.add_element(model_artifact)
                param_graph.save()

            local_jobs[job_id]["status"] = "completed"
            local_jobs[job_id]["progress"] = {
                "value": 100,
                "total": 100,
                "description": "Shared model registered successfully."
            }
            local_jobs[job_id]["result"] = {
                "message": "Shared model registered successfully",
                "model": model_artifact.to_dict(),
                "node_id": model_artifact.id,
                "id": model_artifact.id
            }
            return

        adapter_name = data.pop("adapter", None) or data.pop("model_type", None)
        if not adapter_name:
            raise ValueError("adapter (or model_type) is required")
        config_path = data.get("config_path", "")
        checkpoint_path = data.get("checkpoint_path", "")

        if adapter_name == "stylegan2":
            config_path = config_path or ""
        else:
            if not config_path or not checkpoint_path:
                raise ValueError("config_path and checkpoint_path are required")

        data["config_path"] = str(Path(config_path)) if config_path else ""
        data["checkpoint_path"] = str(Path(checkpoint_path)) if checkpoint_path else ""

        local_jobs[job_id]["progress"] = {
            "value": 30,
            "total": 100,
            "description": "Initializing model weights and adapter..."
        }
        engine = engine_provider.get_engine()
        model_artifact = await engine.register_model(adapter_name, **data)

        local_jobs[job_id]["progress"] = {
            "value": 60,
            "total": 100,
            "description": "Extracting parameter layer topology..."
        }
        try:
            layers = await engine.get_model_layers(model_artifact)
            model_artifact.layers = layers
        except Exception as layer_err:
            print(f"Warning: Failed to extract layers during model registration: {layer_err}")

        local_jobs[job_id]["progress"] = {
            "value": 85,
            "total": 100,
            "description": "Saving model to parameter graph..."
        }
        with graph_lock:
            param_graph.add_element(model_artifact)
            param_graph.save()

        local_jobs[job_id]["status"] = "completed"
        local_jobs[job_id]["progress"] = {
            "value": 100,
            "total": 100,
            "description": "Model registered successfully."
        }
        local_jobs[job_id]["result"] = {
            "message": "Model registered successfully",
            "model": model_artifact.to_dict(),
            "node_id": model_artifact.id,
            "id": model_artifact.id
        }

    except Exception as e:
        print(f"Failed in async model registration: {e}")
        traceback.print_exc()
        local_jobs[job_id]["status"] = "failed"
        local_jobs[job_id]["progress"] = None
        local_jobs[job_id]["error"] = str(e)
        local_jobs[job_id]["traceback"] = traceback.format_exc()


@app.route("/register_model", methods=["POST"])
async def import_model():
    """Register a model asynchronously in a local background task."""
    if param_graph is None or engine_provider is None:
        return jsonify({"error": "No project loaded"}), 400

    try:
        data = request.get_json() or {}
        job_id = f"import_model_{uuid.uuid4().hex[:12]}"
        local_jobs[job_id] = {
            "status": "pending",
            "progress": {"value": 0, "total": 100, "description": "Starting model import..."},
            "result": None,
            "error": None,
            "traceback": None,
            "name": "Import Model"
        }

        threading.Thread(
            target=lambda: asyncio.run(_import_model_task(job_id, data)),
            daemon=True
        ).start()

        return jsonify({
            "success": True,
            "job_id": job_id,
            "message": "Model import job started"
        }), 202

    except Exception as e:
        print(f"Failed to initiate model import: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500


async def _export_shared_model_task(job_id: str, model_id: str) -> None:
    try:
        local_jobs[job_id]["status"] = "running"
        local_jobs[job_id]["progress"] = {
            "value": 10,
            "total": 100,
            "description": "Locating model in parameter graph..."
        }

        with graph_lock:
            model = param_graph.get_element(model_id)
            if not model:
                raise ValueError(f"Model '{model_id}' not found in parameter graph")

            entry = {
                "name": model.name,
                "adapter": model.adapter,
                "model_type": getattr(model, "model_type", None) or model.adapter,
                "checkpoint_uid": model.checkpoint.uid if hasattr(model, "checkpoint") and model.checkpoint else None,
                "checkpoint_path": model.checkpoint.path if hasattr(model, "checkpoint") and model.checkpoint else None,
                "checkpoint_size": model.checkpoint.size if hasattr(model, "checkpoint") and model.checkpoint else None,
                "config": model.config
            }

            encoder = getattr(model, "encoder", None)
            if encoder:
                entry["encoder_uid"] = encoder.uid
                entry["encoder_path"] = encoder.path
                entry["encoder_size"] = encoder.size

            local_jobs[job_id]["progress"] = {
                "value": 50,
                "total": 100,
                "description": "Updating shared models catalog..."
            }

            dummy_config_path = Path(param_graph.root) / "shared_models_dummy.json"
            shared_models_list = []
            if dummy_config_path.exists():
                try:
                    with open(dummy_config_path, "r") as f:
                        shared_models_list = json.load(f)
                        if not isinstance(shared_models_list, list):
                            shared_models_list = []
                except Exception as e:
                    print(f"Warning: Failed to read existing dummy config: {e}. Resetting.")
                    shared_models_list = []

            found = False
            for idx, existing in enumerate(shared_models_list):
                existing_uid = existing.get("checkpoint_uid") or existing.get("checkpoint", {}).get("uid")
                if existing_uid == entry["checkpoint_uid"]:
                    shared_models_list[idx] = entry
                    found = True
                    break

            if not found:
                shared_models_list.append(entry)

            with open(dummy_config_path, "w") as f:
                json.dump(shared_models_list, f, indent=2)

        local_jobs[job_id]["status"] = "completed"
        local_jobs[job_id]["progress"] = {
            "value": 100,
            "total": 100,
            "description": "Shared model exported successfully."
        }
        local_jobs[job_id]["result"] = {
            "message": f"Successfully exported shared model entry to {dummy_config_path.name}",
            "config_path": str(dummy_config_path),
            "node_id": model_id
        }

    except Exception as e:
        print(f"Failed to export shared model: {e}")
        traceback.print_exc()
        local_jobs[job_id]["status"] = "failed"
        local_jobs[job_id]["progress"] = None
        local_jobs[job_id]["error"] = str(e)
        local_jobs[job_id]["traceback"] = traceback.format_exc()


@app.route("/export_shared_model", methods=["POST"])
async def export_shared_model():
    """Exports a registered model to shared_models_dummy.json as an async local job."""
    if param_graph is None:
        return jsonify({"error": "No project loaded"}), 400

    try:
        data = request.get_json() or {}
        model_id = data.get("model_id")
        if not model_id:
            return jsonify({"error": "model_id is required"}), 400

        job_id = f"export_model_{uuid.uuid4().hex[:12]}"
        local_jobs[job_id] = {
            "status": "pending",
            "progress": {"value": 0, "total": 100, "description": "Starting shared model export..."},
            "result": None,
            "error": None,
            "traceback": None,
            "name": "Export Shared Model"
        }

        threading.Thread(
            target=lambda: asyncio.run(_export_shared_model_task(job_id, model_id)),
            daemon=True
        ).start()

        return jsonify({
            "success": True,
            "job_id": job_id,
            "message": "Shared model export job started"
        }), 202

    except Exception as e:
        print(f"Failed to start shared model export: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500


@app.route("/register_grating", methods=["POST"])
def register_grating():
    """Register a new Grating and bind it to a base model."""
    if param_graph is None:
        return jsonify({"error": "No project loaded"}), 400
    
    try:
        data = request.get_json()
        grating_name = data.get("name", "New Grating")
        checkpoint_path = data.get("checkpoint_path")
        base_model_id = data.get("base_model_id")

        if not checkpoint_path or not base_model_id:
            return jsonify({"error": "checkpoint_path and base_model_id are required"}), 400

        path_obj = Path(checkpoint_path)
        if not path_obj.is_file():
            return jsonify({"error": f"Grating path '{checkpoint_path}' does not exist or is not a file."}), 400

        with graph_lock:
            base_model = param_graph.get_element(base_model_id)
            if not base_model:
                return jsonify({"error": f"Base model '{base_model_id}' not found."}), 404

            # Temporarily load the grating to generate a hash from its modules
            loaded_grating = DiffractureGrating.load(str(path_obj.resolve()))
            element_id = uid_generator.from_module(loaded_grating)

            elements_config = [{
                "address": element.address,
                "kernel_type": element.kernel_type,
                "metadata": element.metadata
            } for element in loaded_grating._nodes]

            asset = Asset(path=str(path_obj.resolve()), uid=element_id, extension=path_obj.suffix.lower())
            
            grating_artifact = Grating(
                id=element_id,
                name=grating_name,
                context={},
                file=asset,
                base_model_id=base_model_id,
                elements=elements_config
            )

            param_graph.add_element(grating_artifact)
            param_graph.link(base_model, grating_artifact, relation='binds_to')
            param_graph.save()
        
        return jsonify({
            "message": "Grating registered successfully",
            "grating": grating_artifact.to_dict(),
            "success": True
        })
        
    except Exception as e:
        print(f"Failed to register grating: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/model/<model_id>/layers", methods=["GET"])
async def get_model_layers(model_id):
    """Inspects a model node and returns a list of its candidate layers for bending."""
    if param_graph is None or engine_provider is None:
        return jsonify({"error": "No project loaded"}), 400
        
    try:
        model_element = param_graph.get_element(model_id)
        if not isinstance(model_element, Model):
            return jsonify({"error": f"Node '{model_id}' is not a valid model."}), 400
            
        # Return cached layers directly if they exist
        if hasattr(model_element, 'layers') and model_element.layers is not None:
            return jsonify({
                "success": True,
                "layers": model_element.layers
            }), 200
            
        engine = engine_provider.get_engine()
        layers = await engine.get_model_layers(model_element)
        
        # Cache the layers in the graph element
        with graph_lock:
            param_graph.update_element(model_id, {"layers": layers})
            param_graph.save()
            
        return jsonify({
            "success": True,
            "layers": layers
        }), 200
        
    except Exception as e:
        print(f"Failed to inspect model layers: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


async def _create_grating_task(job_id: str, data: dict) -> None:
    try:
        local_jobs[job_id]["status"] = "running"
        local_jobs[job_id]["progress"] = {
            "value": 15,
            "total": 100,
            "description": "Validating model and layer configuration..."
        }

        model_id = data.get("model_id")
        grating_name = data.get("name", "New Grating")
        elements_input = data.get("elements", [])

        if not model_id or not elements_input:
            raise ValueError("model_id and elements list are required")

        model_element = param_graph.get_element(model_id)
        if not isinstance(model_element, Model):
            raise ValueError(f"Node '{model_id}' is not a valid model.")

        local_jobs[job_id]["progress"] = {
            "value": 45,
            "total": 100,
            "description": "Constructing grating topology..."
        }
        engine = engine_provider.get_engine()
        grating_artifact = await engine.create_grating(model_element, grating_name, elements_input)
        grating_artifact.context = data.get("context", {})

        local_jobs[job_id]["progress"] = {
            "value": 85,
            "total": 100,
            "description": "Saving grating artifact to parameter graph..."
        }
        with graph_lock:
            param_graph.add_element(grating_artifact)
            param_graph.link(model_element, grating_artifact, relation='binds_to')
            param_graph.save()

        local_jobs[job_id]["status"] = "completed"
        local_jobs[job_id]["progress"] = {
            "value": 100,
            "total": 100,
            "description": "Grating created successfully."
        }
        local_jobs[job_id]["result"] = {
            "message": "Grating created successfully",
            "grating": grating_artifact.to_dict(),
            "node_id": grating_artifact.id,
            "id": grating_artifact.id
        }

    except Exception as e:
        print(f"Failed in async grating creation: {e}")
        traceback.print_exc()
        local_jobs[job_id]["status"] = "failed"
        local_jobs[job_id]["progress"] = None
        local_jobs[job_id]["error"] = str(e)
        local_jobs[job_id]["traceback"] = traceback.format_exc()


@app.route("/create_grating", methods=["POST"])
async def create_grating():
    """Dynamically creates a new Bending Grating as an async local background job."""
    if param_graph is None or engine_provider is None:
        return jsonify({"error": "No project loaded"}), 400

    try:
        data = request.get_json() or {}
        job_id = f"create_grating_{uuid.uuid4().hex[:12]}"
        local_jobs[job_id] = {
            "status": "pending",
            "progress": {"value": 0, "total": 100, "description": "Starting grating creation..."},
            "result": None,
            "error": None,
            "traceback": None,
            "name": "Create Grating"
        }

        threading.Thread(
            target=lambda: asyncio.run(_create_grating_task(job_id, data)),
            daemon=True
        ).start()

        return jsonify({
            "success": True,
            "job_id": job_id,
            "message": "Grating creation job started"
        }), 202

    except Exception as e:
        print(f"Failed to start grating creation: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500

@app.route("/wrap_individual", methods=["POST"])
async def wrap_individual():
    """
    Wraps a precursor artifact (such as an Audio node) into a baseline Individual node
    with generation 0 and zero/baseline LoRA weights, linking it to its base model and precursor.
    """
    if param_graph is None or engine_provider is None:
        return jsonify({"error": "No project loaded"}), 400

    try:
        data = request.get_json() or {}
        return await _dispatch_wrap_individual_operation(data)
    except Exception as e:
        print(f"Failed to wrap individual: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500


async def _dispatch_wrap_individual_operation(data: dict):
    """
    Synchronously creates a baseline Individual node from a precursor artifact.
    """
    if param_graph is None or engine_provider is None:
        return jsonify({"error": "No project loaded"}), 400

    params = data.get("params", {}) or {}
    
    # 1. Resolve precursor node ID
    initiator = data.get("initiator") or params.get("initiator") or {}
    initiator_id = initiator.get("id") if isinstance(initiator, dict) else (str(initiator).strip() if initiator else None)

    precursor_val = (
        data.get("precursor_id")
        or data.get("precursor")
        or data.get("precursor_audio_id")
        or data.get("source_node")
        or data.get("source_node_id")
        or data.get("source_audio")
        or data.get("source_audio_id")
        or data.get("source_id")
        or data.get("artifact_id")
        or data.get("node_id")
        or params.get("precursor_id")
        or params.get("precursor")
        or params.get("precursor_audio_id")
        or params.get("source_node")
        or params.get("source_node_id")
        or params.get("source_audio")
        or params.get("source_audio_id")
        or params.get("source_id")
        or params.get("artifact_id")
        or initiator_id
    )
    if isinstance(precursor_val, dict):
        precursor_node_id = precursor_val.get("id")
    else:
        precursor_node_id = str(precursor_val).strip() if precursor_val else None

    if not precursor_node_id:
        return jsonify({"error": "Precursor artifact is required to wrap an Individual."}), 400

    with graph_lock:
        precursor_node = param_graph.get_element(precursor_node_id)
        if not precursor_node:
            return jsonify({"error": f"Precursor artifact '{precursor_node_id}' not found."}), 400

        # 2. Resolve model_id: from data, precursor context, or incoming edges
        model_val = data.get("model_id") or data.get("model") or params.get("model") or params.get("model_id")
        if isinstance(model_val, dict):
            model_id = model_val.get("id")
        else:
            model_id = str(model_val).strip() if model_val else None

        if not model_id:
            ctx = getattr(precursor_node, "context", {}) or {}
            model_id = ctx.get("model_id")

        if not model_id and hasattr(precursor_node, "base_model_id"):
            model_id = precursor_node.base_model_id

        if not model_id:
            incomers = param_graph.get_incomers(precursor_node.id)
            for inc in incomers:
                if isinstance(inc, Model):
                    model_id = inc.id
                    break

        if not model_id:
            node_name = getattr(precursor_node, "name", precursor_node_id)
            return jsonify({
                "error": f"Precursor node '{node_name}' has no associated generative model and cannot be wrapped as an Individual."
            }), 400

        model_element = param_graph.get_element(model_id)
        if not isinstance(model_element, Model):
            return jsonify({"error": f"Associated model '{model_id}' was not found in the project graph."}), 400

        # 3. Resolve baseline grating / linear layer elements
        baseline_grating_id = data.get("baseline_grating_id") or params.get("baseline_grating_id")
        elements_input = data.get("elements") or params.get("elements")
        lora_rank = int(data.get("lora_rank") or params.get("lora_rank") or 1)
        lora_alpha = float(data.get("lora_alpha") or params.get("lora_alpha") or 1.0)

    # If grating node exists
    if baseline_grating_id:
        with graph_lock:
            baseline_grating = param_graph.get_element(baseline_grating_id)
            if not isinstance(baseline_grating, Grating):
                return jsonify({"error": f"Node '{baseline_grating_id}' is not a valid grating."}), 400
            base_grating_path = baseline_grating.file.path
            baseline_elements = baseline_grating.elements
            baseline_name = baseline_grating.name
            diff_base_grating = DiffractureGrating.load(base_grating_path)
    elif isinstance(precursor_node, Grating):
        base_grating_path = precursor_node.file.path
        baseline_elements = precursor_node.elements
        baseline_name = precursor_node.name
        diff_base_grating = DiffractureGrating.load(base_grating_path)
        baseline_grating_id = precursor_node.id
    else:
        # If not provided, inspect model linear layers
        if not elements_input:
            engine = engine_provider.get_engine()
            layers = await engine.get_model_layers(model_element)
            linear_layers = [l for l in layers if "Linear" in l.get("type", "")]
            if not linear_layers:
                return jsonify({"error": f"No compatible Linear layers found on model '{model_id}' for LoRA genome."}), 400

            elements_input = [
                {
                    "address": l["address"],
                    "kernel_type": "lora",
                    "params": {
                        "rank": lora_rank,
                        "alpha": lora_alpha,
                        "in_features": l.get("in_features", 0) or 0,
                        "out_features": l.get("out_features", 0) or 0,
                        "kernel_size": None
                    },
                    "indices": [],
                    "perform_clustering": False,
                    "num_clusters": None,
                    "cluster": None
                }
                for l in linear_layers
            ]

        engine = engine_provider.get_engine()
        grating_name = f"baseline_{uuid.uuid4().hex[:8]}"
        baseline_grating = await engine.create_grating(model_element, grating_name, elements_input)
        base_grating_path = baseline_grating.file.path
        baseline_elements = baseline_grating.elements
        baseline_name = precursor_node.name or precursor_node.alias or "Artifact"
        diff_base_grating = DiffractureGrating.load(base_grating_path)

    # Build baseline genome from base grating
    from evolution.genome import LoRAGenome
    baseline_genome = LoRAGenome.from_grating(diff_base_grating)

    # Compute deterministic UID from baseline genome
    genome_state_dict = baseline_genome.get_state_dict()
    genome_uid = uid_generator.from_state_dict(genome_state_dict)
    individual_id = f"individual_{genome_uid}"

    output_dir = param_graph.root / "generate"
    os.makedirs(output_dir, exist_ok=True)
    genome_path = output_dir / f"{individual_id}.safetensors"
    baseline_genome.save(str(genome_path))

    # Prepare context
    ind_context = copy.deepcopy(getattr(precursor_node, "context", {}) or {})
    ind_context["model_id"] = model_id
    ind_context["baseline_elements"] = baseline_elements
    ind_context["baseline_file_path"] = str(base_grating_path)
    ind_context["precursor_artifact_id"] = precursor_node_id

    slug = generate_slug(2)
    custom_name = data.get("name") or params.get("name") or f"{baseline_name} - Baseline ({slug})"

    individual_node = Individual(
        id=individual_id,
        name=custom_name,
        file=Asset(path=str(genome_path), uid=individual_id, extension=".safetensors"),
        base_model_id=model_id,
        baseline_grating_id=baseline_grating_id,
        generation=0,
        context=ind_context
    )

    with graph_lock:
        param_graph.add_element(individual_node)
        param_graph.link(model_element, individual_node, relation='binds_to')
        param_graph.link(precursor_node, individual_node, relation='precursor')
        
        # If the precursor is an audio artifact, set its parent to the individual so it serves as exemplar
        if isinstance(precursor_node, Audio):
            param_graph.update_element(precursor_node.id, {"parent": individual_id})

        param_graph.save()

    return jsonify({
        "success": True,
        "message": f"Artifact '{precursor_node_id}' wrapped as Individual '{individual_id}'",
        "individual": individual_node.to_dict(),
        "node_id": individual_id
    }), 200


@app.route("/mutate_evolution", methods=["POST"])
@app.route("/start_evolution", methods=["POST"])
async def mutate_evolution():
    """
    Mutates parent individuals to produce variant offspring.
    Genetics runs 100% locally on CPU without remote engine dependencies.
    """
    if param_graph is None or engine_provider is None:
        return jsonify({"error": "No project loaded"}), 400

    try:
        data = request.get_json() or {}
        return await _dispatch_mutate_operation(data)
    except Exception as e:
        print(f"Failed to start mutation: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500


start_evolution = mutate_evolution


def _run_mutate_task(
    parent_job_id: str,
    parent_ids: list[str],
    offspring_size: int,
    lora_noise: float,
    active_flip_prob: float,
    mutation_rate: float,
    generation_context: dict,
    generate_exemplars: bool = False,
) -> None:
    try:
        _mutate_evolution_task(
            parent_job_id=parent_job_id,
            parent_ids=parent_ids,
            offspring_size=offspring_size,
            lora_noise=lora_noise,
            active_flip_prob=active_flip_prob,
            mutation_rate=mutation_rate,
            generation_context=generation_context,
            generate_exemplars=generate_exemplars,
        )
    except Exception as e:
        print(f"Failed in async evolution mutation: {e}")
        traceback.print_exc()
        local_jobs[parent_job_id]["status"] = "failed"
        local_jobs[parent_job_id]["progress"] = None
        local_jobs[parent_job_id]["error"] = str(e)
        local_jobs[parent_job_id]["traceback"] = traceback.format_exc()


def _mutate_evolution_task(
    parent_job_id: str,
    parent_ids: list[str],
    offspring_size: int,
    lora_noise: float,
    active_flip_prob: float,
    mutation_rate: float,
    generation_context: dict,
    generate_exemplars: bool = False,
) -> None:
    try:
        local_jobs[parent_job_id]["status"] = "running"
        local_jobs[parent_job_id]["progress"] = {
            "value": 0,
            "total": offspring_size,
            "description": "Starting mutation..."
        }

        from evolution.genome import LoRAGenome
        from evolution.lora_genome import get_lora_mutation_strategy
        from neutral_selection.variation.mutation import mutate

        with graph_lock:
            parent_nodes: list[Individual] = []
            for pid in parent_ids:
                if param_graph.G.has_node(pid):
                    node = param_graph.get_element(pid)
                    if isinstance(node, Individual) and node not in parent_nodes:
                        parent_nodes.append(node)

            if not parent_nodes:
                raise ValueError("No valid parent individuals found in graph.")

            first_parent = parent_nodes[0]
            model_id = first_parent.base_model_id
            baseline_grating_id = first_parent.baseline_grating_id
            baseline_elements = first_parent.context.get("baseline_elements")
            baseline_file_path = first_parent.context.get("baseline_file_path")

            model_element = param_graph.get_element(model_id)
            if not model_element:
                raise ValueError(f"Base model '{model_id}' not found in parameter graph.")

            # Load baseline Diffracture Grating if needed
            base_grating = None
            if baseline_grating_id:
                base_grating = param_graph.get_element(baseline_grating_id)
            elif baseline_file_path:
                base_grating = Grating(
                    id=f"grating_{uuid.uuid4().hex[:8]}",
                    name="Baseline Grating",
                    context={},
                    file=Asset(path=str(baseline_file_path), uid=f"grating_{uuid.uuid4().hex[:8]}", extension=".safetensors"),
                    base_model_id=model_id,
                    elements=baseline_elements or []
                )

            max_gen = max(getattr(p, "generation", 0) for p in parent_nodes)
            next_generation = max_gen + 1

            merged_context = copy.deepcopy(first_parent.context or {})
            if generation_context:
                merged_context.update(generation_context)

            operation = merged_context.get("operation", "generate")

            # Resolve source audio if applicable
            source_audio_id = (
                merged_context.get("source_audio_id")
                or merged_context.get("source_audio")
                or merged_context.get("precursor_audio_id")
            )
            node_engine_args = {}
            if source_audio_id:
                source_audio_element = param_graph.get_element(source_audio_id)
                if source_audio_element:
                    node_engine_args["source_audio_element"] = source_audio_element

            dumped_params = {}
            for k, v in merged_context.items():
                if isinstance(v, (str, int, float, bool)) and k not in [
                    "model_id", "operation", "gratings", "source_audio_id", "source_audio", "job_id",
                    "baseline_elements", "baseline_file_path", "lineage", "generate_exemplars", "auto_generate_exemplars"
                ]:
                    dumped_params[k] = v

            # 1. Create the Visual Compound Parent Group
            group_id = f"group_{uid_generator.from_string(str(uuid.uuid4()))}"
            group_alias = f"Mutation (Gen {next_generation})"
            if len(parent_nodes) == 1:
                group_alias = f"Mutants - {parent_nodes[0].name}"
            mutation_group = Group(
                id=group_id,
                member_ids=[],
                member_type='individual'
            )
            param_graph.add_element(mutation_group)
            param_graph.update_element(mutation_group.id, {"alias": group_alias})
            param_graph.save()

        mutation_strategy = get_lora_mutation_strategy(
            lora_noise=lora_noise,
            active_flip_prob=active_flip_prob,
            mutation_rate=mutation_rate
        )

        job_ids = []
        individual_ids = []

        for i in range(offspring_size):
            local_jobs[parent_job_id]["progress"] = {
                "value": i,
                "total": offspring_size,
                "description": f"Mutating individual {i+1} of {offspring_size}..."
            }

            # Select parent (round-robin across supplied parent individuals)
            selected_parent = parent_nodes[i % len(parent_nodes)]
            genome_path = param_graph.get_path_from_id(selected_parent.id) or selected_parent.file.path
            parent_genome = LoRAGenome.load(genome_path)

            child_genome = copy.deepcopy(parent_genome)
            child_genome = mutate(child_genome, mutation_strategy)
            if not isinstance(child_genome, LoRAGenome):
                child_genome = LoRAGenome(list(child_genome))

            genome_state_dict = child_genome.get_state_dict()
            genome_uid = uid_generator.from_state_dict(genome_state_dict)
            child_ind_id = f"individual_{genome_uid}"

            output_dir = param_graph.root / "generate"
            os.makedirs(output_dir, exist_ok=True)
            saved_genome_path = output_dir / f"{child_ind_id}.safetensors"
            child_genome.save(str(saved_genome_path))

            child_context = copy.deepcopy(merged_context)
            child_context["baseline_elements"] = baseline_elements
            child_context["baseline_file_path"] = str(baseline_file_path) if baseline_file_path else None
            child_context["mutation_operation"] = {
                "lora_noise": lora_noise,
                "active_flip_prob": active_flip_prob,
                "mutation_rate": mutation_rate,
                "parent_id": selected_parent.id,
                "generation": next_generation
            }
            child_context["lineage"] = {
                "parent_ids": [selected_parent.id],
                "crossover_applied": False,
                "mutated": True
            }

            slug = generate_slug(2)
            child_node = Individual(
                id=child_ind_id,
                name=f"Gen {next_generation} - Mut {i+1} ({slug})",
                file=Asset(path=str(saved_genome_path), uid=child_ind_id, extension=".safetensors"),
                base_model_id=model_id,
                baseline_grating_id=baseline_grating_id,
                generation=next_generation,
                context=child_context
            )

            with graph_lock:
                param_graph.add_element(child_node)
                param_graph.update_element(child_node.id, {"parent": group_id})
                param_graph.link(model_element, child_node, relation='binds_to')
                param_graph.link(selected_parent, child_node, relation='parent')

                mutation_group.member_ids.append(child_ind_id)
                param_graph.update_element(mutation_group.id, {"member_ids": mutation_group.member_ids})
                param_graph.save()

            individual_ids.append(child_ind_id)

            # Optional automatic exemplar generation
            if generate_exemplars and engine_provider is not None:
                try:
                    engine = engine_provider.get_engine()
                    sub_job_id = f"job_exemplar_{uuid.uuid4().hex[:8]}"

                    child_engine_args = dict(node_engine_args)
                    child_engine_args["model_element"] = model_element
                    child_engine_args["individual_elements"] = [child_node]
                    child_engine_args["individual_strengths"] = [1.0]
                    if base_grating:
                        child_engine_args["baseline_grating"] = base_grating

                    active_jobs[sub_job_id] = {
                        "parent_id": child_ind_id,
                        "group_id": None,
                        "linked_elements": [child_node],
                        "validated_params": {
                            **dumped_params,
                            "model_id": model_id,
                            "operation": operation,
                            "individuals": [{"id": child_ind_id, "strength": 1.0}]
                        },
                        "operation": operation
                    }

                    asyncio.run(engine.execute(operation, job_id=sub_job_id, **child_engine_args, **dumped_params))
                    job_ids.append(sub_job_id)
                    print(f"[_mutate_evolution_task] Queued exemplar generation job {sub_job_id} for individual {child_ind_id}")
                except Exception as ex:
                    print(f"[_mutate_evolution_task] Warning: Failed to queue exemplar generation for individual {child_ind_id}: {ex}")
                    traceback.print_exc()

        local_jobs[parent_job_id]["status"] = "completed"
        local_jobs[parent_job_id]["progress"] = {
            "value": offspring_size,
            "total": offspring_size,
            "description": "Mutation complete."
        }
        local_jobs[parent_job_id]["result"] = {
            "individual_ids": individual_ids,
            "group_id": group_id,
            "generation": next_generation,
            "job_ids": job_ids
        }
        print(f"[_mutate_evolution_task] Mutation completed successfully for job {parent_job_id}")

    except Exception as e:
        print(f"Failed in async evolution mutation: {e}")
        traceback.print_exc()
        local_jobs[parent_job_id]["status"] = "failed"
        local_jobs[parent_job_id]["progress"] = None
        local_jobs[parent_job_id]["error"] = str(e)
        local_jobs[parent_job_id]["traceback"] = traceback.format_exc()


_initialize_evolution_task = _mutate_evolution_task


def _extract_individual_parent_ids(parent_input) -> list[str]:
    """
    Extracts valid Individual node IDs from parent inputs, automatically
    unpacking groups/bundles and discarding edge IDs (e.g. containing '->') or invalid elements.
    """
    if not parent_input:
        return []
    if isinstance(parent_input, (str, dict)):
        items = [parent_input]
    elif isinstance(parent_input, list):
        items = parent_input
    else:
        return []

    resolved_ids: list[str] = []
    for item in items:
        if isinstance(item, dict):
            node_obj = item.get("node") if "node" in item else item
            if isinstance(node_obj, dict):
                if node_obj.get("source") and node_obj.get("target"):
                    continue
                pid = node_obj.get("id")
            elif isinstance(node_obj, str):
                pid = node_obj
            else:
                continue
        elif isinstance(item, str):
            pid = item
        else:
            continue

        if not pid or not isinstance(pid, str) or "->" in pid:
            continue

        with graph_lock:
            if param_graph and param_graph.G.has_node(pid):
                elem = param_graph.get_element(pid)
                if isinstance(elem, Individual) and pid not in resolved_ids:
                    resolved_ids.append(pid)
                elif isinstance(elem, (Group, Bundle)) and hasattr(elem, "member_ids") and elem.member_ids:
                    for mid in elem.member_ids:
                        if param_graph.G.has_node(mid):
                            m_elem = param_graph.get_element(mid)
                            if isinstance(m_elem, Individual) and mid not in resolved_ids:
                                resolved_ids.append(mid)
    return resolved_ids


@app.route("/recombine_evolution", methods=["POST"])
@app.route("/reproduce_evolution", methods=["POST"])
async def recombine_evolution():
    """
    Starts an evolutionary recombination (breeding) run.
    Selects, crosses over, and mutates parent individuals to produce a new generation.
    Genetics runs 100% locally on CPU without remote engine dependencies.
    """
    if param_graph is None or engine_provider is None:
        return jsonify({"error": "No project loaded"}), 400

    try:
        data = request.get_json() or {}
        return await _dispatch_recombine_operation(data)
    except Exception as e:
        print(f"Failed to start recombination: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500


reproduce_evolution = recombine_evolution


def _run_recombine_task(
    parent_job_id: str,
    parent_ids: list[str] | None,
    parent_bundle_id: str | None,
    offspring_size: int | None,
    selection_cfg: dict,
    crossover_cfg: dict,
    mutation_cfg: dict,
    crossover_prob: float,
    elitism: int,
    generation_context: dict,
    default_fitness: float = 0.0,
    generate_exemplars: bool = False,
) -> None:
    try:
        _recombine_evolution_task(
            parent_job_id=parent_job_id,
            parent_ids=parent_ids,
            parent_bundle_id=parent_bundle_id,
            offspring_size=offspring_size,
            selection_cfg=selection_cfg,
            crossover_cfg=crossover_cfg,
            mutation_cfg=mutation_cfg,
            crossover_prob=crossover_prob,
            elitism=elitism,
            generation_context=generation_context,
            default_fitness=default_fitness,
            generate_exemplars=generate_exemplars,
        )
    except Exception as e:
        print(f"Failed in async evolution reproduction: {e}")
        traceback.print_exc()
        local_jobs[parent_job_id]["status"] = "failed"
        local_jobs[parent_job_id]["progress"] = None
        local_jobs[parent_job_id]["error"] = str(e)
        local_jobs[parent_job_id]["traceback"] = traceback.format_exc()


def _recombine_evolution_task(
    parent_job_id: str,
    parent_ids: list[str] | None,
    parent_bundle_id: str | None,
    offspring_size: int | None,
    selection_cfg: dict,
    crossover_cfg: dict,
    mutation_cfg: dict,
    crossover_prob: float,
    elitism: int,
    generation_context: dict,
    default_fitness: float = 0.0,
    generate_exemplars: bool = False,
) -> None:
    """
    Background worker that breeds a new generation from parent individuals,
    creates child nodes in the parameter graph with lineage relationships,
    and groups offspring in a generation Group.
    """
    try:
        local_jobs[parent_job_id]["status"] = "running"
        print(f"[_reproduce_evolution_task] Started reproduction task for job {parent_job_id}")

        with graph_lock:
            # 1. Resolve parent individual nodes
            parent_nodes: list[Individual] = []
            if parent_ids:
                for pid in parent_ids:
                    if param_graph.G.has_node(pid):
                        node = param_graph.get_element(pid)
                        if isinstance(node, Individual) and node not in parent_nodes:
                            parent_nodes.append(node)
            elif parent_bundle_id:
                bundle_node = param_graph.get_element(parent_bundle_id)
                if isinstance(bundle_node, Bundle):
                    for mid in bundle_node.member_ids:
                        if param_graph.G.has_node(mid):
                            node = param_graph.get_element(mid)
                            if isinstance(node, Individual) and node not in parent_nodes:
                                parent_nodes.append(node)

            if not parent_nodes:
                raise ValueError("No valid parent individuals could be resolved.")

            first_parent = parent_nodes[0]
            model_id = first_parent.base_model_id
            baseline_grating_id = first_parent.baseline_grating_id
            baseline_elements = first_parent.context.get("baseline_elements")
            baseline_file_path = first_parent.context.get("baseline_file_path")

            # Precursor baseline grating resolution
            base_grating = None
            if baseline_grating_id:
                base_grating = param_graph.get_element(baseline_grating_id)
            elif baseline_file_path:
                base_grating = Grating(
                    id=f"grating_{uuid.uuid4().hex[:8]}",
                    name="Baseline Grating",
                    context={},
                    file=Asset(path=str(baseline_file_path), uid=f"grating_{uuid.uuid4().hex[:8]}", extension=".safetensors"),
                    base_model_id=model_id,
                    elements=baseline_elements or []
                )

            model_element = param_graph.get_element(model_id)
            if not model_element:
                raise ValueError(f"Base model '{model_id}' not found in parameter graph.")

            # Calculate generation index
            max_gen = max(getattr(p, "generation", 0) for p in parent_nodes)
            next_generation = max_gen + 1

            # Determine offspring count
            target_offspring_count = offspring_size if (offspring_size and offspring_size > 0) else len(parent_nodes)

            # Resolve generation context and arguments for execution
            merged_context = copy.deepcopy(first_parent.context or {})
            if generation_context:
                merged_context.update(generation_context)

            operation = merged_context.get("operation", "generate")
            source_audio_id = (
                merged_context.get("source_audio_id")
                or merged_context.get("source_audio")
                or merged_context.get("precursor_audio_id")
            )
            if not source_audio_id:
                for p in parent_nodes:
                    for u, _, _ in param_graph.G.in_edges(p.id, data=True):
                        node_u = param_graph.get_element(u)
                        if isinstance(node_u, Audio):
                            source_audio_id = u
                            break
                    if source_audio_id:
                        break
                    for node_id, attrs in param_graph.G.nodes.items():
                        if attrs.get("parent") == p.id:
                            node_child = param_graph.get_element(node_id)
                            if isinstance(node_child, Audio):
                                source_audio_id = node_id
                                break
                    if source_audio_id:
                        break

            node_engine_args = {}
            source_audio_element = None
            if source_audio_id:
                source_audio_element = param_graph.get_element(source_audio_id)
                if source_audio_element:
                    node_engine_args["source_audio_element"] = source_audio_element

            # Dumped parameters for the engine
            dumped_params = {}
            for k, v in merged_context.items():
                if isinstance(v, (str, int, float, bool)) and k not in [
                    "model_id", "operation", "gratings", "source_audio_id", "source_audio", "job_id",
                    "baseline_elements", "baseline_file_path", "lineage", "generate_exemplars", "auto_generate_exemplars"
                ]:
                    dumped_params[k] = v

            # 2. Load parent genomes into neutral_selection Individuals
            parent_ns_individuals: list[NSIndividual] = []
            parent_id_list: list[str] = []
            for p_node in parent_nodes:
                genome_path = param_graph.get_path_from_id(p_node.id) or p_node.file.path
                genome = LoRAGenome.load(genome_path)
                ns_ind = NSIndividual(genotype=genome)
                fitness_val = p_node.fitness if p_node.fitness is not None else default_fitness
                ns_ind.fitness = fitness_val
                parent_ns_individuals.append(ns_ind)
                parent_id_list.append(p_node.id)

            # 3. Build reproduction strategies
            if isinstance(selection_cfg, dict) and selection_cfg.get("type") in ("uniform", "random"):
                from neutral_selection.variation.selection import RandomSelection
                selection_strat = RandomSelection()
            else:
                selection_strat = build_selection_strategy(selection_cfg)

            crossover_type_name = crossover_cfg.get("type", "hierarchical") if isinstance(crossover_cfg, dict) else str(crossover_cfg)
            crossover_strat = get_lora_crossover_strategy(strategy_type=crossover_type_name)

            lora_noise = float(mutation_cfg.get("lora_noise", merged_context.get("lora_noise", 0.05)))
            active_flip_prob = float(mutation_cfg.get("active_flip_prob", merged_context.get("active_flip_prob", 0.05)))
            mutation_rate = float(mutation_cfg.get("mutation_rate", 0.1))
            mutation_strat = get_lora_mutation_strategy(
                lora_noise=lora_noise,
                active_flip_prob=active_flip_prob,
                mutation_rate=mutation_rate
            )

            # 4. Perform breeding
            offspring_records: list[RecombinedOffspring] = recombine_offspring(
                parents=parent_ns_individuals,
                offspring_count=target_offspring_count,
                selection_strategy=selection_strat,
                crossover_strategy=crossover_strat,
                mutation_strategy=mutation_strat,
                crossover_prob=crossover_prob,
                elitism=elitism,
                parent_ids=parent_id_list,
            )

            # 5. Create Generation Group Node (Visual Container)
            gen_group_id = f"group_{uid_generator.from_string(str(uuid.uuid4()))}"
            gen_group = Group(
                id=gen_group_id,
                member_ids=[],
                member_type='individual'
            )

            param_graph.add_element(gen_group)
            param_graph.update_element(gen_group.id, {"alias": f"Recombination (Gen {next_generation})"})
            param_graph.save()

        job_ids = []
        individual_ids = []

        # 6. Process each offspring individual
        for i, record in enumerate(offspring_records):
            local_jobs[parent_job_id]["progress"] = {
                "value": i,
                "total": target_offspring_count,
                "description": f"Processing offspring {i+1} of {target_offspring_count}..."
            }

            child_genome = record.individual.genotype
            genome_state_dict = child_genome.get_state_dict()
            genome_uid = uid_generator.from_state_dict(genome_state_dict)
            child_ind_id = f"individual_{genome_uid}"

            output_dir = param_graph.root / "generate"
            os.makedirs(output_dir, exist_ok=True)
            child_genome_path = output_dir / f"{child_ind_id}.safetensors"
            child_genome.save(str(child_genome_path))

            child_context = copy.deepcopy(merged_context)
            child_context["baseline_elements"] = baseline_elements
            child_context["baseline_file_path"] = str(baseline_file_path) if baseline_file_path else None
            child_context["recombine_operation"] = {
                "crossover_type": crossover_cfg.get("type", "hierarchical") if isinstance(crossover_cfg, dict) else str(crossover_cfg),
                "selection_type": selection_cfg.get("type", "tournament") if isinstance(selection_cfg, dict) else str(selection_cfg),
                "default_fitness": default_fitness,
                "mutation_rate": mutation_rate,
                "crossover_prob": crossover_prob,
                "elitism": elitism,
                "selection_cfg": selection_cfg,
                "crossover_cfg": crossover_cfg,
                "mutation_cfg": mutation_cfg,
                "offspring_size": target_offspring_count,
                "parent_ids": parent_id_list,
                "model_id": model_id,
                "baseline_grating_id": baseline_grating_id,
                "generation": next_generation
            }
            child_context["lineage"] = {
                "parent_ids": record.lineage.parent_ids,
                "crossover_applied": record.lineage.crossover_applied,
                "mutated": record.lineage.mutated
            }

            slug = generate_slug(2)
            child_node = Individual(
                id=child_ind_id,
                name=f"Gen {next_generation} - Ind {i+1} ({slug})",
                file=Asset(path=str(child_genome_path), uid=child_ind_id, extension=".safetensors"),
                base_model_id=model_id,
                baseline_grating_id=baseline_grating_id,
                generation=next_generation,
                fitness=record.individual.fitness,
                context=child_context
            )

            with graph_lock:
                param_graph.add_element(child_node)
                param_graph.update_element(child_node.id, {"parent": gen_group_id})
                param_graph.link(model_element, child_node, relation='binds_to')

                for pid in record.lineage.parent_ids:
                    p_node_elem = param_graph.get_element(pid)
                    if p_node_elem:
                        param_graph.link(p_node_elem, child_node, relation='parent')

                gen_group.member_ids.append(child_ind_id)
                param_graph.update_element(gen_group.id, {"member_ids": gen_group.member_ids})
                param_graph.save()

            individual_ids.append(child_ind_id)

            # Optional automatic exemplar generation
            if generate_exemplars and engine_provider is not None:
                try:
                    engine = engine_provider.get_engine()
                    sub_job_id = f"job_exemplar_{uuid.uuid4().hex[:8]}"

                    child_engine_args = dict(node_engine_args)
                    child_engine_args["model_element"] = model_element
                    child_engine_args["individual_elements"] = [child_node]
                    child_engine_args["individual_strengths"] = [1.0]
                    if base_grating:
                        child_engine_args["baseline_grating"] = base_grating

                    active_jobs[sub_job_id] = {
                        "parent_id": child_ind_id,
                        "group_id": None,
                        "linked_elements": [child_node],
                        "validated_params": {
                            **dumped_params,
                            "model_id": model_id,
                            "operation": operation,
                            "individuals": [{"id": child_ind_id, "strength": 1.0}]
                        },
                        "operation": operation
                    }

                    asyncio.run(engine.execute(operation, job_id=sub_job_id, **child_engine_args, **dumped_params))
                    job_ids.append(sub_job_id)
                    print(f"[_recombine_evolution_task] Queued exemplar generation job {sub_job_id} for individual {child_ind_id}")
                except Exception as ex:
                    print(f"[_recombine_evolution_task] Warning: Failed to queue exemplar generation for individual {child_ind_id}: {ex}")
                    traceback.print_exc()

        local_jobs[parent_job_id]["status"] = "completed"
        local_jobs[parent_job_id]["progress"] = {
            "value": target_offspring_count,
            "total": target_offspring_count,
            "description": "Recombination complete."
        }
        local_jobs[parent_job_id]["result"] = {
            "individual_ids": individual_ids,
            "group_id": gen_group_id,
            "generation": next_generation,
            "job_ids": job_ids
        }
        print(f"[_recombine_evolution_task] Evolution recombination completed successfully for job {parent_job_id}")

    except Exception as e:
        print(f"Failed in async evolution reproduction: {e}")
        traceback.print_exc()
        local_jobs[parent_job_id]["status"] = "failed"
        local_jobs[parent_job_id]["progress"] = None
        local_jobs[parent_job_id]["error"] = str(e)
        local_jobs[parent_job_id]["traceback"] = traceback.format_exc()


_reproduce_evolution_task = _recombine_evolution_task


@app.route("/express_individual", methods=["POST"])
async def express_individual():
    """
    Expresses the genome of an individual to a full Grating node in the parameter graph,
    linking it as a child of the individual.
    """
    if param_graph is None:
        return jsonify({"error": "No project loaded"}), 400

    try:
        data = request.get_json() or {}
        individual_id = data.get("individual_id")

        if not individual_id:
            return jsonify({"error": "individual_id is required"}), 400

        with graph_lock:
            individual_node = param_graph.get_element(individual_id)
            if not isinstance(individual_node, Individual):
                return jsonify({"error": f"Node '{individual_id}' is not a valid individual."}), 400

            baseline_grating_id = individual_node.baseline_grating_id
            baseline_elements = individual_node.context.get("baseline_elements")
            baseline_file_path = individual_node.context.get("baseline_file_path")

            if baseline_grating_id:
                baseline_grating_node = param_graph.get_element(baseline_grating_id)
                if not isinstance(baseline_grating_node, Grating):
                    return jsonify({"error": f"Baseline grating '{baseline_grating_id}' not found."}), 400
                base_elements = baseline_grating_node.elements
                base_path = baseline_grating_node.file.path
            elif baseline_elements and baseline_file_path:
                base_elements = baseline_elements
                base_path = baseline_file_path
            else:
                return jsonify({"error": "Unable to resolve baseline grating configuration for this individual."}), 400

            # Load LoRAGenome from individual file
            genome = LoRAGenome.load(individual_node.file.path)

            # Express to new Diffracture Grating
            expressed_diff_grating = express_to_grating(genome, base_path)

            grating_id = f"grating_{uid_generator.from_string(str(uuid.uuid4()))}"
            expressed_grating_path = param_graph.root / "generate" / f"{grating_id}.safetensors"
            expressed_diff_grating.save(str(expressed_grating_path))

            # Create Grating Artifact
            grating_artifact = Grating(
                id=grating_id,
                name=f"Expressed {individual_node.name}",
                file=Asset(path=str(expressed_grating_path), uid=grating_id, extension=".safetensors"),
                base_model_id=individual_node.base_model_id,
                elements=base_elements,
                context=individual_node.context or {}
            )

            param_graph.add_element(grating_artifact)
            param_graph.update_element(grating_artifact.id, {"parent": individual_id})

            model_element = param_graph.get_element(individual_node.base_model_id)
            param_graph.link(model_element, grating_artifact, relation='binds_to')
            param_graph.link(individual_node, grating_artifact, relation='expressed_to')
            
            param_graph.save()

        return jsonify({
            "success": True,
            "message": "Individual expressed to grating successfully",
            "grating": grating_artifact.to_dict()
        }), 200

    except Exception as e:
        print(f"Failed to express individual: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500

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
            # Mark completed timestamp and clean up jobs older than 5 minutes to prevent leaking
            job_info["completed_at"] = job_info.get("completed_at", time.time())
            now = time.time()
            expired_keys = [k for k, v in local_jobs.items() if v.get("completed_at") and (now - v["completed_at"] > 300)]
            for k in expired_keys:
                local_jobs.pop(k, None)
            return jsonify(job_info), 200 if status == "completed" else 500
        return jsonify(job_info), 200

    try:
        engine = engine_provider.get_engine()
        
        status_info = await engine.get_job_status(job_id)
        status = status_info.get("status")

        if status in ["completed", "failed", "not_found"]:
            job_context = active_jobs.pop(job_id, {})

            if status == "completed":
                print(f"Job {job_id} completed. Processing artifact...")
                result_dict = status_info.get("result", {})
                artifact_data = result_dict.get('artifact', result_dict)

                if not artifact_data:
                    raise Exception("Completed job did not return a valid artifact.")

                temp_artifact = resolve_element(artifact_data) if isinstance(artifact_data, dict) else artifact_data

                output_dir = param_graph.root / "generate"
                final_artifact = save_artifact_asset(temp_artifact, output_dir, asset_name="file")
                
                # Ensure context is populated for labelling, and merge job-level validated params (like model_id, operation, gratings)
                if isinstance(final_artifact, Artifact):
                    current_context = final_artifact.context or {}
                    merged_context = {**job_context.get("validated_params", {}), **current_context}
                    final_artifact = replace(final_artifact, context=merged_context)
                
                with graph_lock:
                    param_graph.add_element(final_artifact)

                    parent_id = job_context.get("parent_id")
                    group_id = job_context.get("group_id") or job_context.get("batch_id")
                    if parent_id:
                        param_graph.update_element(final_artifact.id, {"parent": parent_id})
                    elif group_id:
                        param_graph.update_element(final_artifact.id, {"parent": group_id})
                        group_node_attrs = param_graph.G.nodes[group_id]
                        if 'member_ids' not in group_node_attrs or not isinstance(group_node_attrs['member_ids'], list):
                            group_node_attrs['member_ids'] = []
                        if final_artifact.id not in group_node_attrs['member_ids']:
                            group_node_attrs['member_ids'].append(final_artifact.id)
                            
                        update_group_labels(group_id)

                    for element in job_context.get("linked_elements", []):
                        if parent_id and element.id == parent_id:
                            continue
                        print(f"Linking {element.id} to {final_artifact.id}")
                        param_graph.link(element, final_artifact, relation='source')
                    param_graph.save()
                
                trigger_embedding_update()
                
                print("Artifact processed and saved to graph successfully.")
                return jsonify({
                    "status": "completed",
                    "message": "Audio generated and registered successfully.",
                    "artifact": final_artifact.to_dict(),
                    "node_id": final_artifact.id,
                    "validated_params": job_context.get("validated_params")
                }), 200

            elif status == "failed":
                error_msg = status_info.get("error", "Unknown error during generation.")
                traceback_msg = status_info.get("traceback")
                print(f"Error: Job {job_id} failed: {error_msg}")
                if traceback_msg:
                    print(f"Traceback:\n{traceback_msg}")
                return jsonify({"status": "failed", "error": error_msg, "traceback": traceback_msg}), 500

            elif status == "not_found":
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
        active_jobs.pop(job_id, None)
            
        return jsonify({
            "message": f"Cancellation requested for job {job_id}.",
            "success": True
        }), 202
    except Exception as e:
        print(f"Failed to cancel job: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/sync_operations", methods=["GET"])
def get_sync_operations():
    """Returns a list of available Synchronous operations and their configurations."""
    try:
        operations = [op.to_dict() for op in sync_registry.get_all()]
        return jsonify({"operations": operations, "success": True}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/operations", methods=["GET"])
async def get_all_operations():
    """Returns a unified list of operations across DSP, Evolution, and Generative Engine."""
    try:
        operations = []

        # 1. Local DSP Operations (immediate)
        for op in sync_registry.get_all():
            operations.append(op.to_dict())

        # 2. Local Evolutionary Operations (queued)
        for op in get_evolution_operations():
            operations.append(dict(op))

        # 3. Neural Engine Generative Operations (queued)
        if engine_provider is not None:
            engine = engine_provider.get_engine()
            if engine is not None:
                async_ops = await engine.get_supported_operations()
                for op in async_ops:
                    op_dict = dict(op)
                    if "category" not in op_dict:
                        op_dict["category"] = "generative"
                    if "execution" not in op_dict:
                        op_dict["execution"] = "queued"
                    if "execution_mode" not in op_dict:
                        op_dict["execution_mode"] = "async"
                    operations.append(op_dict)

        # Deduplicate by operation name, preserving local priority
        seen = set()
        deduped_operations = []
        for op in operations:
            if op["name"] not in seen:
                seen.add(op["name"])
                deduped_operations.append(op)

        return jsonify({"operations": deduped_operations, "success": True}), 200
    except Exception as e:
        print(f"Error getting operations: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/execute_operation", methods=["POST"])
async def execute_operation():
    """
    Unified Operation Dispatcher Endpoint.
    Auto-dispatches operations based on operation identity and registry membership.
    """
    if param_graph is None:
        return jsonify({"error": "No project loaded"}), 400
        
    try:
        payload = request.get_json() or {}
        operation = payload.get("operation")
        if not operation:
            return jsonify({"error": "'operation' is required"}), 400

        # 1. DSP / Immediate Synchronous Operations
        if sync_registry.has(operation):
            return _dispatch_sync_operation(payload)

        # 2. Evolutionary Operations
        if operation in ("wrap_individual", "create_individual"):
            return await _dispatch_wrap_individual_operation(payload)
        elif operation == "mutate":
            return await _dispatch_mutate_operation(payload)
        elif operation in ("recombine", "reproduce"):
            return await _dispatch_recombine_operation(payload)

        # 3. Generative Engine Operations
        return await _dispatch_async_operation(payload)
            
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500

def _dispatch_sync_operation(data):
    try:
        operation = data.get("operation")
        params = data.get("params", {})
        
        if not operation:
            return jsonify({"error": "'operation' is required"}), 400
            
        op_instance = sync_registry.get(operation)
        DynamicArgsModel = create_dynamic_model(op_instance.get_form_config())
        
        # Support fallback where frontend still sends params at top level
        validation_target = params if params else data
        validated_params = DynamicArgsModel.model_validate(validation_target).model_dump()
        
        # 1. Dynamically resolve all requested node links from the graph
        node_args = {}
        source_elements = []
        for field in op_instance.get_form_config():
            if field.get("type") == "node":
                field_name = field.get("name")
                node_id = validated_params.pop(field_name, None)
                if node_id:
                    element = param_graph.get_element(node_id)
                    if element:
                        node_args[f"{field_name}_element"] = element
                        source_elements.append(element)
                        
        # Resolving and caching missing local audio
        for arg_name, element in node_args.items():
            if isinstance(element, Audio):
                cache_used_audio(element.id)
                valid_path = resolve_audio_path(element.id)
                if valid_path and str(valid_path) != element.file.path:
                    element.file = replace(element.file, path=str(valid_path))
                    node_args[arg_name] = element
        
        process_results = op_instance.execute(
            device=device_accelerator, 
            sample_rate=APP_SAMPLE_RATE, 
            **node_args, 
            **validated_params
        )
        is_batch = len(process_results) > 1
        process_results_to_save = process_results
            
        output_dir = param_graph.root / "process"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        final_artifacts = []
        
        # 3. Agnostic artifact data saving based strictly on returned element Types
        for artifact_blueprint, raw_data in process_results_to_save:
            local_path = data_cache_root / path_from_uid(artifact_blueprint.id)
            local_path.parent.mkdir(parents=True, exist_ok=True)
            
            if isinstance(artifact_blueprint, Audio):
                save_audio(raw_data.cpu(), local_path, artifact_blueprint.sample_rate, format="wav")
            elif isinstance(artifact_blueprint, Latent):
                torch.save(raw_data.cpu(), local_path)
            
            artifact_blueprint.file = replace(artifact_blueprint.file, path=str(local_path))
            final_artifacts.append(save_artifact_asset(artifact_blueprint, output_dir, asset_name="file"))

        # 4. Integrate into the graph
        collection_dict = None
        req_group_id = data.get("group_id") or data.get("batch_id") or params.get("group_id") or params.get("batch_id")
        with graph_lock:
            for artifact in final_artifacts:
                param_graph.add_element(artifact)
                
                for source_el in source_elements:
                    param_graph.link(source_el, artifact, relation='source')
                
            if req_group_id:
                if not param_graph.G.has_node(req_group_id):
                    # Create new group element
                    group_node = Group(id=req_group_id, member_ids=[], member_type=final_artifacts[0].type)
                    param_graph.add_element(group_node)
                    
                    # Try to position it near one of the source elements
                    for el in source_elements:
                        if param_graph.G.has_node(el.id):
                            el_pos = param_graph.G.nodes[el.id].get('position')
                            if el_pos:
                                param_graph.update_element(req_group_id, {"position": {"x": el_pos["x"] + 80, "y": el_pos["y"] + 80}})
                                break
                    print(f"Created new group element {req_group_id} for sync operation")
                
                for artifact in final_artifacts:
                    param_graph.update_element(artifact.id, {"parent": req_group_id})
                    group_node_attrs = param_graph.G.nodes[req_group_id]
                    if 'member_ids' not in group_node_attrs or not isinstance(group_node_attrs['member_ids'], list):
                        group_node_attrs['member_ids'] = []
                    if artifact.id not in group_node_attrs['member_ids']:
                        group_node_attrs['member_ids'].append(artifact.id)
                
                update_group_labels(req_group_id)
                collection_dict = param_graph.get_element(req_group_id).to_dict()
            elif is_batch:
                member_ids = [a.id for a in final_artifacts]
                group_id = uid_generator.from_uids(member_ids)
                group = Group(id=group_id, member_ids=member_ids, member_type=final_artifacts[0].type)
                param_graph.add_element(group)
                
                # Try to position it near one of the source elements
                for el in source_elements:
                    if param_graph.G.has_node(el.id):
                        el_pos = param_graph.G.nodes[el.id].get('position')
                        if el_pos:
                            param_graph.update_element(group_id, {"position": {"x": el_pos["x"] + 80, "y": el_pos["y"] + 80}})
                            break
                
                for m_id in member_ids:
                    param_graph.update_element(m_id, {"parent": group_id})
                    
                update_group_labels(group_id)
                collection_dict = param_graph.get_element(group_id).to_dict()

            param_graph.save()
            
        trigger_embedding_update(background=True)
        
        response_data = {
            "message": f"Operation '{operation}' completed successfully.",
            "status": "completed"
        }
        
        if req_group_id:
            response_data["artifact"] = final_artifacts[0].to_dict()
            response_data["node_id"] = final_artifacts[0].id
            if collection_dict:
                response_data["collection"] = collection_dict
        elif is_batch:
            response_data["artifacts"] = [a.to_dict() for a in final_artifacts]
            if collection_dict:
                response_data["collection"] = collection_dict
                response_data["node_id"] = collection_dict["id"]
        else:
            response_data["artifact"] = final_artifacts[0].to_dict()
            response_data["node_id"] = final_artifacts[0].id
            
        return jsonify(response_data), 200
        
    except (ValidationError, ValueError) as e:
        return jsonify({"error": "Invalid request", "details": str(e)}), 400
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500

async def _dispatch_mutate_operation(data):
    """
    Handles the 'mutate' async operation dispatched via /execute_operation or /mutate_evolution.
    Mutates parent individual(s) into variant offspring.
    """
    if param_graph is None or engine_provider is None:
        return jsonify({"error": "No project loaded"}), 400

    try:
        params = data.get("params", {}) or {}
        initiator = data.get("initiator") or params.get("initiator") or {}
        initiator_id = initiator.get("id") if isinstance(initiator, dict) else (str(initiator).strip() if initiator else None)
        initiator_type = initiator.get("type") if isinstance(initiator, dict) else None

        raw_parent_input = (
            data.get("parent_ids")
            or params.get("parent_ids")
            or data.get("parents")
            or params.get("parents")
            or data.get("individual_id")
            or params.get("individual_id")
            or data.get("target_node")
            or params.get("target_node")
        )
        parent_ids = _extract_individual_parent_ids(raw_parent_input)

        parent_group_id = (
            data.get("parent_group_id")
            or params.get("parent_group_id")
            or (initiator_id if initiator_type == "group" else None)
        )

        if not parent_ids and parent_group_id:
            with graph_lock:
                grp = param_graph.get_element(parent_group_id)
                if grp and hasattr(grp, "member_ids") and grp.member_ids:
                    parent_ids = _extract_individual_parent_ids(grp.member_ids)

        if not parent_ids and initiator_id and initiator_type == "individual":
            parent_ids = _extract_individual_parent_ids([initiator_id])

        # Backward compatibility fallback: if an audio precursor is sent to mutate, wrap it into an individual first
        if not parent_ids:
            precursor_val = (
                data.get("source_audio")
                or data.get("source_audio_id")
                or data.get("precursor_audio_id")
                or data.get("source_node")
                or data.get("source_node_id")
                or (initiator_id if initiator_type == "audio" else None)
            )
            if precursor_val:
                wrap_resp, wrap_code = await _dispatch_wrap_individual_operation(data)
                if wrap_code == 200:
                    wrap_json = wrap_resp.get_json()
                    parent_ids = [wrap_json["node_id"]]
                else:
                    return wrap_resp, wrap_code

        if not parent_ids:
            return jsonify({"error": "Either parent_ids, parents, or an Individual/Group initiator must contain valid individual nodes."}), 400

        offspring_size = int(
            data.get("offspring_size")
            or params.get("offspring_size")
            or data.get("population_size")
            or params.get("population_size")
            or 5
        )
        lora_noise = float(
            data.get("lora_noise")
            or params.get("lora_noise")
            or data.get("lora_down_noise")
            or 0.05
        )
        active_flip_prob = float(
            data.get("active_flip_prob")
            or params.get("active_flip_prob")
            or 0.05
        )
        mutation_rate = float(
            data.get("mutation_rate")
            or params.get("mutation_rate")
            or 1.0
        )
        generation_context = data.get("generation_context") or params.get("generation_context") or {}
        merged_generation_context = {**params, **generation_context}

        generate_exemplars = (
            data.get("generate_exemplars")
            if data.get("generate_exemplars") is not None
            else (
                params.get("generate_exemplars")
                if params.get("generate_exemplars") is not None
                else (
                    data.get("auto_generate_exemplars")
                    if data.get("auto_generate_exemplars") is not None
                    else params.get("auto_generate_exemplars", False)
                )
            )
        )
        if isinstance(generate_exemplars, str):
            generate_exemplars = generate_exemplars.lower() in ("true", "1", "yes")
        else:
            generate_exemplars = bool(generate_exemplars)

        parent_job_id = data.get("job_id") or f"evolution_mutate_{uuid.uuid4().hex[:12]}"
        local_jobs[parent_job_id] = {
            "status": "pending",
            "progress": None,
            "result": None,
            "error": None
        }

        # Spawn background task
        thread = threading.Thread(
            target=_run_mutate_task,
            kwargs=dict(
                parent_job_id=parent_job_id,
                parent_ids=parent_ids,
                offspring_size=offspring_size,
                lora_noise=lora_noise,
                active_flip_prob=active_flip_prob,
                mutation_rate=mutation_rate,
                generation_context=merged_generation_context,
                generate_exemplars=generate_exemplars,
            ),
            daemon=True
        )
        thread.start()

        return jsonify({
            "success": True,
            "job_id": parent_job_id,
            "status": "pending",
            "message": "Mutation job initialized"
        }), 202

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500


async def _dispatch_recombine_operation(data):
    """
    Handles the 'recombine' evolutionary operation dispatched via /execute_operation.
    Breeds a new generation of individuals from parent individuals or group.
    """
    if param_graph is None:
        return jsonify({"error": "No project loaded"}), 400

    try:
        params = data.get("params", {}) or {}
        initiator = data.get("initiator") or params.get("initiator") or {}
        initiator_id = initiator.get("id") if isinstance(initiator, dict) else (str(initiator).strip() if initiator else None)
        initiator_type = initiator.get("type") if isinstance(initiator, dict) else None

        parent_bundle_id = (
            data.get("parent_bundle_id")
            or params.get("parent_bundle_id")
            or (initiator_id if initiator_type == "bundle" else None)
        )
        parent_group_id = (
            data.get("parent_group_id")
            or params.get("parent_group_id")
            or (initiator_id if initiator_type == "group" else None)
        )
        raw_parent_input = (
            data.get("parent_ids")
            or params.get("parent_ids")
            or data.get("parents")
            or params.get("parents")
        )
        parent_ids = _extract_individual_parent_ids(raw_parent_input)

        # Resolve parent IDs from group if only group passed
        if not parent_ids and parent_group_id:
            with graph_lock:
                grp = param_graph.get_element(parent_group_id)
                if grp and hasattr(grp, "member_ids") and grp.member_ids:
                    parent_ids = _extract_individual_parent_ids(grp.member_ids)

        # Resolve parent ID from single individual initiator
        if not parent_ids and initiator_id and initiator_type == "individual":
            parent_ids = _extract_individual_parent_ids([initiator_id])

        if not parent_ids and not parent_bundle_id:
            return jsonify({"error": "Either parent_ids, parents, or parent_group_id must contain valid individual nodes."}), 400

        offspring_size = data.get("offspring_size") or params.get("offspring_size")
        if offspring_size is not None:
            offspring_size = int(offspring_size)

        selection_input = (
            data.get("selection")
            or params.get("selection")
            or data.get("selection_type")
            or params.get("selection_type")
            or "tournament"
        )
        if isinstance(selection_input, str):
            selection_cfg = {"type": selection_input, "tournament_size": 2}
        else:
            selection_cfg = selection_input

        crossover_input = (
            data.get("crossover")
            or params.get("crossover")
            or data.get("crossover_type")
            or params.get("crossover_type")
            or "hierarchical"
        )
        if isinstance(crossover_input, str):
            crossover_cfg = {"type": crossover_input, "num_cut_points": 1}
        else:
            crossover_cfg = crossover_input

        mutation_rate = float(
            data.get("mutation_rate")
            or params.get("mutation_rate")
            or (data.get("mutation") or {}).get("mutation_rate")
            or (params.get("mutation") or {}).get("mutation_rate")
            or 0.1
        )
        mutation_cfg = data.get("mutation") or params.get("mutation") or {"mutation_rate": mutation_rate}
        crossover_prob = float(
            data.get("crossover_prob")
            or params.get("crossover_prob")
            or 1.0
        )
        elitism = int(
            data.get("elitism")
            or params.get("elitism")
            or 0
        )
        generation_context = data.get("generation_context") or params.get("generation_context") or {}
        merged_generation_context = {**params, **generation_context}

        generate_exemplars = (
            data.get("generate_exemplars")
            if data.get("generate_exemplars") is not None
            else (
                params.get("generate_exemplars")
                if params.get("generate_exemplars") is not None
                else (
                    data.get("auto_generate_exemplars")
                    if data.get("auto_generate_exemplars") is not None
                    else params.get("auto_generate_exemplars", False)
                )
            )
        )
        if isinstance(generate_exemplars, str):
            generate_exemplars = generate_exemplars.lower() in ("true", "1", "yes")
        else:
            generate_exemplars = bool(generate_exemplars)

        raw_default_fitness = (
            data.get("default_fitness")
            if data.get("default_fitness") is not None
            else (
                params.get("default_fitness")
                if params.get("default_fitness") is not None
                else (
                    data.get("null_fitness_override")
                    if data.get("null_fitness_override") is not None
                    else (
                        params.get("null_fitness_override")
                        if params.get("null_fitness_override") is not None
                        else (
                            data.get("override_null_fitness")
                            if data.get("override_null_fitness") is not None
                            else params.get("override_null_fitness")
                        )
                    )
                )
            )
        )
        default_fitness = float(raw_default_fitness) if raw_default_fitness is not None else 0.0

        parent_job_id = data.get("job_id") or f"evolution_recombine_{uuid.uuid4().hex[:12]}"
        local_jobs[parent_job_id] = {
            "status": "pending",
            "progress": None,
            "result": None,
            "error": None
        }

        thread = threading.Thread(
            target=_run_recombine_task,
            kwargs=dict(
                parent_job_id=parent_job_id,
                parent_ids=parent_ids,
                parent_bundle_id=parent_bundle_id,
                offspring_size=offspring_size,
                selection_cfg=selection_cfg,
                crossover_cfg=crossover_cfg,
                mutation_cfg=mutation_cfg,
                crossover_prob=crossover_prob,
                elitism=elitism,
                generation_context=merged_generation_context,
                default_fitness=default_fitness,
                generate_exemplars=generate_exemplars,
            ),
            daemon=True
        )
        thread.start()

        return jsonify({
            "success": True,
            "job_id": parent_job_id,
            "status": "pending",
            "message": "Evolution recombination job initiated"
        }), 202

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500


_dispatch_reproduce_operation = _dispatch_recombine_operation


async def _dispatch_async_operation(data):
    """Routes an AI operation to the async Engine."""
    if engine_provider is None:
        return jsonify({"error": "Engine provider not initialized"}), 400
        
    try:
        operation = data.get("operation")
        if operation in ("wrap_individual", "create_individual"):
            return await _dispatch_wrap_individual_operation(data)
        elif operation == "mutate":
            return await _dispatch_mutate_operation(data)
        elif operation in ("recombine", "reproduce"):
            return await _dispatch_recombine_operation(data)

        model_id = data.get("model_id")
        job_id = data.get("job_id")
        operation = data.get("operation")
        params = data.get("params", {})
        
        initiator = data.get("initiator") or params.get("initiator") or {}
        initiator_id = initiator.get("id") if isinstance(initiator, dict) else (str(initiator).strip() if initiator else None)
        initiator_type = initiator.get("type") if isinstance(initiator, dict) else None

        # Resolve targeted Individual elements (if any)
        target_individual_nodes: list[Individual] = []
        target_individual_strengths: list[float] = []

        raw_individuals = data.get("individuals") or params.get("individuals")
        individual_id = (
            data.get("individual_id")
            or params.get("individual_id")
            or (initiator_id if initiator_type == "individual" else None)
        )

        with graph_lock:
            if raw_individuals and isinstance(raw_individuals, list):
                for ind_item in raw_individuals:
                    ind_id = ind_item.get("id") if isinstance(ind_item, dict) else ind_item
                    ind_str = ind_item.get("strength", 1.0) if isinstance(ind_item, dict) else 1.0
                    if ind_id and param_graph.G.has_node(ind_id):
                        ind_elem = param_graph.get_element(ind_id)
                        if isinstance(ind_elem, Individual) and ind_elem not in target_individual_nodes:
                            target_individual_nodes.append(ind_elem)
                            target_individual_strengths.append(float(ind_str))
            elif individual_id and param_graph.G.has_node(individual_id):
                ind_elem = param_graph.get_element(individual_id)
                if isinstance(ind_elem, Individual):
                    target_individual_nodes.append(ind_elem)
                    target_individual_strengths.append(1.0)

            # Auto-infer model_id from target individual if not explicitly specified
            if not model_id and target_individual_nodes:
                model_id = target_individual_nodes[0].base_model_id

        if not job_id:
            return jsonify({"error": "'job_id' is required."}), 400
        if not model_id:
            return jsonify({"error": "'model_id' is required."}), 400
        if not operation:
            return jsonify({"error": "'operation' is required."}), 400

        model_element = param_graph.get_element(model_id)
        if not isinstance(model_element, Model):
            return jsonify({"error": f"Node '{model_id}' is not a valid model."}), 400
        
        engine = engine_provider.get_engine()
        form_config = await engine.get_adapter_config(model_element.adapter)
        
        form_config = form_config.get(operation, [])
        if not form_config:
            return jsonify({"error": f"Adapter '{model_element.adapter}' has no '{operation}' configuration"}), 404

        DynamicArgsModel = create_dynamic_model(form_config)
        # Support fallback where frontend still sends params at top level
        validation_target = params if params else data
        validated_params = DynamicArgsModel.model_validate(validation_target)
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

        # --- Special Case Handlers (Gratings, Individuals & Inversion Sources) ---
        grating_strengths = []
        gratings = data.get("gratings") or params.get("gratings")
        if gratings:
            grating_elements = []
            for g_conf in gratings:
                g_id = g_conf.get("id")
                g_element = param_graph.get_element(g_id)
                if not isinstance(g_element, Grating):
                    return jsonify({"error": f"Node '{g_id}' is not a valid grating."}), 400
                grating_elements.append(g_element)
                grating_strengths.append(g_conf.get("strength", 1.0))
            node_engine_args["grating_elements"] = grating_elements

        # Pass individual_elements and resolve baseline grating
        if target_individual_nodes:
            node_engine_args["individual_elements"] = target_individual_nodes

            # Resolve baseline grating from the first individual if needed
            first_ind = target_individual_nodes[0]
            base_grating_elem = None
            if first_ind.baseline_grating_id and param_graph.G.has_node(first_ind.baseline_grating_id):
                base_grating_elem = param_graph.get_element(first_ind.baseline_grating_id)
            elif first_ind.context.get("baseline_file_path"):
                base_grating_elem = Grating(
                    id=f"grating_{uuid.uuid4().hex[:8]}",
                    name="Baseline Grating",
                    context={},
                    file=Asset(path=str(first_ind.context.get("baseline_file_path")), uid=f"grating_{uuid.uuid4().hex[:8]}", extension=".safetensors"),
                    base_model_id=model_id,
                    elements=first_ind.context.get("baseline_elements") or []
                )
            if base_grating_elem:
                node_engine_args["baseline_grating"] = base_grating_elem

        source_audio_id = data.get("source_audio_id") or params.get("source_audio_id")
        if source_audio_id:
            source_audio_element = param_graph.get_element(source_audio_id)
            if not isinstance(source_audio_element, Audio):
                return jsonify({"error": f"Node '{source_audio_id}' is not a valid audio artifact."}), 400
            node_engine_args["source_audio_element"] = source_audio_element

        # Resolve element list for edge creation
        resolved_elements = []
        for val in node_engine_args.values():
            if isinstance(val, list):
                resolved_elements.extend(val)
            else:
                resolved_elements.append(val)

        # Cache external audio files used in this step, and resolve to cache if missing
        for arg_name, element in node_engine_args.items():
            if isinstance(element, list):
                for el in element:
                    if not isinstance(el, (Audio, Model, Grating, Latent, Individual)):
                        return jsonify({"error": f"Node '{el.id}' is not a valid artifact."}), 400
            else:
                if isinstance(element, Audio):
                    cache_used_audio(element.id)
                    valid_path = resolve_audio_path(element.id)
                    if valid_path and str(valid_path) != element.file.path:
                        element.file = replace(element.file, path=str(valid_path))
                        node_engine_args[arg_name] = element
                
                if not isinstance(element, (Audio, Model, Grating, Latent, Individual)):
                    field_name = arg_name.removesuffix("_element")
                    return jsonify({"error": f"Node '{element.id}' for field '{field_name}' is not a valid artifact."}), 400

        engine_args = {"model_element": model_element, **node_engine_args}
        if gratings:
            engine_args["grating_strengths"] = grating_strengths
            engine_args["gratings"] = gratings
        if target_individual_nodes:
            engine_args["individual_strengths"] = target_individual_strengths
        
        # Determine parent and linked elements
        parent_id = None
        if target_individual_nodes:
            parent_id = target_individual_nodes[0].id
            linked_elements = [target_individual_nodes[0], *[el for el in resolved_elements if el.id != target_individual_nodes[0].id]]
        elif "grating_elements" in node_engine_args and node_engine_args["grating_elements"]:
            linked_elements = [*resolved_elements]
        else:
            linked_elements = [model_element, *resolved_elements]
        
        # --- Grouping/Batching Logic ---
        group_id = data.get("group_id") or data.get("batch_id")
        if group_id:
            with graph_lock:
                if not param_graph.G.has_node(group_id):
                    group_element = Group(id=group_id, member_type="audio" if operation != "invert" else "latent")
                    param_graph.add_element(group_element)
                    
                    # Try to position it near one of the source/linked elements
                    for el in linked_elements:
                        if param_graph.G.has_node(el.id):
                            el_pos = param_graph.G.nodes[el.id].get('position')
                            if el_pos:
                                param_graph.update_element(group_id, {"position": {"x": el_pos["x"] + 80, "y": el_pos["y"] + 80}})
                                break
                    
                    param_graph.save()
                    print(f"Created new group element {group_id}")

        # --- Execute ---
        print(f"Submitting {operation} job {job_id} to engine...")
        returned_job_id = await engine.execute(operation, job_id=job_id, **engine_args, **dumped_params)
        
        if returned_job_id != job_id:
             job_id = returned_job_id
             
        # Store job context to process the artifact later when the frontend polls /job_status
        v_params = validated_params.model_dump()
        v_params["model_id"] = model_id
        v_params["operation"] = operation
        if gratings:
            v_params["gratings"] = gratings
        if target_individual_nodes:
            v_params["individuals"] = [
                {"id": ind.id, "strength": str_val}
                for ind, str_val in zip(target_individual_nodes, target_individual_strengths)
            ]
            
        active_jobs[job_id] = {
            "parent_id": parent_id,
            "group_id": group_id,
            "linked_elements": linked_elements,
            "validated_params": v_params,
            "operation": operation
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
        print(f"AI Operation failed: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500

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

async def _rescan_source_task(job_id: str, source_name: str) -> None:
    try:
        local_jobs[job_id]["status"] = "running"
        local_jobs[job_id]["progress"] = {
            "value": 10,
            "total": 100,
            "description": f"Scanning external source '{source_name}'..."
        }

        path_node_id = None
        with graph_lock:
            for node_id, data in param_graph.G.nodes(data=True):
                if data.get("type") == "local_path" and (data.get("name") == source_name or node_id == source_name):
                    path_node_id = node_id
                    break
            if not path_node_id:
                for node_id, data in param_graph.G.nodes(data=True):
                    if data.get("type") == "local_path":
                        path_node_id = node_id
                        break

        if not path_node_id:
            raise ValueError(f"No source path matching '{source_name}' found.")

        await _expand_path_task(job_id, path_node_id)

    except Exception as e:
        print(f"Failed in async source rescan: {e}")
        traceback.print_exc()
        local_jobs[job_id]["status"] = "failed"
        local_jobs[job_id]["progress"] = None
        local_jobs[job_id]["error"] = str(e)
        local_jobs[job_id]["traceback"] = traceback.format_exc()


@app.route("/rescan_source", methods=["POST"])
async def rescan_source():
    """Rescan external source asynchronously as a local background job."""
    if param_graph is None:
        return jsonify({"error": "No project loaded"}), 400

    try:
        data = request.get_json() or {}
        source_name = data.get('source_name')
        if not source_name:
            return jsonify({"error": "source_name is required"}), 400

        job_id = f"rescan_source_{uuid.uuid4().hex[:12]}"
        local_jobs[job_id] = {
            "status": "pending",
            "progress": {"value": 0, "total": 100, "description": "Starting source rescan..."},
            "result": None,
            "error": None,
            "traceback": None,
            "name": "Rescan Source"
        }

        threading.Thread(
            target=lambda: asyncio.run(_rescan_source_task(job_id, source_name)),
            daemon=True
        ).start()

        return jsonify({
            "success": True,
            "job_id": job_id,
            "message": "Source rescan job started"
        }), 202

    except Exception as e:
        print(f"Failed to start source rescan: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500


async def _expand_path_task(job_id: str, path_node_id: str) -> None:
    try:
        local_jobs[job_id]["status"] = "running"
        local_jobs[job_id]["progress"] = {
            "value": 5,
            "total": 100,
            "description": "Locating path element..."
        }

        path_element = None
        with graph_lock:
            path_element = param_graph.get_element(path_node_id)
        if not path_element or not hasattr(path_element, "path"):
            raise ValueError("Invalid path element")

        dir_path = Path(path_element.path)
        if not dir_path.is_dir():
            raise ValueError(f"Path {dir_path} is not a valid directory")

        existing_dir_id = None
        new_dir_created = False
        new_dir = None
        new_edge = None
        with graph_lock:
            for u, v, edge_data in param_graph.G.out_edges(path_node_id, data=True):
                target_node = param_graph.get_element(v)
                if getattr(target_node, "type", None) == "directory":
                    existing_dir_id = v
                    break

            if not existing_dir_id:
                existing_dir_id = f"{path_node_id}.dir"
                new_dir = Directory(id=existing_dir_id, name=path_element.name, path=path_element.path)
                param_graph.add_element(new_dir)
                new_edge = param_graph.link(path_element, new_dir, relation='expands')
                new_dir_created = True

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
        entries = [entry for entry in dir_path.iterdir() if not entry.name.startswith('.')]
        total_entries = len(entries)
        new_elements = []

        for idx, entry in enumerate(entries):
            local_jobs[job_id]["progress"] = {
                "value": int(10 + (idx / max(total_entries, 1)) * 75),
                "total": 100,
                "description": f"Scanning entry {idx + 1} of {total_entries}: {entry.name}"
            }

            entry_str = str(entry)
            if entry_str in existing_paths:
                continue

            if entry.is_dir():
                element_id = uid_generator.from_string(str(entry.resolve()))
                new_elements.append(LocalPath(id=element_id, name=entry.name, path=entry_str))
            elif entry.is_file() and entry.suffix.lower() in audio_exts:
                try:
                    audio_tensor = load_audio(device_accelerator, entry_str, APP_SAMPLE_RATE)
                    element_id = uid_generator.from_tensor(audio_tensor)
                    asset = Asset(path=entry_str, uid=element_id, extension=entry.suffix.lower())
                    duration = audio_tensor.shape[-1] / float(APP_SAMPLE_RATE)
                    new_elements.append(Audio(id=element_id, name=entry.name, context={}, file=asset, duration=duration))
                except Exception as e:
                    print(f"Failed to load audio for {entry.name} to generate UID: {e}")

        local_jobs[job_id]["progress"] = {
            "value": 90,
            "total": 100,
            "description": "Saving directory contents to parameter graph..."
        }
        with graph_lock:
            for el in new_elements:
                param_graph.add_element(el)
                param_graph.update_element(el.id, {"parent": existing_dir_id})
            param_graph.save()

        # Trigger background incremental embedding updates
        trigger_embedding_update(background=True)

        result_payload = {
            "message": f"Successfully expanded path {path_element.name}",
            "directory_id": existing_dir_id,
            "node_id": existing_dir_id,
            "id": existing_dir_id,
            "elements": [el.to_dict() for el in new_elements]
        }
        if new_dir_created and new_dir:
            result_payload["new_directory"] = new_dir.to_dict()
        if new_edge:
            result_payload["new_edge"] = new_edge

        local_jobs[job_id]["status"] = "completed"
        local_jobs[job_id]["progress"] = {
            "value": 100,
            "total": 100,
            "description": f"Expanded {len(new_elements)} items."
        }
        local_jobs[job_id]["result"] = result_payload

    except Exception as e:
        print(f"Failed in async directory expansion: {e}")
        traceback.print_exc()
        local_jobs[job_id]["status"] = "failed"
        local_jobs[job_id]["progress"] = None
        local_jobs[job_id]["error"] = str(e)
        local_jobs[job_id]["traceback"] = traceback.format_exc()


@app.route("/expand_path", methods=["POST"])
async def expand_path():
    """Reads a PathNode, creates a Directory compound node if needed, and populates it asynchronously."""
    if param_graph is None:
        return jsonify({"error": "No project loaded"}), 400

    try:
        data = request.get_json() or {}
        path_node_id = data.get("path_node_id")
        if not path_node_id:
            return jsonify({"error": "path_node_id is required"}), 400

        job_id = f"expand_path_{uuid.uuid4().hex[:12]}"
        local_jobs[job_id] = {
            "status": "pending",
            "progress": {"value": 0, "total": 100, "description": "Starting directory expansion..."},
            "result": None,
            "error": None,
            "traceback": None,
            "name": "Expand Directory"
        }

        threading.Thread(
            target=lambda: asyncio.run(_expand_path_task(job_id, path_node_id)),
            daemon=True
        ).start()

        return jsonify({
            "success": True,
            "job_id": job_id,
            "message": "Directory expansion job started"
        }), 202

    except Exception as e:
        print(f"Failed to start directory expansion: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500

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

def resolve_image_path(image_id):
    """Gets the image path, falling back to the project cache if the original file was deleted."""
    if param_graph is None:
        return None
        
    path_str = param_graph.get_path_from_id(image_id, relative=False)
    if not path_str:
        return None
        
    original_path = Path(path_str)
    if original_path.exists():
        return original_path
        
    # Check fallback cache
    cache_dir = Path(param_graph.root) / "cache"
    if cache_dir.exists():
        cached_files = list(cache_dir.glob(f"{image_id}.*"))
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

def update_group_labels(group_id: str):
    """
    Recalculates the shared parameters of a group and updates the group's alias,
    as well as the aliases of all its children (based on their unique parameters).
    Assumes caller holds `graph_lock`.
    """
    if param_graph is None or not param_graph.G.has_node(group_id):
        return

    group_node_attrs = param_graph.G.nodes[group_id]
    if group_node_attrs.get('type') != 'group':
        return
        
    member_ids = group_node_attrs.get('member_ids', [])
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
    
    def diff_recursive(vals: list):
        if not vals:
            return None, []
        if all(v == vals[0] for v in vals):
            return vals[0], [None] * len(vals)
            
        if all(isinstance(v, dict) for v in vals):
            shared_dict = {}
            diff_dicts = [{} for _ in vals]
            all_keys = set().union(*(d.keys() for d in vals))
            
            for k in all_keys:
                if not all(k in d for d in vals):
                    for idx, d in enumerate(vals):
                        if k in d:
                            diff_dicts[idx][k] = d[k]
                    continue
                    
                k_vals = [d[k] for d in vals]
                shared_k, diff_k = diff_recursive(k_vals)
                
                if shared_k is not None or any(dk is not None for dk in diff_k):
                    if shared_k is not None:
                        shared_dict[k] = shared_k
                    for idx, dk in enumerate(diff_k):
                        if dk is not None:
                            diff_dicts[idx][k] = dk
            return (shared_dict if shared_dict else None), (diff_dicts if any(d for d in diff_dicts) else [None]*len(vals))
            
        if all(isinstance(v, list) for v in vals) and all(len(v) == len(vals[0]) for v in vals):
            shared_list = []
            diff_lists = [[] for _ in vals]
            has_diff = False
            
            for i in range(len(vals[0])):
                i_vals = [v[i] for v in vals]
                shared_i, diff_i = diff_recursive(i_vals)
                
                shared_list.append(shared_i)
                for idx, di in enumerate(diff_i):
                    diff_lists[idx].append(di)
                if any(di is not None for di in diff_i):
                    has_diff = True
                    
            if has_diff:
                return (shared_list if any(x is not None for x in shared_list) else None), diff_lists
            else:
                return shared_list, [None] * len(vals)
                
        return None, vals

    def flatten_diff(val, path="") -> list[tuple[str, any]]:
        if isinstance(val, dict):
            items = []
            for k, v in val.items():
                new_path = f"{path}.{k}" if path else k
                items.extend(flatten_diff(v, new_path))
            return items
        elif isinstance(val, list):
            items = []
            for i, v in enumerate(val):
                if v is not None:
                    new_path = f"{path}[{i}]"
                    items.extend(flatten_diff(v, new_path))
            return items
        else:
            return [(path, val)]

    def clean_path(path: str, ctx: dict) -> str:
        import re
        def replace_override(match):
            g_idx = int(match.group(1))
            o_idx = int(match.group(2))
            try:
                return ctx["gratings"][g_idx]["overrides"][o_idx]["address"]
            except (KeyError, IndexError, TypeError):
                return f"grating[{g_idx}]"
                
        path = re.sub(r'gratings\[(\d+)\]\.overrides\[(\d+)\]', replace_override, path)
        path = path.replace('.metadata.', '.')
        path = re.sub(r'gratings\[(\d+)\]', r'grating[\1]', path)
        return path

    shared_context = {}
    member_diffs_list = [{} for _ in member_ids]
    
    if contexts:
        shared_res, diff_res = diff_recursive(contexts)
        shared_context = shared_res or {}
        member_diffs_list = [d or {} for d in diff_res]

    print(f"Shared context: {shared_context}")
    print(f"Member diffs list: {member_diffs_list}")

    # Generate a label for the group based on the shared prompt/context
    group_alias = shared_context.get('prompt', "Artifact Group")
    if len(str(group_alias)) > 30:
        group_alias = str(group_alias)[:27] + "..."

    param_graph.update_element(group_id, {"shared_context": shared_context, "alias": group_alias})

    # Update member aliases
    for member_id, ctx, diff_dict in zip(member_ids, contexts, member_diffs_list):
        flat_diff = flatten_diff(diff_dict)
        diff_items = []
        for path, val in flat_diff:
            cleaned = clean_path(path, ctx)
            parts = cleaned.split('.')
            if len(parts) > 1:
                param_name = parts[-1]
                prefix = ".".join(parts[:-1])
                prefix_parts = prefix.split('.')
                short_prefix = ".".join(prefix_parts[-3:])
                diff_items.append(f"{param_name}: {val} ({short_prefix})")
            else:
                diff_items.append(f"{cleaned}: {val}")
            
        diff_label = "\n".join(diff_items) if diff_items else None
        if diff_label and len(diff_label) > 40:
            diff_label = diff_label[:37] + "..."
            
        param_graph.update_element(member_id, {"alias": diff_label})

def trigger_labeling_update():
    """
    Forces a graph-wide recalculation of all group and node labels.
    """
    if param_graph is None:
        return

    with graph_lock:
        group_ids = [
            node for node, data in param_graph.G.nodes(data=True)
            if data.get('type') == 'group'
        ]
        for group_id in group_ids:
            update_group_labels(group_id)
            
        param_graph.save()
    print("Labeling update completed successfully.")

def trigger_embedding_update(force_recalculate=False, background=True, job_id=None):
    """
    Background task to compute missing embeddings and recalculate similarity edges.
    If force_recalculate is True, it will recompute embeddings for all nodes.
    """
    if param_graph is None:
        return

    def update_embeddings_task():
        try:
            if job_id and job_id in local_jobs:
                local_jobs[job_id]["status"] = "running"
                local_jobs[job_id]["progress"] = {
                    "value": 10,
                    "total": 100,
                    "description": "Initializing CLAP/CLIP encoders..."
                }

            # Instantiate encoders lazily per task run
            clap_encoder = None
            clip_encoder = None
            
            # Initialize interrogator for semantic edge labels
            interrogator = SemanticInterrogator(device_accelerator)
            bank_path = Path(__file__).parent / "data" / "semantic_transitions_bank.pt"
            has_label_bank = bank_path.exists()
            if has_label_bank:
                interrogator.load_bank_from_disk(str(bank_path))
            
            group_keys = list(SIMILARITY_GROUPS.items())
            for g_idx, (group_type, embedding_type) in enumerate(group_keys):
                if job_id and job_id in local_jobs:
                    local_jobs[job_id]["progress"] = {
                        "value": int(20 + (g_idx / max(len(group_keys), 1)) * 60),
                        "total": 100,
                        "description": f"Processing {group_type} embeddings..."
                    }

                if group_type == "audio":
                    if clap_encoder is None:
                        clap_encoder = CLAPEncoder()
                    encoder = clap_encoder
                    resolver = resolve_audio_path
                elif group_type == "image":
                    if clip_encoder is None:
                        clip_encoder = CLIPEncoder()
                    encoder = clip_encoder
                    resolver = resolve_image_path
                else:
                    print(f"Unknown similarity group type: {group_type}")
                    continue

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
                        file_path = resolver(node)
                        if not file_path:
                            continue
                            
                        embedding = encoder.get_embedding(str(file_path))
                        
                        with graph_lock:
                            current_data = param_graph.G.nodes[node]
                            current_embeddings = current_data.get('embeddings', {})
                            current_embeddings[embedding_type] = embedding.tolist()
                            param_graph.update_element(node, {'embeddings': current_embeddings})
                            
                        print(f"Updated {embedding_type} embedding for node {node}")
                    except Exception as e:
                        print(f"Could not update {embedding_type} embedding for node {node}. Error: {e}")

                with graph_lock:
                    if param_graph is None or getattr(param_graph, "G", None) is None:
                        return
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
                            if has_label_bank and group_type == "audio":
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
                            if has_label_bank and group_type == "audio":
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
            if job_id and job_id in local_jobs:
                local_jobs[job_id]["status"] = "completed"
                local_jobs[job_id]["progress"] = {
                    "value": 100,
                    "total": 100,
                    "description": "Embeddings updated successfully."
                }
                local_jobs[job_id]["result"] = {
                    "message": "Embeddings updated successfully",
                    "success": True
                }

        except Exception as e:
            print(f"Failed to update embeddings: {e}")
            traceback.print_exc()
            if job_id and job_id in local_jobs:
                local_jobs[job_id]["status"] = "failed"
                local_jobs[job_id]["progress"] = None
                local_jobs[job_id]["error"] = str(e)
                local_jobs[job_id]["traceback"] = traceback.format_exc()
            
    if background:
        thread = threading.Thread(target=update_embeddings_task)
        thread.start()
    else:
        update_embeddings_task()

@app.route("/update_embeddings", methods=["POST"])
def update_embeddings():
    """Update all embeddings and create similarity edges asynchronously."""
    if param_graph is None:
        return jsonify({"error": "No project loaded"}), 400

    data = request.get_json(silent=True) or {}
    force = data.get("force", False)

    job_id = f"update_embeddings_{uuid.uuid4().hex[:12]}"
    local_jobs[job_id] = {
        "status": "pending",
        "progress": {"value": 0, "total": 100, "description": "Starting embeddings update..."},
        "result": None,
        "error": None,
        "traceback": None,
        "name": "Update Embeddings"
    }

    trigger_embedding_update(force_recalculate=force, background=True, job_id=job_id)

    return jsonify({
        "success": True,
        "job_id": job_id,
        "message": "Embeddings update started"
    }), 202

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

@app.route("/repair_edges", methods=["POST"])
def repair_edges():
    """Rebuild missing structural edges based on node context."""
    if param_graph is None:
        return jsonify({"error": "No project loaded"}), 400

    try:
        with graph_lock:
            edges_added = 0
            
            for node_id, node_data in param_graph.G.nodes(data=True):
                element = param_graph.get_element(node_id)
                if not element:
                    continue

                potential_source_ids = []
                
                if hasattr(element, 'base_model_id') and element.base_model_id:
                    potential_source_ids.append((element.base_model_id, 'binds_to'))
                    
                if hasattr(element, 'context') and isinstance(element.context, dict):
                    for key, value in element.context.items():
                        if key.endswith('_id') and isinstance(value, str):
                            potential_source_ids.append((value, 'source'))
                        elif key == 'gratings' and isinstance(value, list):
                            for grating_item in value:
                                g_id = grating_item.get('id')
                                if g_id:
                                    potential_source_ids.append((g_id, 'source'))
                                    
                for source_id, relation in potential_source_ids:
                    if param_graph.G.has_node(source_id):
                        # We check out_edges explicitly to ensure we don't skip structural edges 
                        # just because a 'spring' edge already exists between the nodes.
                        edge_exists = any(v == node_id and attrs.get('type') != 'spring' 
                                          for _, v, attrs in param_graph.G.out_edges(source_id, data=True))
                        if not edge_exists:
                            print(f"Restoring missing edge: {source_id} -> {node_id}")
                            source_el = param_graph.get_element(source_id)
                            param_graph.link(source_el, element, relation=relation)
                            edges_added += 1

            if edges_added > 0:
                param_graph.save()

        return jsonify({
            "message": f"Graph repaired: {edges_added} missing edges restored.",
            "edges_added": edges_added,
            "success": True
        })

    except Exception as e:
        print(f"Failed to repair edges: {e}")
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
        buffer = save_audio_to_buffer(audio_tensor, APP_SAMPLE_RATE, format="wav")
        
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

@app.route("/image_path/<string:image_id>", methods=["GET"])
def get_image_path(image_id):
    """Get the absolute path of an image file"""
    print(f"get_image_path called with id: {image_id}")
    if param_graph is None:
        print("get_image_path: param_graph is None. No project loaded.")
        return jsonify({"error": "No project loaded"}), 400
    
    try:
        image_path = resolve_image_path(image_id)
        print(f"Path resolved for graph: {image_path}")

        if not image_path:
            print(f"Image file not found for id: {image_id}")
            return jsonify({"error": "Image file not found"}), 404
        
        print(f"Returning image path: {image_path}")
        return jsonify({"path": str(image_path)})
        
    except Exception as e:
        print(f"Failed to get image path: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/image/<string:image_id>", methods=["GET"])
def serve_image(image_id):
    """Serve image files"""
    if param_graph is None:
        return jsonify({"error": "No project loaded"}), 400
    
    try:
        image_path = resolve_image_path(image_id)
        if not image_path:
            return jsonify({"error": "Image file not found"}), 404
        
        return send_file(str(image_path))
        
    except Exception as e:
        print(f"Failed to serve image: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

async def _export_audio_task(job_id: str, names: list[str], export_dir_str: str | None) -> None:
    try:
        local_jobs[job_id]["status"] = "running"
        local_jobs[job_id]["progress"] = {
            "value": 5,
            "total": 100,
            "description": "Preparing export directory..."
        }

        is_file_target = False
        if export_dir_str:
            export_path_obj = Path(export_dir_str)
            if len(names) == 1 and export_path_obj.suffix:
                export_dir = export_path_obj.parent
                is_file_target = True
            else:
                export_dir = export_path_obj
        else:
            export_dir = Path(param_graph.root) / "export"

        export_dir.mkdir(parents=True, exist_ok=True)

        exported_paths = []
        total_names = len(names)

        for idx, name in enumerate(names):
            local_jobs[job_id]["progress"] = {
                "value": int(10 + (idx / max(total_names, 1)) * 85),
                "total": 100,
                "description": f"Exporting file {idx + 1} of {total_names}: {name}"
            }

            element = None
            with graph_lock:
                if param_graph.G.has_node(name):
                    element = param_graph.get_element(name)
                else:
                    for node_id, node_data in param_graph.G.nodes(data=True):
                        if node_data.get('name') == name:
                            element = param_graph.get_element(node_id)
                            break

            if not element:
                raise ValueError(f"Element '{name}' not found")

            audio_path = resolve_audio_path(element.id)
            if not audio_path or not audio_path.exists():
                raise ValueError(f"Audio file for element '{name}' not found")

            if is_file_target:
                dest_path = export_path_obj
            else:
                dest_filename = element.name if element.name.endswith(audio_path.suffix) else f"{element.name}{audio_path.suffix}"
                dest_path = export_dir / dest_filename

            shutil.copy2(audio_path, dest_path)
            exported_paths.append(dest_path)

        export_path = exported_paths[0] if len(exported_paths) == 1 else export_dir

        local_jobs[job_id]["status"] = "completed"
        local_jobs[job_id]["progress"] = {
            "value": 100,
            "total": 100,
            "description": "Export completed."
        }
        local_jobs[job_id]["result"] = {
            "message": "Export completed",
            "export_path": str(export_path),
            "exported_files": [str(p) for p in exported_paths]
        }

    except Exception as e:
        print(f"Failed in async audio export: {e}")
        traceback.print_exc()
        local_jobs[job_id]["status"] = "failed"
        local_jobs[job_id]["progress"] = None
        local_jobs[job_id]["error"] = str(e)
        local_jobs[job_id]["traceback"] = traceback.format_exc()


@app.route("/export", methods=["POST"])
async def export_audio():
    """Export audio files asynchronously as a local background job."""
    if param_graph is None:
        return jsonify({"error": "No project loaded"}), 400

    try:
        data = request.get_json() or {}
        names = data.get('names', [])
        export_dir_str = data.get('export_dir') or data.get('custom_path')

        if not names:
            return jsonify({"error": "names list is required"}), 400

        job_id = f"export_{uuid.uuid4().hex[:12]}"
        local_jobs[job_id] = {
            "status": "pending",
            "progress": {"value": 0, "total": 100, "description": "Starting export..."},
            "result": None,
            "error": None,
            "traceback": None,
            "name": "Export Audio"
        }

        threading.Thread(
            target=lambda: asyncio.run(_export_audio_task(job_id, names, export_dir_str)),
            daemon=True
        ).start()

        return jsonify({
            "success": True,
            "job_id": job_id,
            "message": "Export job started"
        }), 202

    except Exception as e:
        print(f"Failed to start export: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500

# --------------------
#  Element Updates
# --------------------

@app.route("/update_element", methods=["POST"])
def update_element():
    """Update element attributes"""
    if param_graph is None:
        return jsonify({"error": "No project loaded"}), 400
    
    try:
        data = request.get_json() or {}
        element_name = data.get('id') or data.get('name')
        attributes = data.get('attributes', {})
        
        if not element_name:
            return jsonify({"error": "id or name is required"}), 400
        
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

    keep_children = request.args.get('keep_children', 'false').lower() == 'true'

    try:
        with graph_lock:
            if keep_children and param_graph.G.has_node(element_id):
                node_attrs = param_graph.G.nodes[element_id]
                
                # Find and unlink any nodes specifying this element as their parent
                children = [n for n, d in param_graph.G.nodes(data=True) if d.get('parent') == element_id]
                for child_id in children:
                    param_graph.update_element(child_id, {"parent": None, "alias": None})

                if node_attrs.get('type') == 'batch':
                    member_ids = node_attrs.get('member_ids', [])
                    for m_id in member_ids:
                        param_graph.update_element(m_id, {"parent": None, "alias": None})
                    
                    # CRITICAL: strip the batch of its members so remove_element doesn't cascade
                    param_graph.update_element(element_id, {"member_ids": []})
                    node_attrs['member_ids'] = []

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

@app.route("/elements", methods=["DELETE"])
def remove_elements():
    """Remove multiple elements from the graph"""
    if param_graph is None:
        return jsonify({"error": "No project loaded"}), 400

    try:
        data = request.get_json()
        element_ids = data.get('element_ids', [])
        keep_children = data.get('keep_children', False)

        with graph_lock:
            for element_id in element_ids:
                if keep_children and param_graph.G.has_node(element_id):
                    node_attrs = param_graph.G.nodes[element_id]
                    
                    # Find and unlink any nodes specifying this element as their parent
                    children = [n for n, d in param_graph.G.nodes(data=True) if d.get('parent') == element_id]
                    for child_id in children:
                        param_graph.update_element(child_id, {"parent": None, "alias": None})

                    if node_attrs.get('type') == 'group':
                        member_ids = node_attrs.get('member_ids', [])
                        for m_id in member_ids:
                            param_graph.update_element(m_id, {"parent": None, "alias": None})
                        
                        # CRITICAL: strip the group of its members so remove_element doesn't cascade
                        param_graph.update_element(element_id, {"member_ids": []})
                        node_attrs['member_ids'] = []

                param_graph.remove_element(element_id)
            param_graph.save()

        return jsonify({
            "message": f"Successfully removed {len(element_ids)} elements",
            "success": True
        })

    except Exception as e:
        print(f"Failed to remove elements: {e}")
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