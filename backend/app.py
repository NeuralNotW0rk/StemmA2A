# backend/app.py
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import torch
import torchaudio
import io
import traceback
from pathlib import Path
import os

from vfs.factory import get_vfs_provider
from pydantic import ValidationError
from pydantic_util import create_dynamic_model
from param_graph.util import load_audio
from param_graph.graph import ParameterGraph
from param_graph.engine import Engine, Model
from engines.stable_audio_tools import StableAudioTools

provider = get_vfs_provider()

app = Flask(__name__)
CORS(app)

# Global state
device_type_accelerator = "cpu"
device_accelerator = torch.device(device_type_accelerator)
# A default sample rate for processing and playback
APP_SAMPLE_RATE = 48000

param_graph: ParameterGraph = None
engine: Engine = None

engine_registry = {
    'stable_audio_tools': StableAudioTools
}

def set_engine(engine_name: str) -> bool:
    """
    Sets the global engine instance. Avoids re-initializing the engine if it's already active.
    Returns True on success, False on failure.
    """
    global engine
    
    if engine and engine.name == engine_name:
        return True

    engine_class = engine_registry.get(engine_name)
    if not engine_class:
        print(f"Error: Engine '{engine_name}' not found in engine_registry.")
        return False

    try:
        engine = engine_class()
        print(f"Engine activated: {engine.name}")
        return True
    except Exception as e:
        print(f"Error initializing engine '{engine_name}': {e}")
        traceback.print_exc()
        engine = None
        return False


# --------------------
#  Health Check
# --------------------

@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint"""
    global param_graph
    backend_type = os.environ.get("BACKEND_TYPE", "local").lower()
    return jsonify({
        "status": "healthy",
        "device": str(device_accelerator),
        "project_loaded": param_graph is not None,
        "backend_type": backend_type
    })



# --------------------
#  Project Management
# --------------------

@app.route("/projects", methods=["GET"])
def list_projects():
    """List all available projects."""
    try:
        projects = provider.list_projects()
        return jsonify({"projects": projects})
    except Exception as e:
        print(f"Failed to list projects: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/load_project", methods=["POST"])
def load_project():
    """
    Load a project from the VFS.
    """
    global param_graph
    
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Request body is required"}), 400

        project_name = data.get("project_name")
        if not project_name:
            return jsonify({"error": "project_name is required"}), 400
        
        project_path_str = provider.get_project_path(project_name)
        project_path = Path(project_path_str)

        param_graph = ParameterGraph(str(project_path))
        if param_graph.load():
            return jsonify({
                "message": f"Project '{project_name}' loaded successfully.",
                "project_name": project_name,
                "project_path": project_path_str,
                "success": True
            })
        
        return jsonify({
            "message": f"Project '{project_name}' not found.",
            "success": False
        })
        
    except Exception as e:
        print(f"Failed to load project: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
    
@app.route("/create_project", methods=["POST"])
def create_project():
    """
    Create a new project in the VFS.
    """
    global param_graph
    
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Request body is required"}), 400
            
        project_name = data.get("project_name")
        if not project_name:
            return jsonify({"error": "project_name is required"}), 400
        
        project_path_str = provider.get_project_path(project_name)
        project_path = Path(project_path_str)

        param_graph = ParameterGraph(str(project_path))
        param_graph.save()
        
        return jsonify({
            "message": f"Project '{project_name}' created successfully.",
            "project_name": project_name,
            "project_path": project_path_str,
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

@app.route("/data_root", methods=["GET"])
def get_data_root():
    """Get the data root."""
    try:
        data_root = os.getenv("DATA_ROOT", "data")
        return jsonify({"data_root": os.path.abspath(data_root)})
    except Exception as e:
        print(f"Failed to get data root: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


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

@app.route("/graph_tsne", methods=["GET"])
def get_tsne_graph():
    """Get graph data in cluster mode"""
    if param_graph is None:
        return jsonify({"error": "No project loaded"}), 400
    
    try:
        # param_graph.update_tsne()
        graph_data = param_graph.to_json(mode='cluster')
        return jsonify({
            "message": "success",
            "graph_data": graph_data,
            "success": True
        })
    except Exception as e:
        print(f"Failed to get t-SNE graph data: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# --------------------
#  Model Operations
# --------------------
@app.route("/models", methods=["GET"])
def list_models():
    """List all available models."""
    # In the local filesystem implementation, models are global.
    # For a remote provider, we might want to list models per project.
    project_id = request.args.get("project_id")
    try:
        models = provider.list_models(project_id)
        return jsonify({"models": models})
    except Exception as e:
        print(f"Failed to list models: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/register_model", methods=["POST"])
def import_model():
    # Register a model
    if param_graph is None:
        return jsonify({"error": "No project loaded"}), 400
    
    try:
        data = request.get_json()
        engine_name = data.pop("engine")
        copy_to_staging = data.pop("copy_to_staging", False)
        set_engine(engine_name)

        # Staging logic
        if copy_to_staging:
            provider.export_model_package({
                "name": data.get("name"),
                "config_path": data.get("config_path"),
                "checkpoint_path": data.get("checkpoint_path"),
            })

        # Resolve logical paths to physical paths using the VFS provider
        config_logical_path = data.get("config_path")
        checkpoint_logical_path = data.get("checkpoint_path")

        if not config_logical_path or not checkpoint_logical_path:
            return jsonify({"error": "config_path and checkpoint_path are required"}), 400

        # The provider handles resolving these, whether they are local full paths
        # or remote logical paths.
        data["config_path"] = provider.resolve_to_physical_path(config_logical_path)
        data["checkpoint_path"] = provider.resolve_to_physical_path(checkpoint_logical_path)

        param_graph.add_artifact(engine.register_model(**data))
        param_graph.save()
        
        return jsonify({
            "message": "Model registered successfully",
            "success": True
        })
        
    except Exception as e:
        print(f"Failed to import model: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/engine_config/<engine_name>", methods=["GET"])
def get_engine_config(engine_name):
    """Get the form configuration for a specific engine."""
    try:
        engine_class = engine_registry.get(engine_name)
        if not engine_class:
            return jsonify({"error": f"Engine '{engine_name}' not found"}), 404

        engine_instance = engine_class()
        if not hasattr(engine_instance, 'get_form_config'):
             return jsonify({"error": f"Engine '{engine_name}' does not have a form configuration"}), 404

        config = engine_instance.get_form_config()
        return jsonify(config)

    except Exception as e:
        print(f"Failed to get engine config: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/generate", methods=["POST"])
def generate():
    """Generate audio using dynamic validation based on engine config."""
    if param_graph is None:
        return jsonify({"error": "No project loaded"}), 400
    
    try:
        json_data = request.get_json()
        model_id = json_data.get("model_id")
        if not model_id:
            return jsonify({"error": "'model_id' is required."}), 400

        # 1. Get the full model dataclass from the graph
        model_element = param_graph.get_element(model_id)
        
        # 2. Validate that it's a model
        if not isinstance(model_element, Model):
            return jsonify({"error": f"Node '{model_id}' is not a valid model."}), 400
        
        # 3. Set engine and load the model object
        if not set_engine(model_element.engine):
            return jsonify({"error": f"Failed to set engine '{model_element.engine}'."}), 500
        
        engine.load_model(model_element) # Pass the dataclass instance
            
        # 4. Get engine config for validation
        form_config = engine.get_form_config().get("generate")
        if not form_config:
            return jsonify({"error": f"Engine '{engine.name}' has no 'generate' configuration"}), 404

        # 5. Validate the request body
        DynamicArgsModel = create_dynamic_model(form_config)
        validated_params = DynamicArgsModel.model_validate(json_data)

        # 6. Call actual generation
        output_dir = param_graph.root / "generated"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        audio_artifact = engine.generate(output_dir=str(output_dir), **validated_params.model_dump())
        
        # 7. Add the new audio artifact to the graph
        param_graph.add_artifact(audio_artifact)
        param_graph.save()
        
        return jsonify({
            "message": "Audio generated and registered successfully.",
            "artifact": audio_artifact.to_dict(),
            "validated_params": validated_params.model_dump()
        }), 200

    except (ValidationError, ValueError) as e:
        # Catch validation errors from Pydantic or ValueErrors from get_element
        return jsonify({"error": "Invalid request", "details": str(e)}), 400
    except Exception as e:
        print(f"Generation failed: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# --------------------
#  External Sources
# --------------------

@app.route("/add_external_source", methods=["POST"])
def add_external_source():
    """
    Add external audio source.
    This is currently only supported for local backends.
    """
    if param_graph is None:
        return jsonify({"error": "No project loaded"}), 400
    
    try:
        backend_type = os.environ.get("BACKEND_TYPE", "local").lower()
        if backend_type != "local":
            return jsonify({
                "error": "Adding external sources from remote clients is not yet implemented.",
                "success": False
            }), 501 # 501 Not Implemented

        data = request.get_json()
        source_path = data.get('source_path')
        
        if not source_path:
            return jsonify({"error": "source_path is required"}), 400
        
        param_graph.add_external_source(source_path)
        param_graph.save()
        
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
        
        return jsonify({
            "message": "Source rescanned successfully",
            "success": True
        })
        
    except Exception as e:
        print(f"Failed to rescan source: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# --------------------
#  Audio Operations
# --------------------

@app.route("/audio_data/<path:filename>", methods=["GET"])
def serve_audio_data(filename):
    """
    Loads an audio file, processes it, and returns the raw audio data.
    This ensures a consistent format (stereo WAV) for playback.
    """
    if param_graph is None:
        return jsonify({"error": "No project loaded"}), 400
    
    try:
        audio_path = param_graph.get_path_from_name(filename, relative=False)
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

@app.route("/audio_path/<path:filename>", methods=["GET"])
def get_audio_path(filename):
    """Get the absolute path of an audio file"""
    print(f"get_audio_path called with filename: {filename}")
    if param_graph is None:
        print("get_audio_path: param_graph is None. No project loaded.")
        return jsonify({"error": "No project loaded"}), 400
    
    try:
        audio_path = param_graph.get_path_from_name(filename, relative=False)
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

@app.route("/audio/<path:filename>", methods=["GET"])
def serve_audio(filename):
    """Serve audio files"""
    if param_graph is None:
        return jsonify({"error": "No project loaded"}), 400
    
    try:
        audio_path = param_graph.get_path_from_name(filename, relative=False)
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
#  VFS API (for Remote Client)
# --------------------
# These endpoints are used by the RemoteClientVFS to interact with the
# RemoteServerVFS. When the backend is run with VFS_MODE=remote_server,
# the 'provider' will be a RemoteServerVFS instance.

@app.route("/api/vfs/projects", methods=["GET"])
def vfs_list_projects():
    try:
        return jsonify(provider.list_projects())
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/vfs/models", methods=["GET"])
def vfs_list_models():
    try:
        project_id = request.args.get("project_id", "")
        return jsonify(provider.list_models(project_id))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/vfs/metadata", methods=["GET"])
def vfs_get_file_metadata():
    try:
        path = request.args.get("path")
        if not path:
            return jsonify({"error": "path parameter is required"}), 400
        
        metadata = provider.get_file_metadata(path)
        if metadata is None:
            return jsonify({"error": "File not found"}), 404
        return jsonify(metadata)
    except PermissionError as e:
        return jsonify({"error": str(e)}), 403
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/vfs/exists", methods=["GET"])
def vfs_exists():
    try:
        path = request.args.get("path")
        if not path:
            return jsonify({"error": "path parameter is required"}), 400
        return jsonify(provider.exists(path))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/vfs/project-path", methods=["GET"])
def vfs_get_project_path():
    try:
        project_name = request.args.get("project_name")
        if not project_name:
            return jsonify({"error": "project_name parameter is required"}), 400
        return jsonify({"path": provider.get_project_path(project_name)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/vfs/model-path", methods=["GET"])
def vfs_get_model_path():
    try:
        model_name = request.args.get("model_name")
        if not model_name:
            return jsonify({"error": "model_name parameter is required"}), 400
        return jsonify({"path": provider.get_model_path(model_name)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/vfs/resolve-path", methods=["GET"])
def vfs_resolve_path():
    try:
        logical_path = request.args.get("path")
        if not logical_path:
            return jsonify({"error": "path parameter is required"}), 400
        
        physical_path = provider.resolve_to_physical_path(logical_path)
        return jsonify({"path": physical_path})
    except (PermissionError, FileNotFoundError) as e:
        return jsonify({"error": str(e)}), 404
    except Exception as e:
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
        debug=True,
        threaded=True,
        use_reloader=True
    )