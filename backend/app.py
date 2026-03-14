# backend/app.py
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import torch
import torchaudio
import io
import traceback
from pathlib import Path
import os

from pydantic import ValidationError
from pydantic_util import create_dynamic_model
from param_graph.util import load_audio
from param_graph.graph import ParameterGraph
from param_graph.elements.models.base import Model
from engine.engine_provider import EngineProvider
from export_util import export_project

app = Flask(__name__)
CORS(app)

# Global state
device_type_accelerator = "cpu"
device_accelerator = torch.device(device_type_accelerator)
# A default sample rate for processing and playback
APP_SAMPLE_RATE = 48000

param_graph: ParameterGraph = None
execution_url = os.environ.get("ENGINE_URL")
engine_provider = EngineProvider(remote_url=execution_url)
if execution_url:
    print(f"Engine: Using remote engine at {execution_url}")
else:
    print("Engine: Using local engine")


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
        "project_loaded": param_graph is not None
    })



# --------------------
#  Project Management
# --------------------

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

@app.route("/export_project", methods=["POST"])
def export_project_route():
    """Export a project to a specified directory."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Request body is required"}), 400

        project_path = data.get("project_path")
        export_path = data.get("export_path")

        if not project_path or not export_path:
            return jsonify({"error": "project_path and export_path are required"}), 400
        
        destination = export_project(project_path, export_path)
        
        return jsonify({
            "message": "Project exported successfully.",
            "destination": destination,
            "success": True
        })
        
    except Exception as e:
        print(f"Failed to export project: {e}")
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

        param_graph.add_artifact(model_artifact)
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
    """Generate audio using dynamic validation based on adapter config."""
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
        
        # 3. Get adapter config for validation
        engine = engine_provider.get_engine()
        form_config = await engine.get_adapter_config(model_element.adapter)
        form_config = form_config.get("generate")
        if not form_config:
            return jsonify({"error": f"Adapter '{model_element.adapter}' has no 'generate' configuration"}), 404

        # 4. Validate the request body
        DynamicArgsModel = create_dynamic_model(form_config)
        validated_params = DynamicArgsModel.model_validate(json_data)

        # 5. Call actual generation using the provider
        output_dir = param_graph.root / "generated"
        output_dir.mkdir(parents=True, exist_ok=True)

        audio_artifact = await engine.execute(
            model_element,
            output_dir=str(output_dir),
            **validated_params.model_dump()
        )
        
        # 6. Add the new audio artifact to the graph
        param_graph.add_artifact(audio_artifact)
        param_graph.save()
        
        return jsonify({
            "message": "Audio generated and registered successfully.",
            "artifact": audio_artifact.to_dict(),
            "validated_params": validated_params.model_dump()
        }), 200

    except (ValidationError, ValueError) as e:
        # Catch validation errors from Pydantic or ValueErrors from get_element
        print(f"Invalid request: {e}")
        traceback.print_exc()
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