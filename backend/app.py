# backend/app.py
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
import torch
import torchaudio
import io
import json
import traceback
from pathlib import Path
from typing import Dict, Any, Optional



# Import your existing modules
from param_graph.util import load_audio
from param_graph.graph import ParameterGraph
from param_graph.engine import Engine

from engines.stable_audio_tools import StableAudioTools


app = Flask(__name__)
CORS(app)

# Global state
device_type_accelerator = "cpu"
device_accelerator = torch.device(device_type_accelerator)
# A default sample rate for processing and playback
APP_SAMPLE_RATE = 48000

param_graph: ParameterGraph = None
engine: Engine = None

engine_map = {
    'stable_audio_tools': StableAudioTools
}

def set_engine(engine_name):
    global engine
    
    if engine and engine_name == engine.name:
        return
    engine = engine_map[engine_name]()
    print(f"Engine activated: {engine.name}")


# Argument type mappings
ARG_TYPES = {
    "sample_rate": int,
    "chunk_size": int,
    "batch_size": int,
    "steps": int,
    "seed": int,
    "noise_level": float,
    "chunk_interval": int,
}

def parse_args(args_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Parse and convert argument types"""
    parsed = {}
    for key, value in args_dict.items():
        if key in ARG_TYPES and value is not None:
            try:
                parsed[key] = ARG_TYPES[key](value)
            except (ValueError, TypeError):
                print(f"Failed to convert {key}={value} to {ARG_TYPES[key]}")
                parsed[key] = value
        else:
            parsed[key] = value
    return parsed


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
    """Load a project from a given absolute path."""
    global param_graph
    
    try:
        data = request.get_json()
        project_path_str = data.get("project_path") if data else None
        
        if not project_path_str:
            return jsonify({"error": "project_path is required"}), 400
        
        project_path = Path(project_path_str)
        project_name = project_path.name

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
    """Create a new project at a given absolute path."""
    global param_graph
    
    try:
        data = request.get_json()
        project_path_str = data.get("project_path") if data else None
        
        if not project_path_str:
            return jsonify({"error": "project_path is required"}), 400
        
        project_path = Path(project_path_str)

        param_graph = ParameterGraph(str(project_path))
        param_graph.save()
        project_name = project_path.name
        
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
def import_model():
    """Register a model"""
    if param_graph is None:
        return jsonify({"error": "No project loaded"}), 400
    
    try:
        data = request.get_json()
        engine_name = data.pop('engine')
        if engine_name == "default":
            engine_name = "stable_audio_tools"
        set_engine(engine_name)
        
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

@app.route("/generate", methods=["POST"])
def generate():
    """Generate audio"""
    if param_graph is None:
        return jsonify({"error": "No project loaded"}), 400
    
    try:
        data = request.get_json()
        args = parse_args(data)
        
        model_name = args.get('model_name')
        if not model_name:
            return jsonify({"error": "model_name is required"}), 400
        
        # Generation logic is commented out, preserving structure
        '''
        ...
        '''
        return jsonify({"message": "Generation endpoint is currently a placeholder."}), 200

    except Exception as e:
        print(f"Generation failed: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/variation", methods=["POST"])
def variation():
    """Create audio variation"""
    if param_graph is None:
        return jsonify({"error": "No project loaded"}), 400
    
    try:
        data = request.get_json()
        args = parse_args(data)
        
        source_name = args.get('source_name')
        if not source_name:
            return jsonify({"error": "source_name is required"}), 400
            
        source_path = param_graph.get_path_from_name(source_name, relative=False)
        source_audio = load_audio(device_accelerator, str(source_path), APP_SAMPLE_RATE)
        
        # Variation logic is commented out, preserving structure
        '''
        ...
        '''
        return jsonify({"message": "Variation endpoint is currently a placeholder."}), 200
        
    except Exception as e:
        print(f"Variation failed: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# --------------------
#  External Sources
# --------------------

@app.route("/add_external_source", methods=["POST"])
def add_external_source():
    """Add external audio source"""
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
        use_reloader=False
    )