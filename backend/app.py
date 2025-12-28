# backend/app.py
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
import torch
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import your existing modules
from param_graph.util import load_audio
from param_graph.graph import ParameterGraph

app = Flask(__name__)
CORS(app)

# Configuration
PROJECT_DIR = Path("projects")
PROJECT_DIR.mkdir(exist_ok=True)

# Global state
device_type_accelerator = "cpu"
device_accelerator = torch.device(device_type_accelerator)

gen_graph: Optional[ParameterGraph] = None

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
                logger.warning(f"Failed to convert {key}={value} to {ARG_TYPES[key]}")
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
    return jsonify({
        "status": "healthy",
        "device": str(device_accelerator),
        "project_loaded": param_graph is not None
    })

# --------------------
#  Project Management
# --------------------

@app.route("/load", methods=["POST"])
def load_project():
    """Load a project"""
    global param_graph
    
    try:
        data = request.get_json()
        if not data or 'project_name' not in data:
            # Fallback to form data for compatibility
            project_name = request.form.get("project_name")
        else:
            project_name = data['project_name']
        
        if not project_name:
            return jsonify({"error": "project_name is required"}), 400
        
        project_path = PROJECT_DIR / project_name
        if not project_path.exists():
            # Create new project
            project_path.mkdir(parents=True, exist_ok=True)
        
        param_graph = ParameterGraph(str(project_path))
        
        return jsonify({
            "message": f"Project loaded: {project_name}",
            "project": project_name,
            "success": True
        })
        
    except Exception as e:
        logger.error(f"Failed to load project: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/project", methods=["GET"])
def get_project():
    """Get current project info"""
    if param_graph is not None:
        return jsonify({
            "message": "success",
            "project_name": param_graph.root.name,
            "success": True
        })
    else:
        return jsonify({
            "message": "no project selected",
            "success": False
        })

@app.route("/list-projects", methods=["GET"])
def list_projects():
    """List available projects"""
    try:
        project_names = []
        for path in PROJECT_DIR.iterdir():
            if path.is_dir():
                project_names.append(path.name)
        
        return jsonify({
            "message": "success",
            "project_names": project_names,
            "success": True
        })
    except Exception as e:
        logger.error(f"Failed to list projects: {e}")
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
        logger.error(f"Failed to get graph data: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/graph-tsne", methods=["GET"])
def get_tsne_graph():
    """Get graph data in cluster mode"""
    if param_graph is None:
        return jsonify({"error": "No project loaded"}), 400
    
    try:
        # Update t-SNE embeddings if needed
        param_graph.update_tsne()
        graph_data = param_graph.to_json(mode='cluster')
        return jsonify({
            "message": "success",
            "graph_data": graph_data,
            "success": True
        })
    except Exception as e:
        logger.error(f"Failed to get t-SNE graph data: {e}")
        return jsonify({"error": str(e)}), 500

# --------------------
#  Model Operations
# --------------------

@app.route("/import-model", methods=["POST"])
def import_model():
    """Import a model"""
    if param_graph is None:
        return jsonify({"error": "No project loaded"}), 400
    
    try:
        data = request.get_json()
        model_path = data.get('model_path')
        
        if not model_path:
            return jsonify({"error": "model_path is required"}), 400
        
        # Import the model
        param_graph.import_model(model_path)
        param_graph.save()
        
        return jsonify({
            "message": "Model imported successfully",
            "success": True
        })
        
    except Exception as e:
        logger.error(f"Failed to import model: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/generate", methods=["POST"])
def generate():
    """Generate audio"""
    if param_graph is None:
        return jsonify({"error": "No project loaded"}), 400
    
    try:
        data = request.get_json()
        args = parse_args(data)
        
        # Extract parameters
        model_name = args.get('model_name')
        steps = args.get('steps', 100)
        batch_size = args.get('batch_size', 4)
        seed = args.get('seed')
        sample_rate = args.get('sample_rate', 48000)
        chunk_size = args.get('chunk_size', 32768)
        
        if not model_name:
            return jsonify({"error": "model_name is required"}), 400
        
        '''
        # Create generation request
        generation_request = Request(
            request_type=RequestType.GENERATION,
            model_path=model_name,
            model_type=ModelType.DD,
            sample_rate=sample_rate,
            chunk_size=chunk_size,
            batch_size=batch_size,
            steps=steps,
            seed=seed
        )
        
        # Generate audio
        results = request_handler.generate_request(generation_request)
        
        # Log results to knowledge graph
        for i, result in enumerate(results):
            param_graph.log_inference(
                mode='generation',
                model_name=model_name,
                seed=seed + i if seed else None,
                sample_rate=sample_rate,
                output=result,
                steps=steps,
                batch_size=batch_size,
                chunk_size=chunk_size
            )
        
        param_graph.save()
        
        return jsonify({
            "message": "Generation completed",
            "samples_generated": len(results),
            "success": True
        })
        '''
        
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/variation", methods=["POST"])
def variation():
    """Create audio variation"""
    if param_graph is None:
        return jsonify({"error": "No project loaded"}), 400
    
    try:
        data = request.get_json()
        args = parse_args(data)
        
        # Extract parameters
        source_name = args.get('source_name')
        model_name = args.get('model_name')
        noise_level = args.get('noise_level', 0.7)
        steps = args.get('steps', 100)
        batch_size = args.get('batch_size', 4)
        seed = args.get('seed')
        
        if not source_name:
            return jsonify({"error": "source_name is required"}), 400
        
        # Load source audio
        source_path = param_graph.get_path_from_name(source_name, relative=False)
        source_audio = load_audio(device_accelerator, str(source_path), 48000)
        
        '''# Create variation request
        variation_request = Request(
            request_type=RequestType.VARIATION,
            model_path=model_name or "default",
            model_type=ModelType.DD,
            source_audio=source_audio,
            noise_level=noise_level,
            steps=steps,
            batch_size=batch_size,
            seed=seed
        )
        
        # Generate variations
        results = request_handler.generate_request(variation_request)
        
        # Log results to knowledge graph
        for i, result in enumerate(results):
            param_graph.log_inference(
                mode='variation',
                model_name=model_name or "default",
                source_name=source_name,
                seed=seed + i if seed else None,
                sample_rate=48000,
                output=result,
                noise_level=noise_level,
                steps=steps,
                batch_size=batch_size
            )
        
        param_graph.save()
        
        return jsonify({
            "message": "Variation completed",
            "samples_generated": len(results),
            "success": True
        })'''
        
    except Exception as e:
        logger.error(f"Variation failed: {e}")
        return jsonify({"error": str(e)}), 500

# --------------------
#  External Sources
# --------------------

@app.route("/add-external-source", methods=["POST"])
def add_external_source():
    """Add external audio source"""
    if param_graph is None:
        return jsonify({"error": "No project loaded"}), 400
    
    try:
        data = request.get_json()
        source_path = data.get('source_path')
        
        if not source_path:
            return jsonify({"error": "source_path is required"}), 400
        
        # Add external source
        param_graph.add_external_source(source_path)
        param_graph.save()
        
        return jsonify({
            "message": "External source added successfully",
            "success": True
        })
        
    except Exception as e:
        logger.error(f"Failed to add external source: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/rescan-source", methods=["POST"])
def rescan_source():
    """Rescan external source"""
    if param_graph is None:
        return jsonify({"error": "No project loaded"}), 400
    
    try:
        data = request.get_json()
        source_name = data.get('source_name')
        
        if not source_name:
            return jsonify({"error": "source_name is required"}), 400
        
        # Rescan external source
        param_graph.scan_external_source(source_name)
        param_graph.save()
        
        return jsonify({
            "message": "Source rescanned successfully",
            "success": True
        })
        
    except Exception as e:
        logger.error(f"Failed to rescan source: {e}")
        return jsonify({"error": str(e)}), 500

# --------------------
#  Audio Operations
# --------------------

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
        logger.error(f"Failed to serve audio: {e}")
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
        
        # Export audio files
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
        logger.error(f"Export failed: {e}")
        return jsonify({"error": str(e)}), 500

# --------------------
#  Element Updates
# --------------------

@app.route("/update-element", methods=["POST"])
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
        
        # Update element
        param_graph.update_element(element_name, attributes)
        param_graph.save()
        
        return jsonify({
            "message": "Element updated successfully",
            "success": True
        })
        
    except Exception as e:
        logger.error(f"Failed to update element: {e}")
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
    logger.info(f"Starting StemmA2A backend on device: {device_accelerator}")
    logger.info(f"Projects directory: {PROJECT_DIR.absolute()}")
    
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True,
        threaded=True
    )