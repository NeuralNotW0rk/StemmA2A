# backend/engine_service.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import traceback
from pathlib import Path

from pydantic import ValidationError
from pydantic_util import create_dynamic_model
from engine.engine_provider import EngineProvider
from param_graph.registry import resolve_element


app = Flask(__name__)
CORS(app)

# Global state
device_type_accelerator = "cuda" if torch.cuda.is_available() else "cpu"
device_accelerator = torch.device(device_type_accelerator)

# The engine service acts as a simple, remote executor.
# It maintains a local file cache for assets required for generation.
data_cache_root = Path("/app/data")
data_cache_root.mkdir(parents=True, exist_ok=True)

# Initialize the engine provider. Since this is the engine service,
# it will always use the local engine implementation.
engine_provider = EngineProvider()

# --------------------
#  Health Check
# --------------------

@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "device": str(device_accelerator)
    })

# --------------------
#  Engine API
# --------------------

@app.route("/adapter_config/<adapter_name>", methods=["GET"])
async def get_adapter_config(adapter_name):
    """Get the form configuration for a specific adapter."""
    try:
        engine = engine_provider.get_engine()
        config = await engine.get_adapter_config(adapter_name)
        return jsonify(config)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/register_model", methods=["POST"])
async def register_model():
    """Register a model with the engine."""
    try:
        data = request.get_json()
        adapter_name = data.get("adapter")
        params = data.get("params")

        engine = engine_provider.get_engine()
        model_artifact = await engine.register_model(adapter_name, **params)
        return jsonify(model_artifact.to_dict())
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/execute", methods=["POST"])
async def execute():
    """
    Execute a generation task.
    This is the primary endpoint for the remote execution engine.
    """
    try:
        data = request.get_json()
        adapter_params = data.get("adapter_params") # This is the model dataclass
        generation_params = data.get("generation_params")
        output_dir = data.get("output_dir")

        if not all([adapter_params, generation_params, output_dir]):
            return jsonify({"error": "Missing required parameters in request."}), 400

        engine = engine_provider.get_engine()

        # The remote engine now handles its own asset checking if needed,
        # but the local engine (which this service runs) will assume assets are present.
        model_element = resolve_element(adapter_params)

        # Compare the model's required hashes against local storage.
        required_hashes = set(model_element.get_hashes())
        missing_hashes = [hsh for hsh in required_hashes if not (data_cache_root / hsh).exists()]
        print(f"Missing assets: {missing_hashes}")
        if missing_hashes:
            return jsonify({"error": "Missing assets", "missing_hashes": missing_hashes}), 422
        
        # Execute generation
        audio_artifact = await engine.execute(
            model_element.anchor(data_cache_root),
            output_dir=output_dir,
            **generation_params
        )
        
        return jsonify(audio_artifact.to_dict())

    except (ValidationError, ValueError) as e:
        traceback.print_exc()
        return jsonify({"error": "Invalid request", "details": str(e)}), 400
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/upload", methods=["POST"])
def upload():
    """
    Uploads a file (expected to be a model asset) to the engine's local data store.
    """
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file:
        # The filename is expected to be the hash of the file
        filename = file.filename
        # Assets are saved in a 'data' subdirectory of our mock project root
        upload_path = data_cache_root / "data" / filename
        upload_path.parent.mkdir(parents=True, exist_ok=True)
        file.save(str(upload_path))
        return jsonify({"message": f"File {filename} uploaded successfully"})

    return jsonify({"error": "File upload failed"}), 500

# --------------------
#  Main
# --------------------

if __name__ == "__main__":
    print(f"Starting StemmA2A Engine Service on device: {device_accelerator}")
    app.run(
        host="0.0.0.0",
        port=5001,
        debug=True
    )
