# backend/engine_service.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import traceback
from pathlib import Path
import os

from engine.engine import Engine
from engine.stable_audio_tools import StableAudioTools
from param_graph.graph import ParameterGraph # Needed for asset paths

app = Flask(__name__)
CORS(app)

# Global state
device_type_accelerator = "cuda" if torch.cuda.is_available() else "cpu"
device_accelerator = torch.device(device_type_accelerator)

engine: Engine = None

# This is a mock param_graph. In the remote engine, we don't have a full project context.
# We only need it to resolve asset paths, assuming assets are uploaded to a predictable 'data' directory.
mock_project_root = Path("/app/local_data")
param_graph = ParameterGraph(str(mock_project_root))


engine_registry = {
    'stable_audio_tools': StableAudioTools
}

def set_engine(engine_name: str) -> bool:
    """
    Sets the global engine instance.
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
    return jsonify({
        "status": "healthy",
        "device": str(device_accelerator)
    })

# --------------------
#  Execution Endpoint
# --------------------

@app.route("/execute", methods=["POST"])
async def execute():
    """
    Execute a generation task.
    This is the primary endpoint for the remote execution engine.
    """
    try:
        data = request.get_json()
        engine_name = data.get("engine")
        engine_params = data.get("engine_params")
        generation_params = data.get("generation_params")
        output_dir = data.get("output_dir")

        if not all([engine_name, engine_params, generation_params, output_dir]):
            return jsonify({"error": "Missing required parameters in request."}), 400

        # 1. Set the engine
        if not set_engine(engine_name):
            return jsonify({"error": f"Engine '{engine_name}' not found"}), 404

        # 2. Check for missing assets before loading the model
        required_assets = engine.get_required_assets(engine_params)
        missing_hashes = []
        for asset_hash in required_assets:
            # The remote execution strategy is expected to upload assets to '/app/local_data/data/<hash>'
            asset_path = param_graph.root / "data" / asset_hash
            if not asset_path.exists():
                missing_hashes.append(asset_hash)

        if missing_hashes:
            return jsonify({
                "error": "Missing required assets for model execution.",
                "missing_hashes": missing_hashes
            }), 422  # Unprocessable Entity

        # 3. Load the model from the provided parameters
        model_info = engine.register_model(**engine_params)
        engine.load_model(model_info)

        # 4. Execute generation
        artifact = await engine.generate(output_dir=output_dir, **generation_params)
        
        # 5. Return the resulting artifact
        return jsonify(artifact.to_dict())

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
        upload_path = param_graph.root / "data" / filename
        upload_path.parent.mkdir(parents=True, exist_ok=True)
        file.save(str(upload_path))
        return jsonify({"message": f"File {filename} uploaded successfully"})

    return jsonify({"error": "File upload failed"}), 500

# --------------------
#  Main
# --------------------

if __name__ == "__main__":
    # The app runs on 0.0.0.0 to be accessible from the host machine
    # when running in a Docker container. The port is 5000.
    print(f"Starting StemmA2A Engine Service on device: {device_accelerator}")
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )
