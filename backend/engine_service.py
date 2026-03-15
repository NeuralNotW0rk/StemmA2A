# backend/engine_service.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import traceback
from pathlib import Path

from pydantic import ValidationError

from asset_cache import AssetCache
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
asset_cache = AssetCache(data_cache_root)

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

@app.route("/execute", methods=["POST"])
async def execute():
    """
    Execute a generic operation.
    This is the primary endpoint for the remote execution engine.
    """
    try:
        data = request.get_json()
        operation_id = data.get("operation")
        params = data.get("params")

        if not all([operation_id, params]):
            return jsonify({"error": "Missing 'operation' or 'params' in request."}), 400

        engine = engine_provider.get_engine()

        # Find all graph elements in the params and resolve them from dicts to objects
        resolved_params = params.copy()
        all_elements = {}
        for key, value in params.items():
            # This is a simple check. We might need a more robust way to identify
            # dicts that are meant to be graph elements.
            if isinstance(value, dict) and 'id' in value and 'type' in value:
                element = resolve_element(value)
                resolved_params[key] = element
                all_elements[key] = element
                print(f"Received {key}: {element.type} {element.id}")
        
        # Check for missing assets across all resolved elements
        required_hashes = set()
        for element in all_elements.values():
            required_hashes.update(element.get_hashes())
        print(f"Required hashes: {list(required_hashes)}")

        missing_hashes = [hsh for hsh in required_hashes if not asset_cache.exists(hsh)]
        if missing_hashes:
            print(f"Missing assets: {missing_hashes}")
            return jsonify({"error": "Missing assets", "missing_hashes": missing_hashes}), 422
        print("All assets present")
        
        # Anchor all elements with the local asset cache paths
        anchored_params = resolved_params.copy()
        for key, element in all_elements.items():
            # This anchor logic might need to be more sophisticated if elements have multiple assets
            # For now, we assume the first hash is representative for finding the anchor root.
            if element.get_hashes():
                 anchored_params[key] = element.anchor(asset_cache.get_anchor(element.get_hashes()[0]))

        # If the operation is an element, get its ID
        op_id_str = operation_id
        if isinstance(operation_id, dict) and 'id' in operation_id:
            op_id_str = operation_id['id']

        # Execute the operation
        result_artifact = await engine.execute(op_id_str, **anchored_params)
        
        # De-anchor the result before sending it back
        return jsonify(result_artifact.de_anchor().to_dict())

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
        hsh = file.filename
        asset_cache.save_file(file, hsh)
        return jsonify({"message": f"File {hsh} uploaded successfully"})

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
