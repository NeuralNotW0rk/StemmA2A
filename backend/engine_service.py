# backend/engine_service.py
import json
from flask import Flask, make_response, request, jsonify, send_file
from flask_cors import CORS
import torch
import traceback
from pathlib import Path

from pydantic import ValidationError

from engine.engine_provider import EngineProvider
from param_graph.registry import resolve_element
from utils.uid_utils import path_from_uid


app = Flask(__name__)
CORS(app)

# Global state
device_type_accelerator = "cuda" if torch.cuda.is_available() else "cpu"
device_accelerator = torch.device(device_type_accelerator)

# The engine service acts as a simple, remote executor.
# It maintains a local file cache for assets required for generation.
data_cache_root = Path("/app/data")

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
        required_uids = set()
        for element in all_elements.values():
            required_uids.update(element.get_uids())
        print(f"Required uids: {list(required_uids)}")

        missing_uids = [uid for uid in required_uids if not (data_cache_root / path_from_uid(uid)).exists()]
        if missing_uids:
            print(f"Missing assets: {missing_uids}")
            return jsonify({"error": "Missing assets", "missing_uids": missing_uids}), 422
        print("All assets present")
        
        # Anchor all elements with the local asset cache paths
        anchored_params = resolved_params.copy()
        for key, element in all_elements.items():
            anchored_params[key] = element.anchor(data_cache_root)

        # If the operation is an element, get its ID
        op_id_str = operation_id
        if isinstance(operation_id, dict) and 'id' in operation_id:
            op_id_str = operation_id['id']

        # Execute the operation
        result_artifact = await engine.execute(op_id_str, **anchored_params)
        
        # The result is anchored to a local file. We need to send this file
        # back, along with the de-anchored graph element.
        
        # 1. Get the de-anchored element and serialize it to JSON
        element_dict = result_artifact.de_anchor().to_dict()
        element_json = json.dumps(element_dict)

        # 2. Get the path to the local file asset
        # This is a bit of a simplification. We're assuming the first asset
        # is the one we want to send.
        asset_path = result_artifact.file.path

        # 3. Use send_file to prepare the file part of the response
        import os
        response = make_response(send_file(asset_path, as_attachment=True, download_name=f"{result_artifact.name}{Path(asset_path).suffix}"))
        
        # 4. Add the JSON as a custom header
        response.headers['X-Graph-Element'] = element_json
        
        # 5. Expose the custom header so the client can access it
        response.headers['Access-Control-Expose-Headers'] = 'X-Graph-Element'
        
        return response

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
        # The filename is expected to be the uid of the file
        uid = file.filename
        file.save(data_cache_root / path_from_uid(uid))
        return jsonify({"message": f"File {uid} uploaded successfully"})

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
