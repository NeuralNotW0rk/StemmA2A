# backend/engine_service.py
import json
import os
import traceback
from pathlib import Path

from flask import Flask, make_response, request, jsonify, send_file
from flask_cors import CORS
import torch
from pydantic import ValidationError

from engine.engine_provider import EngineProvider
from utils.uid import path_from_uid
from param_graph.utils import resolve_elements_from_dicts
from param_graph.registry import resolve_element


app = Flask(__name__)
CORS(app)

# Global state
device_type_accelerator = "cuda" if torch.cuda.is_available() else "cpu"
device_accelerator = torch.device(device_type_accelerator)

is_container = os.environ.get("RUNNING_IN_CONTAINER") == "true"

if is_container:
    data_path_str = os.environ.get("CONTAINER_DATA_PATH")
    if not data_path_str:
        raise ValueError("CONTAINER_DATA_PATH must be set when running in a container.")
else:
    data_path_str = os.environ.get("LOCAL_DATA_PATH")
    if not data_path_str:
        raise ValueError("LOCAL_DATA_PATH must be set when running locally.")

data_cache_root = Path(data_path_str).expanduser()
data_cache_root.mkdir(parents=True, exist_ok=True)

# Set Hugging Face cache directory to a persistent location within the data cache.
# This prevents models like CLAP from redownloading every time the container restarts.
hf_cache_dir = data_cache_root / "huggingface"
hf_cache_dir.mkdir(parents=True, exist_ok=True)
os.environ["HF_HOME"] = str(hf_cache_dir)

# Initialize the engine provider with the data cache path.
engine_provider = EngineProvider(data_root=str(data_cache_root))

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
    Receives an operation, validates assets, and queues it for execution.
    Returns a job ID.
    """
    try:
        data = request.get_json()
        operation_id = data.get("operation")
        params = data.get("params")

        # Extract job_id from top-level payload or fallback to popping it from params
        job_id = data.get("job_id")
        if not job_id and isinstance(params, dict) and "job_id" in params:
            job_id = params.pop("job_id")
        elif isinstance(params, dict) and "job_id" in params:
            del params["job_id"]
            
        print(f"engine_service.py: extracted job_id: {job_id}")

        if not operation_id or params is None:
            return jsonify({"error": "Missing 'operation' or 'params' in request."}), 400

        engine = engine_provider.get_engine()
        resolved_params, all_elements = resolve_elements_from_dicts(params)
        
        required_uids = {uid for element in all_elements.values() for uid in element.get_uids()}
        missing_uids = [uid for uid in required_uids if not (data_cache_root / path_from_uid(uid)).exists()]
        
        if missing_uids:
            print(f"Missing assets: {missing_uids}")
            return jsonify({"error": "Missing assets", "missing_uids": missing_uids}), 422
        
        # This was a bug. We need to anchor all resolved elements, not just `all_elements`.
        # And we need to preserve the non-element params.
        anchored_params = resolved_params.copy()
        for key, element in all_elements.items():
            anchored_params[key] = element.anchor(str(data_cache_root), with_extension=False)

        op_id_str = operation_id['id'] if isinstance(operation_id, dict) and 'id' in operation_id else operation_id

        # The engine's execute method now returns a job ID. Pass job_id if provided.
        if job_id:
            print(f"engine_service.py: calling engine.execute with job_id={job_id}")
            job_id = await engine.execute(op_id_str, job_id=job_id, **anchored_params)
        else:
            print(f"engine_service.py: calling engine.execute without job_id")
            job_id = await engine.execute(op_id_str, **anchored_params)
        
        print(f"engine_service.py: engine.execute returned job_id={job_id}")
        # Return the job ID to the client
        return jsonify({"job_id": job_id})

    except (ValidationError, ValueError) as e:
        traceback.print_exc()
        return jsonify({"error": "Invalid request", "details": str(e)}), 400
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/job_status/<job_id>", methods=["GET"])
async def get_job_status(job_id):
    """Gets the status of a previously submitted job."""
    try:
        engine = engine_provider.get_engine()
        status = await engine.get_job_status(job_id)

        # De-anchor completed results before sending them over the wire
        if status.get("status") == "completed" and "result" in status:
            result_data = status.get("result")
            if result_data and "id" in result_data:
                element = resolve_element(result_data)
                de_anchored_element = element.de_anchor()
                status["result"] = de_anchored_element.to_dict()

        return jsonify(status)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/jobs/<job_id>/cancel", methods=["POST"])
async def cancel_job(job_id):
    """
    Requests cancellation of a specific job.
    """
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


@app.route("/download_asset/<asset_id>", methods=["GET"])
def download_asset(asset_id):
    """Downloads a generated asset file from the content-addressable cache."""
    try:
        asset_path = data_cache_root / path_from_uid(asset_id)
        
        if not asset_path.exists():
            # The local engine saves with a .wav extension, but the path_from_uid doesn't
            # account for it. We need to find the file. Let's assume .wav for now.
            # A more robust solution might store extensions or check for common types.
            asset_path = asset_path.with_suffix('.wav')
            if not asset_path.exists():
                 return jsonify({"error": f"Asset not found for id {asset_id}"}), 404
            
        return send_file(str(asset_path), as_attachment=True)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/upload", methods=["POST"])
def upload():
    """
    Uploads a file to the engine's local data store.
    """
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file:
        uid = file.filename
        destination = data_cache_root / path_from_uid(uid)
        destination.parent.mkdir(parents=True, exist_ok=True)
        file.save(destination)
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
