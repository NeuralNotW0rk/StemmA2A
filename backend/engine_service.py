import json
import os
import traceback
import threading
import uuid
from dataclasses import replace
from pathlib import Path
import logging

from flask import Flask, make_response, request, jsonify, send_file
from flask_cors import CORS
import torch
from pydantic import ValidationError

from engine.engine_provider import EngineProvider
from engine import uploads
from engine.cache_policy import CachePolicy, GIB, touch
from engine.inbox import Inbox
from engine.model_store import ModelStore, MissingModelFiles
from engine.results import map_result_elements
from utils.filesystem import get_path_size, remove_path
from utils.uid import path_from_uid
from utils.migrations import run_global_migrations
from utils.log_capture import start_log_capture
from param_graph.registry import resolve_element


app = Flask(__name__)
CORS(app)

# Filter out health check and job status logs
class QuietLogFilter(logging.Filter):
    def filter(self, record):
        msg = record.getMessage()
        return msg.find('/health') == -1 and msg.find('/job_status') == -1

logging.getLogger('werkzeug').addFilter(QuietLogFilter())

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
run_global_migrations(data_cache_root)

# Set Hugging Face cache directory to a persistent location within the data cache.
# This prevents models like CLAP from redownloading every time the container restarts.
hf_cache_dir = data_cache_root / "huggingface"
hf_cache_dir.mkdir(parents=True, exist_ok=True)
os.environ["HF_HOME"] = str(hf_cache_dir)
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"

# Set Triton and PyTorch extensions cache directories to a persistent location within the data cache.
# This prevents costly Triton/PyTorch kernel re-compilations every time the container is restarted.
triton_cache_dir = data_cache_root / "triton"
triton_cache_dir.mkdir(parents=True, exist_ok=True)
os.environ["TRITON_CACHE_DIR"] = str(triton_cache_dir)

torch_ext_dir = data_cache_root / "torch_extensions"
torch_ext_dir.mkdir(parents=True, exist_ok=True)
os.environ["TORCH_EXTENSIONS_DIR"] = str(torch_ext_dir)

# Initialize the engine provider with the data cache path.
engine_provider = EngineProvider(data_root=str(data_cache_root))


def _is_job_active(job_id: str) -> bool:
    status = engine_provider.get_engine().job_statuses.get(job_id, {}).get("status")
    return status in ("pending", "running")


# Size-capped eviction for the asset cache. ENGINE_CACHE_MAX_GB=0 turns eviction off
cache_policy = CachePolicy(
    data_root=data_cache_root,
    max_bytes=int(float(os.environ.get("ENGINE_CACHE_MAX_GB", "50")) * GIB),
    min_age_seconds=float(os.environ.get("ENGINE_CACHE_MIN_AGE_HOURS", "24")) * 3600,
    is_job_active=_is_job_active,
)
# Shared model files, outside the cache and its eviction
model_store = ModelStore(data_cache_root)
# Drop folder for copying assets into the cache by hand (see engine/inbox.py)
inbox = Inbox(data_cache_root, cache_policy.lock, model_store.model_path)

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

@app.route("/operations", methods=["GET"])
async def get_operations():
    """Gets the supported operations from the engine."""
    try:
        engine = engine_provider.get_engine()
        ops = await engine.get_supported_operations()
        return jsonify({"operations": ops, "success": True})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500

@app.route("/shared_models", methods=["GET"])
async def get_shared_models():
    """Gets the shared models from the engine."""
    try:
        engine = engine_provider.get_engine()
        models = await engine.get_shared_models()
        return jsonify({"shared_models": models, "success": True})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500


@app.route("/shared_models", methods=["POST"])
async def add_shared_model():
    """
    Promotes a model into the shared registry. Files are named by UID; any the server lacks are
    reported as 422 missing_uids for the client to upload, as with /execute.
    """
    entry = request.get_json() or {}
    try:
        # Promotion moves files out of the cache, so it must not interleave with a sweep or /execute
        with cache_policy.lock:
            model = await engine_provider.get_engine().add_shared_model(entry)
        return jsonify({"shared_model": model, "success": True})
    except MissingModelFiles as e:
        return jsonify({"error": "Missing assets", "missing_uids": e.uids}), 422
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/shared_models/<model_id>", methods=["DELETE"])
async def remove_shared_model(model_id):
    """Removes a shared model. Its files return to the cache and age out like any other entry."""
    try:
        with cache_policy.lock:
            await engine_provider.get_engine().remove_shared_model(model_id)
        return jsonify({"success": True})
    except ValueError as e:
        return jsonify({"error": str(e)}), 404
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


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
        
        all_elements = {}
        def _resolve_and_collect(val):
            if isinstance(val, dict) and "id" in val and "type" in val:
                element = resolve_element(val)
                all_elements[element.id] = element
                return element
            elif isinstance(val, list):
                return [_resolve_and_collect(v) for v in val]
            elif isinstance(val, dict):
                return {k: _resolve_and_collect(v) for k, v in val.items()}
            return val

        resolved_params = _resolve_and_collect(params)

        required_assets = {}
        for element in all_elements.values():
            if hasattr(element, 'get_assets'):
                for asset in element.get_assets():
                    required_assets[asset.uid] = asset

        # The job ID is fixed up front so the job's inputs can be leased before it is queued
        if not job_id:
            job_id = str(uuid.uuid4())

        required_uids = {uid for element in all_elements.values() for uid in element.get_uids()}
        with cache_policy.lock:
            missing_uids = []
            for uid in required_uids:
                # Model store files were checked when promoted, and are never evicted here
                if model_store.model_path(uid).exists():
                    continue
                cache_file = data_cache_root / path_from_uid(uid)
                asset = required_assets.get(uid)
                if not cache_file.exists():
                    missing_uids.append(uid)
                    continue
                # A directory asset is stored as the directory, so its recorded size applies as-is
                cached_size = get_path_size(cache_file)
                if cached_size == 0 and cache_file.is_file():
                    problem = "Zero-byte cached asset"
                elif asset and asset.size and asset.size > 0 and cached_size != asset.size:
                    problem = f"Corrupt or incomplete cached asset (expected {asset.size} bytes, found {cached_size} bytes)"
                else:
                    touch(cache_file)
                    continue
                print(f"[Engine Service] Warning: {problem} '{uid}'. Evicting from cache.")
                try:
                    remove_path(cache_file)
                except Exception as e:
                    print(f"Failed to remove corrupt cache entry {cache_file}: {e}")
                missing_uids.append(uid)

            if missing_uids:
                print(f"[Engine Service] Requesting missing/corrupt assets from client: {missing_uids}")
                return jsonify({"error": "Missing assets", "missing_uids": missing_uids}), 422

            def _anchor_elements(val):
                if hasattr(val, 'anchor') and callable(val.anchor):
                    anchored = val.anchor(str(data_cache_root), with_extension=False)
                    # Assets in the model store live under models/, not the cache
                    updates = {
                        name: replace(asset, path=str(model_store.model_path(asset.uid)))
                        for name, asset in anchored._iter_assets()
                        if model_store.model_path(asset.uid).exists()
                    }
                    return replace(anchored, **updates) if updates else anchored
                elif isinstance(val, list):
                    return [_anchor_elements(v) for v in val]
                elif isinstance(val, dict):
                    return {k: _anchor_elements(v) for k, v in val.items()}
                return val

            anchored_params = _anchor_elements(resolved_params)

            op_id_str = operation_id['id'] if isinstance(operation_id, dict) and 'id' in operation_id else operation_id

            # Held until the job is no longer pending or running. The lock stays held until the job
            # is queued, so a sweep can't see the lease before the job exists and drop it
            cache_policy.lease(job_id, required_uids)
            print(f"engine_service.py: calling engine.execute with job_id={job_id}")
            job_id = await engine.execute(op_id_str, job_id=job_id, **anchored_params)
        
        print(f"engine_service.py: engine.execute returned job_id={job_id}")
        # Return the job ID to the client
        return jsonify({"job_id": job_id})

    except (ValidationError, ValueError) as e:
        traceback.print_exc()
        return jsonify({"error": "Invalid request", "details": str(e)}), 400
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500

@app.route("/job_status/<job_id>", methods=["GET"])
async def get_job_status(job_id):
    """Gets the status of a previously submitted job."""
    try:
        engine = engine_provider.get_engine()
        status = await engine.get_job_status(job_id)

        # De-anchor completed results (every artifact of a multi-artifact result) before sending them over the wire
        if status.get("status") == "completed" and status.get("result"):
            status["result"] = map_result_elements(
                status["result"], lambda data: resolve_element(data).de_anchor().to_dict()
            )

        return jsonify(status)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500


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


def _format_bytes(num_bytes: int | float) -> str:
    if num_bytes < 1024:
        return f"{num_bytes:.0f} B"
    elif num_bytes < 1024 * 1024:
        return f"{num_bytes / 1024:.2f} KB"
    elif num_bytes < 1024 * 1024 * 1024:
        return f"{num_bytes / (1024 * 1024):.2f} MB"
    else:
        return f"{num_bytes / (1024 * 1024 * 1024):.2f} GB"


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
                print(f"[Engine Service] Warning: [Download] Asset not found: {asset_id}")
                return jsonify({"error": f"Asset not found for id {asset_id}"}), 404
        
        if asset_path.is_dir():
            return jsonify({"error": f"Asset {asset_id} is a directory, which can't be downloaded as a file"}), 400
        touch(asset_path)
        file_size = os.path.getsize(asset_path)
        print(f"[Engine Service] [Download] Serving asset '{asset_id}' ({_format_bytes(file_size)}) to client...")
        return send_file(str(asset_path), as_attachment=True)
    except Exception as e:
        print(f"[Engine Service] [Download Error] Failed to serve asset '{asset_id}': {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


_upload_locks: dict[str, threading.Lock] = {}
_locks_mutex = threading.Lock()

def _get_upload_lock(uid: str) -> threading.Lock:
    with _locks_mutex:
        if uid not in _upload_locks:
            _upload_locks[uid] = threading.Lock()
        return _upload_locks[uid]


def _already_present(uid: str) -> bool:
    """Whether the cache or the model store already holds uid, so an upload of it is redundant."""
    return uploads.destination_path(data_cache_root, uid).exists() or model_store.model_path(uid).exists()


@app.route("/upload", methods=["POST"])
def upload():
    """
    Receives one chunk of a file asset, or of one file inside a directory asset when the form has a
    "relpath". A directory asset is completed by /upload/finalize. See engine/uploads.py.
    """
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    uid = request.form.get("uid") or file.filename
    relpath = request.form.get("relpath") or None
    chunk_index = int(request.form.get("chunk_index", 0))
    total_chunks = int(request.form.get("total_chunks", 1))
    total_size = int(request.form.get("total_size", 0))
    label = f"{uid}/{relpath}" if relpath else uid

    chunk_data = file.read()
    try:
        with _get_upload_lock(uid):
            if _already_present(uid):
                # Arrived another way meanwhile, e.g. through the inbox; the client can stop sending
                remove_path(uploads.staging_path(data_cache_root, uid))
                return jsonify({"message": f"{uid} is already on the engine", "already_present": True})
            if total_chunks > 1:
                print(f"[Engine Service] [Upload] Received chunk {chunk_index + 1}/{total_chunks} for '{label}' ({_format_bytes(len(chunk_data))})")
            complete = uploads.write_chunk(
                data_cache_root, uid, chunk_data, chunk_index, total_chunks, total_size, relpath
            )
    except uploads.UploadError as e:
        print(f"[Engine Service] [Upload Error] {e}")
        return jsonify({"error": str(e)}), 400

    if complete:
        where = "staged" if relpath else "saved to cache"
        print(f"[Engine Service] [Upload] Completed '{label}' ({_format_bytes(total_size)}, {total_chunks} chunk(s)) {where}.")
        return jsonify({"message": f"File {label} uploaded successfully"})
    return jsonify({"message": f"Chunk {chunk_index} of {label} uploaded"})


@app.route("/upload/finalize", methods=["POST"])
def finalize_upload():
    """Completes a directory asset whose files were sent to /upload, after checking it against its UID."""
    data = request.get_json() or {}
    uid = data.get("uid")
    if not uid:
        return jsonify({"error": "uid is required"}), 400
    total_size = int(data.get("total_size", 0))
    try:
        with _get_upload_lock(uid):
            if _already_present(uid):
                remove_path(uploads.staging_path(data_cache_root, uid))
                return jsonify({"message": f"{uid} is already on the engine", "already_present": True})
            uploads.finalize_directory(data_cache_root, uid, total_size)
    except uploads.UploadError as e:
        print(f"[Engine Service] [Upload Error] {e}")
        return jsonify({"error": str(e)}), 409
    print(f"[Engine Service] [Upload] Completed directory '{uid}' ({_format_bytes(total_size)}) saved to cache.")
    return jsonify({"message": f"Directory {uid} uploaded successfully"})


# --------------------
#  Asset Cache
# --------------------

@app.route("/cache", methods=["GET"])
def get_cache_status():
    """Reports the cache size and what a sweep would evict right now, without evicting anything."""
    try:
        return jsonify({**cache_policy.sweep(dry_run=True), "success": True})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/cache/sweep", methods=["POST"])
def sweep_cache():
    """Runs an eviction sweep now instead of waiting for the background one."""
    try:
        return jsonify({**cache_policy.sweep(), "success": True})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# --------------------
#  Main
# --------------------

if __name__ == "__main__":
    # Started here rather than at import so test runs do not write to the service log
    start_log_capture(data_cache_root / "logs", "engine_service")
    print(f"Starting StemmA2A Engine Service on device: {device_accelerator}")
    # With debug on, this block also runs in the reloader's watcher process; sweep only in the server
    if os.environ.get("WERKZEUG_RUN_MAIN") == "true":
        cache_policy.start_background_sweeps(float(os.environ.get("ENGINE_CACHE_SWEEP_MINUTES", "15")) * 60)
        inbox.start_background_scans(float(os.environ.get("ENGINE_INBOX_SCAN_SECONDS", "10")))
    app.run(
        host="0.0.0.0",
        port=5001,
        debug=True
    )
