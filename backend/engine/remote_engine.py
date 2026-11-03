import aiohttp
import asyncio
import os
import json
import threading
import time
import uuid
from pathlib import Path
from typing import Any
from param_graph.elements.base_elements import GraphElement

from .engine import Engine
from .results import is_multi_result
from param_graph.registry import resolve_element
from utils.uid import path_from_uid

# The engine service sits behind a proxy (Cloudflare) that drops any single request still running after
# about 100 s. Assets are therefore sent in small chunks, a few assets at a time, so each request gets
# enough of the upload bandwidth to finish quickly however many jobs are queued at once.
UPLOAD_CHUNK_SIZE = 4 * 1024 * 1024
MAX_CONCURRENT_UPLOADS = int(os.environ.get("REMOTE_ENGINE_MAX_UPLOADS", 2))
# Attempts per asset; a failed file is re-sent from its first chunk, which restarts it on the server
UPLOAD_ATTEMPTS = 3
# Rounds of uploading missing assets before giving up on an /execute request
MAX_UPLOAD_ROUNDS = 3
# How long a finished transfer stays in get_transfers(), so the UI can show how it ended
TRANSFER_HISTORY_SECONDS = 30


def _request_error(e: Exception) -> Exception:
    """Describes a failed engine service request, telling an unreachable server apart from a failed request."""
    if isinstance(e, aiohttp.ClientResponseError):
        hint = " (timed out in transit through the proxy)" if e.status == 524 else ""
        return Exception(f"Engine service request to {e.request_info.url} failed: {e.status} {e.message}{hint}")
    return Exception("Cannot reach engine service. Please verify that the remote server is running.")


async def check_engine_service(remote_url: str, timeout: float = 5.0) -> dict[str, Any]:
    """
    Probes an engine service's /health. Returns {"reachable": True, "device": ...} or
    {"reachable": False, "error": ...}. Works without a RemoteEngine, so it can check a URL before any project loads.
    """
    headers = {}
    if os.environ.get("CF_ACCESS_CLIENT_ID") and os.environ.get("CF_ACCESS_CLIENT_SECRET"):
        headers["CF-Access-Client-Id"] = os.environ["CF_ACCESS_CLIENT_ID"]
        headers["CF-Access-Client-Secret"] = os.environ["CF_ACCESS_CLIENT_SECRET"]

    try:
        async with aiohttp.ClientSession(headers=headers, timeout=aiohttp.ClientTimeout(total=timeout)) as session:
            # Cloudflare Access answers rejected credentials with a redirect to its login page
            async with session.get(f"{remote_url}/health", allow_redirects=False) as response:
                if 300 <= response.status < 400:
                    return {"reachable": False, "error": "Redirected to a login page. Check the Cloudflare Access credentials."}
                if response.status != 200:
                    return {"reachable": False, "error": f"Engine service responded with HTTP {response.status}."}
                res = await response.json(content_type=None)
    except asyncio.TimeoutError:
        return {"reachable": False, "error": f"No response within {timeout:g} s."}
    except aiohttp.ClientError as e:
        return {"reachable": False, "error": f"Cannot connect: {e}"}
    except ValueError:
        return {"reachable": False, "error": "The response isn't from an engine service."}

    if not isinstance(res, dict) or res.get("status") != "healthy":
        return {"reachable": False, "error": "The response isn't from an engine service."}
    return {"reachable": True, "device": res.get("device")}


def _format_bytes(num_bytes: int | float) -> str:
    if num_bytes < 1024:
        return f"{num_bytes:.0f} B"
    elif num_bytes < 1024 * 1024:
        return f"{num_bytes / 1024:.2f} KB"
    elif num_bytes < 1024 * 1024 * 1024:
        return f"{num_bytes / (1024 * 1024):.2f} MB"
    else:
        return f"{num_bytes / (1024 * 1024 * 1024):.2f} GB"


class _TransferTracker:
    """
    Thread-safe state tracker for an in-flight upload or download transfer. It also carries the
    progress reported to the UI (see RemoteEngine.get_transfers); each field has a single writer, the
    task doing the transfer, so readers only ever see a slightly stale value.
    """
    def __init__(self, direction: str = "upload", uid: str = "", name: str = "") -> None:
        self.done_event = threading.Event()
        self.success: bool = False
        self.id = uuid.uuid4().hex
        self.direction = direction
        self.uid = uid
        self.name = name or uid
        # queued (waiting for an upload slot), active, done or failed
        self.state = "queued"
        self.bytes_done = 0
        self.total_bytes = 0
        self.files_done = 0
        self.total_files = 1
        self.attempt = 1
        self.bytes_per_second = 0.0
        self.error: str | None = None
        # Jobs waiting on this transfer, including ones that joined it in flight
        self.job_ids: set[str] = set()
        self.started_at = time.time()
        self.finished_at: float | None = None

    def add_bytes(self, count: int, seconds: float) -> None:
        self.bytes_done += count
        rate = count / max(seconds, 0.001)
        # Smoothed, so one slow chunk doesn't swing the reported speed
        self.bytes_per_second = rate if not self.bytes_per_second else 0.7 * self.bytes_per_second + 0.3 * rate

    def settle(self, success: bool, error: str | None = None) -> None:
        self.success = success
        self.state = "done" if success else "failed"
        self.error = None if success else (error or "Transfer failed; see the backend log")
        self.finished_at = time.time()
        self.done_event.set()

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "direction": self.direction,
            "uid": self.uid,
            "name": self.name,
            "state": self.state,
            "bytes_done": self.bytes_done,
            "total_bytes": self.total_bytes,
            "files_done": self.files_done,
            "total_files": self.total_files,
            "attempt": self.attempt,
            "max_attempts": UPLOAD_ATTEMPTS if self.direction == "upload" else 1,
            "bytes_per_second": round(self.bytes_per_second),
            "error": self.error,
            "job_ids": sorted(self.job_ids),
            "started_at": self.started_at,
            "finished_at": self.finished_at,
        }


class RemoteEngine(Engine):
    def __init__(self, remote_url: str, timeout: int = 300, data_root: str = None):
        super().__init__(data_root=data_root)
        self.remote_url = remote_url
        self.timeout = timeout
        self.cf_client_id = os.environ.get("CF_ACCESS_CLIENT_ID")
        self.cf_client_secret = os.environ.get("CF_ACCESS_CLIENT_SECRET")
        # In-flight transfer tracking to deduplicate concurrent uploads and downloads across threads and event loops
        self._active_uploads: dict[str, _TransferTracker] = {}
        self._active_downloads: dict[str, _TransferTracker] = {}
        self._transfer_lock = threading.Lock()
        # Every transfer reported to the UI, in flight or recently finished, by tracker id
        self._transfers: dict[str, _TransferTracker] = {}
        # Upload slots shared by every thread and event loop (see MAX_CONCURRENT_UPLOADS)
        self._upload_slots = threading.BoundedSemaphore(MAX_CONCURRENT_UPLOADS)

    def _start_transfer(self, direction: str, uid: str, name: str, job_id: str | None) -> _TransferTracker:
        """A new tracker, listed by get_transfers(). Callers hold _transfer_lock."""
        tracker = _TransferTracker(direction, uid, name)
        if job_id:
            tracker.job_ids.add(job_id)
        self._transfers[tracker.id] = tracker
        return tracker

    def get_transfers(self) -> list[dict[str, Any]]:
        """Transfers in flight, and ones that finished in the last TRANSFER_HISTORY_SECONDS, oldest first."""
        now = time.time()
        with self._transfer_lock:
            for tracker_id in [
                t.id for t in self._transfers.values()
                if t.finished_at and now - t.finished_at > TRANSFER_HISTORY_SECONDS
            ]:
                del self._transfers[tracker_id]
            return [t.to_dict() for t in sorted(self._transfers.values(), key=lambda t: t.started_at)]

    async def _wait_for_transfer(self, tracker: _TransferTracker, timeout: float | None = 600.0) -> bool:
        """
        Asynchronously waits for an in-flight transfer tracker to complete across any event loop or thread.
        A timeout of None waits until the transfer finishes.
        """
        start_time = time.time()
        while not tracker.done_event.is_set():
            if timeout is not None and time.time() - start_time > timeout:
                return False
            await asyncio.sleep(0.02)
        return tracker.success

    async def register_model(self, adapter_name: str, **kwargs) -> GraphElement:
        """Register a model by providing absolute paths to its files."""
        # This operation remains synchronous as it's not a long-running task.
        adapter_class = self._get_adapter_class(adapter_name)
        adapter_instance = adapter_class()
        model = adapter_instance.register_model(**kwargs)
        return model

    async def get_supported_operations(self) -> list[dict]:
        """Fetches supported operations from the remote engine."""
        auth_headers = self._get_auth_headers()
        timeout = aiohttp.ClientTimeout(total=self.timeout)

        try:
            async with aiohttp.ClientSession(headers=auth_headers, timeout=timeout) as session:
                async with session.get(f"{self.remote_url}/operations") as response:
                    response.raise_for_status()
                    res = await response.json()
                    return res.get("operations", [])
        except Exception as e:
            print(f"Failed to fetch supported operations from remote engine: {e}")
            # Fallback to local default if remote call fails or is not available
            return await super().get_supported_operations()

    async def get_shared_models(self) -> list[dict]:
        """Fetches shared models from the remote engine."""
        auth_headers = self._get_auth_headers()
        timeout = aiohttp.ClientTimeout(total=self.timeout)

        try:
            async with aiohttp.ClientSession(headers=auth_headers, timeout=timeout) as session:
                async with session.get(f"{self.remote_url}/shared_models") as response:
                    response.raise_for_status()
                    res = await response.json()
                    return res.get("shared_models", [])
        except Exception as e:
            print(f"Failed to fetch shared models from remote engine: {e}")
            return []

    async def add_shared_model(
        self, entry: dict, local_files: dict[str, str] | None = None, job_id: str | None = None
    ) -> dict:
        """
        Registers a shared model on the engine service, uploading any files it doesn't have yet. The
        uploads are reported under job_id, if given.
        """
        # Paths on this machine mean nothing to the engine service; it identifies files by UID
        payload = {
            k: ({rk: rv for rk, rv in v.items() if rk != "path"} if k in ("checkpoint", "encoder", "config_file") else v)
            for k, v in entry.items()
        }
        timeout = aiohttp.ClientTimeout(total=self.timeout)
        try:
            async with aiohttp.ClientSession(headers=self._get_auth_headers(), timeout=timeout) as session:
                upload_rounds = 0
                while True:
                    async with session.post(f"{self.remote_url}/shared_models", json=payload) as response:
                        if response.status == 422:
                            missing_uids = (await response.json()).get("missing_uids", [])
                            if not missing_uids or upload_rounds >= MAX_UPLOAD_ROUNDS:
                                raise Exception(f"Engine service still lacks model files: {', '.join(missing_uids)}")
                            upload_rounds += 1
                            if not await self.upload_missing_assets(missing_uids, local_files or {}, session, job_id):
                                raise Exception(
                                    f"Could not upload model file(s) {', '.join(missing_uids)}; "
                                    f"see the backend log for details."
                                )
                            continue
                        if response.status >= 400:
                            error = (await response.json()).get("error", response.reason)
                            raise Exception(f"Engine service refused the shared model: {error}")
                        return (await response.json())["shared_model"]
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            raise _request_error(e) from e

    async def remove_shared_model(self, model_id: str) -> None:
        timeout = aiohttp.ClientTimeout(total=self.timeout)
        try:
            async with aiohttp.ClientSession(headers=self._get_auth_headers(), timeout=timeout) as session:
                async with session.delete(f"{self.remote_url}/shared_models/{model_id}") as response:
                    if response.status >= 400:
                        error = (await response.json()).get("error", response.reason)
                        raise Exception(f"Engine service could not remove the shared model: {error}")
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            raise _request_error(e) from e

    def _get_auth_headers(self) -> dict:
        headers = {}
        if self.cf_client_id and self.cf_client_secret:
            headers["CF-Access-Client-Id"] = self.cf_client_id
            headers["CF-Access-Client-Secret"] = self.cf_client_secret
        return headers

    async def execute(self, operation: str, **kwargs) -> str:
        """
        Queues a remote operation and returns a job ID.
        Handles asset synchronization before queueing.
        """
        job_id = kwargs.pop('job_id', None)
        
        def _serialize_and_collect(val, collected_assets):
            if isinstance(val, GraphElement):
                collected_assets.update(val.get_local_assets())
                return val.de_anchor().to_dict()
            elif isinstance(val, list):
                return [_serialize_and_collect(v, collected_assets) for v in val]
            elif isinstance(val, dict):
                return {k: _serialize_and_collect(v, collected_assets) for k, v in val.items()}
            return val

        local_assets = {}
        de_anchored_params = _serialize_and_collect(kwargs, local_assets)

        payload = {"operation": operation, "params": de_anchored_params}
        if job_id:
            payload["job_id"] = job_id
            
        auth_headers = self._get_auth_headers()
        timeout = aiohttp.ClientTimeout(total=self.timeout)

        try:
            async with aiohttp.ClientSession(headers=auth_headers, timeout=timeout) as session:
                upload_rounds = 0
                while True:
                    async with session.post(f"{self.remote_url}/execute", data=json.dumps(payload),
                                             headers={'Content-Type': 'application/json'}) as response:
                        if response.status == 422:
                            error_details = await response.json()
                            missing_uids = error_details.get("missing_uids", [])
                            if not missing_uids:
                                response.raise_for_status()
                            if upload_rounds >= MAX_UPLOAD_ROUNDS:
                                raise Exception(
                                    f"Engine service still reports missing assets after {upload_rounds} upload "
                                    f"rounds: {', '.join(missing_uids)}"
                                )

                            upload_rounds += 1
                            if not await self.upload_missing_assets(missing_uids, local_assets, session, job_id):
                                raise Exception(
                                    f"Could not upload asset(s) required by the engine service "
                                    f"({', '.join(missing_uids)}); see the backend log for details."
                                )
                            continue
                        
                        response.raise_for_status()

                        # The execute endpoint now returns a JSON with the job_id
                        result_json = await response.json()
                        job_id = result_json.get("job_id")
                        if not job_id:
                            raise Exception("Remote execute endpoint did not return a job_id.")
                            
                        return job_id
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            raise _request_error(e) from e

    async def cancel_job(self, job_id: str) -> None:
        """Sends a cancellation request to the remote engine."""
        auth_headers = self._get_auth_headers()
        timeout = aiohttp.ClientTimeout(total=self.timeout)

        async with aiohttp.ClientSession(headers=auth_headers, timeout=timeout) as session:
            async with session.post(f"{self.remote_url}/jobs/{job_id}/cancel") as response:
                if response.status not in [200, 202]: # Accept OK or Accepted
                    # Log the error but don't raise for now, as the frontend might not handle it gracefully.
                    # The job will likely just stay in a 'cancelling' state.
                    error_text = await response.text()
                    print(f"Failed to cancel remote job {job_id}. Status: {response.status}. Body: {error_text}")
                else:
                    print(f"Successfully requested cancellation for remote job {job_id}.")

    async def _download_single_asset(self, asset_id: str, local_path: Path, session: aiohttp.ClientSession) -> bool:
        """Downloads a single asset from the remote server with atomic staging and progress reporting."""
        local_staging_path = local_path.with_suffix(local_path.suffix + ".part")
        # Registered by _fetch_result_element(); a direct call reports to a tracker nobody lists
        tracker = self._active_downloads.get(asset_id) or _TransferTracker("download", asset_id)
        try:
            local_path.parent.mkdir(parents=True, exist_ok=True)
            async with session.get(f"{self.remote_url}/download_asset/{asset_id}") as file_response:
                if file_response.status == 200:
                    total_size = int(file_response.headers.get("Content-Length", 0))
                    tracker.total_bytes = total_size
                    dl_start = time.time()
                    print(f"[RemoteEngine Download] Downloading result asset '{asset_id}'" + (f" ({_format_bytes(total_size)})..." if total_size > 0 else "..."))
                    
                    downloaded_bytes = 0
                    chunk_size = 1024 * 1024  # 1 MB
                    
                    with open(local_staging_path, "wb") as f:
                        chunk_start = time.time()
                        async for chunk in file_response.content.iter_chunked(chunk_size):
                            f.write(chunk)
                            downloaded_bytes += len(chunk)
                            tracker.add_bytes(len(chunk), time.time() - chunk_start)
                            chunk_start = time.time()
                            if total_size > 5 * 1024 * 1024:
                                pct = (downloaded_bytes / total_size) * 100
                                print(f"  -> Download progress: {_format_bytes(downloaded_bytes)} / {_format_bytes(total_size)} ({pct:.1f}%)")

                    # Atomically promote staging file to destination
                    if local_path.exists():
                        local_path.unlink()
                    local_staging_path.replace(local_path)
                    
                    dl_elapsed = max(time.time() - dl_start, 0.001)
                    dl_speed = (downloaded_bytes / (1024 * 1024)) / dl_elapsed
                    print(f"[RemoteEngine Download] Downloaded '{asset_id}' ({_format_bytes(downloaded_bytes)}) in {dl_elapsed:.2f}s ({dl_speed:.2f} MB/s)")
                    return True
                else:
                    error_text = await file_response.text()
                    print(f"[RemoteEngine Download] Failed to download asset {asset_id}. Status: {file_response.status}. Body: {error_text}")
                    tracker.error = f"Engine service replied {file_response.status}"
                    return False
        except Exception as e:
            print(f"[RemoteEngine Download] Error downloading asset '{asset_id}': {e}")
            tracker.error = str(e)
            return False
        finally:
            if local_staging_path.exists():
                try:
                    local_staging_path.unlink()
                except Exception:
                    pass

    async def _fetch_result_element(
        self, element_dict: dict, session: aiohttp.ClientSession, job_id: str | None = None
    ) -> dict:
        """
        Downloads a result element's asset (with transfer deduplication) and returns the element anchored
        to the local download directory. Raises if the download fails. job_id is the job it belongs to.
        """
        result_element = resolve_element(element_dict)

        # Save the file to a stable temporary location that won't be auto-deleted.
        tmp_root = Path(__file__).parent.parent / "tmp"
        tmp_root.mkdir(exist_ok=True)

        # Construct the path from the UID to save locally.
        base_path = path_from_uid(result_element.id)
        local_path = tmp_root / base_path

        # Download asset if not already cached locally
        if not (local_path.exists() and local_path.stat().st_size > 0):
            tracker: _TransferTracker | None = None
            is_initiator = False

            with self._transfer_lock:
                if result_element.id in self._active_downloads:
                    print(f"[RemoteEngine Download] Asset '{result_element.id}' is already being downloaded. Joining in-flight transfer...")
                    tracker = self._active_downloads[result_element.id]
                    if job_id:
                        tracker.job_ids.add(job_id)
                    is_initiator = False
                else:
                    name = getattr(result_element, "name", None) or result_element.id
                    tracker = self._start_transfer("download", result_element.id, name, job_id)
                    tracker.state = "active"
                    self._active_downloads[result_element.id] = tracker
                    is_initiator = True

            if is_initiator:
                success = False
                try:
                    success = await self._download_single_asset(result_element.id, local_path, session)
                finally:
                    with self._transfer_lock:
                        tracker.settle(success, tracker.error)
                        if self._active_downloads.get(result_element.id) is tracker:
                            del self._active_downloads[result_element.id]
            else:
                success = await self._wait_for_transfer(tracker, timeout=float(self.timeout))

            if not success:
                raise Exception(f"Failed to download asset {result_element.id}.")

        # Anchor the element's path to the root of our stable temp directory.
        return result_element.anchor(str(tmp_root), with_extension=False).to_dict()

    async def get_job_status(self, job_id: str) -> dict[str, Any]:
        """
        Polls the remote server for the status of a job.
        If the job is complete, it downloads the resulting file(s) with transfer deduplication.
        """
        auth_headers = self._get_auth_headers()
        timeout = aiohttp.ClientTimeout(total=self.timeout)

        try:
            async with aiohttp.ClientSession(headers=auth_headers, timeout=timeout) as session:
                async with session.get(f"{self.remote_url}/job_status/{job_id}") as response:
                    response.raise_for_status()

                    status_info = await response.json()

                    # If the job is complete and has a result, we need to download its file(s).
                    if status_info.get("status") == "completed" and "result" in status_info:
                        result = status_info["result"]

                        # The result from the service IS the element dictionary (or a multi-artifact result of them).
                        if is_multi_result(result):
                            elements = result["artifacts"]
                        elif result and isinstance(result, dict) and "id" in result:
                            elements = [result]
                        else:
                            return status_info  # Return as-is if there's no valid element

                        try:
                            fetched = [await self._fetch_result_element(element, session, job_id) for element in elements]
                        except Exception as e:
                            print(f"Error: {e}")
                            status_info["status"] = "failed"
                            status_info["error"] = str(e)
                            return status_info

                        # Replace the dict result with the anchored elements' dict representations.
                        status_info["result"] = {**result, "artifacts": fetched} if is_multi_result(result) else fetched[0]

                    return status_info
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            raise _request_error(e) from e

    async def _upload_single_asset(self, asset_uid: str, asset_path: str, session: aiohttp.ClientSession) -> bool:
        """
        Uploads a single asset to the remote engine. A directory is sent file by file, never
        archived, and then finalized, which checks it against its UID on the engine side.
        """
        # Registered by upload_missing_assets(); a direct call reports to a tracker nobody lists
        tracker = self._active_uploads.get(asset_uid) or _TransferTracker("upload", asset_uid)
        try:
            source = Path(asset_path)
            if not source.exists():
                print(f"[RemoteEngine Upload] File not found for upload: '{asset_path}'")
                tracker.error = f"File not found: {asset_path}"
                return False
            if source.is_dir():
                # The same files get_path_size() counts, so the total matches the asset's recorded size
                members = sorted((p for p in source.glob("**/*") if p.is_file()), key=lambda p: p.relative_to(source).as_posix())
                files = [(p, p.relative_to(source).as_posix()) for p in members]
            else:
                files = [(source, None)]
            sent: set[int] = set()
            sizes = [p.stat().st_size for p, _ in files]
            tracker.total_bytes = sum(sizes)
            tracker.total_files = len(files)

            # Wait for an upload slot without blocking this event loop
            while not self._upload_slots.acquire(blocking=False):
                await asyncio.sleep(0.05)
            tracker.state = "active"
            try:
                for attempt in range(1, UPLOAD_ATTEMPTS + 1):
                    tracker.attempt = attempt
                    # Progress restarts from the files already sent in full
                    tracker.bytes_done = sum(sizes[i] for i in sent)
                    try:
                        # A retry resumes from the first file not yet sent in full
                        for index, (path, relpath) in enumerate(files):
                            if index not in sent:
                                if await self._send_asset_chunks(asset_uid, str(path), session, relpath, tracker):
                                    print(f"[RemoteEngine Upload] '{asset_uid}' is already on the engine; stopped uploading it")
                                    return True
                                sent.add(index)
                                tracker.files_done = len(sent)
                        if source.is_dir():
                            await self._finalize_directory(asset_uid, tracker.total_bytes, session)
                        return True
                    except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                        if attempt == UPLOAD_ATTEMPTS:
                            raise
                        print(f"[RemoteEngine Upload] Attempt {attempt}/{UPLOAD_ATTEMPTS} for '{asset_uid}' failed ({e}); retrying...")
                        await asyncio.sleep(2 * attempt)
            finally:
                self._upload_slots.release()
        except Exception as e:
            print(f"[RemoteEngine Upload] Error uploading asset '{asset_uid}': {e}")
            tracker.error = str(e)
            return False

    async def _finalize_directory(self, asset_uid: str, total_size: int, session: aiohttp.ClientSession) -> None:
        """Asks the engine service to check and complete a directory asset whose files were all sent."""
        payload = {"uid": asset_uid, "total_size": total_size}
        async with session.post(f"{self.remote_url}/upload/finalize", json=payload) as response:
            if response.status == 409:
                # The staged contents don't match the UID, so sending them again would not help
                body = await response.json(content_type=None)
                raise RuntimeError(body.get("error") or f"Directory '{asset_uid}' was rejected")
            response.raise_for_status()
        print(f"[RemoteEngine Upload] Directory '{asset_uid}' verified and stored ({_format_bytes(total_size)})")

    async def _send_asset_chunks(
        self,
        asset_uid: str,
        upload_path: str,
        session: aiohttp.ClientSession,
        relpath: str | None = None,
        tracker: _TransferTracker | None = None,
    ) -> bool:
        """
        Sends a file to the engine service chunk by chunk, starting from the first chunk. With
        relpath, the file is that member of the directory asset asset_uid. Progress goes to tracker.
        Returns True, having stopped early, if the engine already has asset_uid (for example because
        it was copied into its inbox meanwhile).
        """
        asset_start = time.time()
        uploaded_asset_bytes = 0
        file_size = os.path.getsize(upload_path)
        total_chunks = max(1, (file_size + UPLOAD_CHUNK_SIZE - 1) // UPLOAD_CHUNK_SIZE)
        label = f"{asset_uid}/{relpath}" if relpath else asset_uid

        print(f"[RemoteEngine Upload] Uploading asset '{label}' ({_format_bytes(file_size)}, {total_chunks} chunk(s))...")

        with open(upload_path, 'rb') as f:
            for i in range(total_chunks):
                chunk_data = f.read(UPLOAD_CHUNK_SIZE)
                chunk_len = len(chunk_data)
                uploaded_asset_bytes += chunk_len

                data = aiohttp.FormData()
                data.add_field('uid', asset_uid)
                if relpath:
                    data.add_field('relpath', relpath)
                data.add_field('chunk_index', str(i))
                data.add_field('total_chunks', str(total_chunks))
                data.add_field('total_size', str(file_size))
                data.add_field('file', chunk_data, filename=asset_uid, content_type='application/octet-stream')

                chunk_start = time.time()
                async with session.post(f"{self.remote_url}/upload", data=data) as response:
                    response.raise_for_status()
                    try:
                        body = await response.json(content_type=None)
                    except ValueError:
                        body = None
                if isinstance(body, dict) and body.get("already_present"):
                    return True
                chunk_duration = max(time.time() - chunk_start, 0.001)
                chunk_speed = (chunk_len / (1024 * 1024)) / chunk_duration
                if tracker:
                    tracker.add_bytes(chunk_len, chunk_duration)

                pct = (uploaded_asset_bytes / file_size) * 100 if file_size > 0 else 100
                print(f"  -> Chunk {i + 1}/{total_chunks} sent: {_format_bytes(uploaded_asset_bytes)} / {_format_bytes(file_size)} ({pct:.1f}%) @ {chunk_speed:.2f} MB/s")

        asset_elapsed = max(time.time() - asset_start, 0.001)
        asset_speed = (file_size / (1024 * 1024)) / asset_elapsed
        print(f"[RemoteEngine Upload] Uploaded '{label}' in {asset_elapsed:.2f}s (avg {asset_speed:.2f} MB/s)")
        return False

    async def upload_missing_assets(
        self,
        missing_uids: list[str],
        local_assets: dict[str, str],
        session: aiohttp.ClientSession,
        job_id: str | None = None,
    ) -> bool:
        """
        Synchronizes missing assets to the remote engine, deduplicating parallel in-flight uploads across
        threads and event loops. job_id is the job waiting on them, as reported by get_transfers().
        """
        initiator_tasks: list[tuple[str, _TransferTracker, str]] = []
        waiter_trackers: list[tuple[str, _TransferTracker]] = []
        
        for asset_uid in missing_uids:
            asset_path = local_assets.get(asset_uid)
            if not asset_path:
                print(f"[RemoteEngine] Warning: could not find path for missing asset '{asset_uid}'")
                return False
            
            with self._transfer_lock:
                if asset_uid in self._active_uploads:
                    print(f"[RemoteEngine] Asset '{asset_uid}' is already being transferred by another task. Joining in-flight transfer...")
                    tracker = self._active_uploads[asset_uid]
                    if job_id:
                        tracker.job_ids.add(job_id)
                    waiter_trackers.append((asset_uid, tracker))
                else:
                    tracker = self._start_transfer("upload", asset_uid, Path(asset_path).name, job_id)
                    self._active_uploads[asset_uid] = tracker
                    initiator_tasks.append((asset_uid, tracker, asset_path))

        async def _do_upload(asset_uid: str, tracker: _TransferTracker, asset_path: str) -> bool:
            success = False
            try:
                success = await self._upload_single_asset(asset_uid, asset_path, session)
                return success
            except Exception as e:
                print(f"[RemoteEngine] Error uploading asset '{asset_uid}': {e}")
                tracker.error = str(e)
                return False
            finally:
                with self._transfer_lock:
                    tracker.settle(success, tracker.error)
                    if self._active_uploads.get(asset_uid) is tracker:
                        del self._active_uploads[asset_uid]

        upload_coros = [_do_upload(uid, tr, pth) for uid, tr, pth in initiator_tasks]
        # The uploading task always settles its tracker (its requests time out and retries are bounded),
        # so waiters follow it rather than timing out while it is queued for an upload slot
        wait_coros = [self._wait_for_transfer(tr, timeout=None) for _, tr in waiter_trackers]

        all_success = True
        if upload_coros or wait_coros:
            results = await asyncio.gather(*(upload_coros + wait_coros))
            if not all(results):
                all_success = False

        return all_success

    async def get_model_layers(self, model_element: GraphElement) -> list[dict]:
        """Inspects the model element locally to extract layers, avoiding remote queries for anonymized CAS."""
        adapter_class = self._get_adapter_class(model_element.adapter)
        adapter = adapter_class()
        try:
            adapter.load_model(model_element)
            if not hasattr(adapter, 'model') or adapter.model is None:
                raise RuntimeError("Model failed to load or does not expose PyTorch module.")
            return self._extract_model_layers(adapter.model)
        finally:
            if hasattr(adapter, 'cleanup'):
                adapter.cleanup()

    async def cluster_features(self, model_element: GraphElement, address: str, num_clusters: int) -> list[int]:
        """Dispatches feature clustering to the remote engine via the async job queue."""
        # 1. Execute the remote job
        job_id = await self.execute(
            "cluster_features",
            model_element=model_element,
            address=address,
            num_clusters=num_clusters
        )
        
        # 2. Poll for job completion
        print(f"RemoteEngine: submitted cluster_features job {job_id}. Polling for completion...")
        while True:
            status = await self.get_job_status(job_id)
            status_name = status.get("status")
            if status_name == "completed":
                result = status.get("result")
                if not isinstance(result, list):
                    raise Exception("Remote server did not return the cluster map list.")
                return result
            elif status_name == "failed":
                raise Exception(f"Remote feature clustering job failed: {status.get('error')}")
            elif status_name == "cancelled":
                raise Exception("Remote feature clustering job was cancelled.")
                
            await asyncio.sleep(1.0)
