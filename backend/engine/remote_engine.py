import aiohttp
import asyncio
import os
import json
import threading
import time
from pathlib import Path
import tempfile
from typing import Any
from param_graph.elements.base_elements import GraphElement

from .engine import Engine
from param_graph.registry import resolve_element
from utils.uid import path_from_uid


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
    """Thread-safe state tracker for an in-flight upload or download transfer."""
    def __init__(self) -> None:
        self.done_event = threading.Event()
        self.success: bool = False


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

    async def _wait_for_transfer(self, tracker: _TransferTracker, timeout: float = 600.0) -> bool:
        """Asynchronously waits for an in-flight transfer tracker to complete across any event loop or thread."""
        start_time = time.time()
        while not tracker.done_event.is_set():
            if time.time() - start_time > timeout:
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
                while True:
                    async with session.post(f"{self.remote_url}/execute", data=json.dumps(payload),
                                             headers={'Content-Type': 'application/json'}) as response:
                        if response.status == 422:
                            error_details = await response.json()
                            missing_uids = error_details.get("missing_uids", [])
                            if not missing_uids:
                                response.raise_for_status()
                            
                            can_retry = await self.upload_missing_assets(missing_uids, local_assets, session)
                            if not can_retry:
                                response.raise_for_status()
                            continue
                        
                        response.raise_for_status()

                        # The execute endpoint now returns a JSON with the job_id
                        result_json = await response.json()
                        job_id = result_json.get("job_id")
                        if not job_id:
                            raise Exception("Remote execute endpoint did not return a job_id.")
                            
                        return job_id
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            raise Exception("Cannot reach engine service. Please verify that the remote server is running.") from e

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
        try:
            local_path.parent.mkdir(parents=True, exist_ok=True)
            async with session.get(f"{self.remote_url}/download_asset/{asset_id}") as file_response:
                if file_response.status == 200:
                    total_size = int(file_response.headers.get("Content-Length", 0))
                    dl_start = time.time()
                    print(f"[RemoteEngine Download] Downloading result asset '{asset_id}'" + (f" ({_format_bytes(total_size)})..." if total_size > 0 else "..."))
                    
                    downloaded_bytes = 0
                    chunk_size = 1024 * 1024  # 1 MB
                    
                    with open(local_staging_path, "wb") as f:
                        async for chunk in file_response.content.iter_chunked(chunk_size):
                            f.write(chunk)
                            downloaded_bytes += len(chunk)
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
                    return False
        except Exception as e:
            print(f"[RemoteEngine Download] Error downloading asset '{asset_id}': {e}")
            return False
        finally:
            if local_staging_path.exists():
                try:
                    local_staging_path.unlink()
                except Exception:
                    pass

    async def get_job_status(self, job_id: str) -> dict[str, Any]:
        """
        Polls the remote server for the status of a job.
        If the job is complete, it downloads the resulting file with transfer deduplication.
        """
        auth_headers = self._get_auth_headers()
        timeout = aiohttp.ClientTimeout(total=self.timeout)

        try:
            async with aiohttp.ClientSession(headers=auth_headers, timeout=timeout) as session:
                async with session.get(f"{self.remote_url}/job_status/{job_id}") as response:
                    response.raise_for_status()

                    status_info = await response.json()

                    # If the job is complete and has a result, we need to download the file.
                    if status_info.get("status") == "completed" and "result" in status_info:
                        result_dict = status_info["result"]

                        # The result from the service IS the element dictionary.
                        element_dict = result_dict
                        if not element_dict or not isinstance(element_dict, dict) or "id" not in element_dict:
                            return status_info  # Return as-is if there's no valid element

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
                                    is_initiator = False
                                else:
                                    tracker = _TransferTracker()
                                    self._active_downloads[result_element.id] = tracker
                                    is_initiator = True

                            if is_initiator:
                                success = False
                                try:
                                    success = await self._download_single_asset(result_element.id, local_path, session)
                                finally:
                                    with self._transfer_lock:
                                        tracker.success = success
                                        tracker.done_event.set()
                                        if self._active_downloads.get(result_element.id) is tracker:
                                            del self._active_downloads[result_element.id]
                            else:
                                success = await self._wait_for_transfer(tracker, timeout=float(self.timeout))

                            if not success:
                                error_msg = f"Failed to download asset {result_element.id}."
                                print(f"Error: {error_msg}")
                                status_info["status"] = "failed"
                                status_info["error"] = error_msg
                                return status_info

                        # Anchor the element's path to the root of our stable temp directory.
                        anchored_element = result_element.anchor(str(tmp_root), with_extension=False)

                        # Replace the dict result with the anchored element's dict representation.
                        status_info["result"] = anchored_element.to_dict()

                    return status_info
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            raise Exception("Cannot reach engine service. Please verify that the remote server is running.") from e

    async def _upload_single_asset(self, asset_uid: str, asset_path: str, session: aiohttp.ClientSession) -> bool:
        """Uploads a single asset file or directory archive to the remote engine."""
        temp_zip_path = None
        try:
            if os.path.isdir(asset_path):
                print(f"[RemoteEngine Upload] Archiving directory '{asset_path}' for upload...")
                import shutil
                # Create a temporary zip archive path
                temp_fd, temp_zip_path = tempfile.mkstemp(suffix=".zip")
                os.close(temp_fd)
                # shutil.make_archive appends .zip, so strip it from the base name
                base_name = temp_zip_path[:-4] if temp_zip_path.endswith(".zip") else temp_zip_path
                shutil.make_archive(base_name, 'zip', asset_path)
                if not temp_zip_path.endswith(".zip"):
                    temp_zip_path += ".zip"
                upload_path = temp_zip_path
            else:
                upload_path = asset_path

            if not os.path.exists(upload_path):
                print(f"[RemoteEngine Upload] File not found for upload: '{upload_path}'")
                return False

            file_size = os.path.getsize(upload_path)
            chunk_size = 10 * 1024 * 1024  # 10 MB
            total_chunks = max(1, (file_size + chunk_size - 1) // chunk_size)
            
            asset_start = time.time()
            uploaded_asset_bytes = 0

            print(f"[RemoteEngine Upload] Uploading asset '{asset_uid}' ({_format_bytes(file_size)}, {total_chunks} chunk(s))...")
            
            with open(upload_path, 'rb') as f:
                for i in range(total_chunks):
                    chunk_data = f.read(chunk_size)
                    chunk_len = len(chunk_data)
                    uploaded_asset_bytes += chunk_len

                    data = aiohttp.FormData()
                    data.add_field('uid', asset_uid)
                    data.add_field('chunk_index', str(i))
                    data.add_field('total_chunks', str(total_chunks))
                    data.add_field('total_size', str(file_size))
                    data.add_field('file', chunk_data, filename=asset_uid, content_type='application/octet-stream')

                    chunk_start = time.time()
                    async with session.post(f"{self.remote_url}/upload", data=data) as response:
                        response.raise_for_status()
                    chunk_duration = max(time.time() - chunk_start, 0.001)
                    chunk_speed = (chunk_len / (1024 * 1024)) / chunk_duration

                    pct = (uploaded_asset_bytes / file_size) * 100 if file_size > 0 else 100
                    print(f"  -> Chunk {i + 1}/{total_chunks} sent: {_format_bytes(uploaded_asset_bytes)} / {_format_bytes(file_size)} ({pct:.1f}%) @ {chunk_speed:.2f} MB/s")

            asset_elapsed = max(time.time() - asset_start, 0.001)
            asset_speed = (file_size / (1024 * 1024)) / asset_elapsed
            print(f"[RemoteEngine Upload] Uploaded '{asset_uid}' in {asset_elapsed:.2f}s (avg {asset_speed:.2f} MB/s)")
            return True
        except Exception as e:
            print(f"[RemoteEngine Upload] Error uploading asset '{asset_uid}': {e}")
            return False
        finally:
            if temp_zip_path and os.path.exists(temp_zip_path):
                try:
                    os.remove(temp_zip_path)
                except Exception as e:
                    print(f"Warning: failed to clean up temp zip file {temp_zip_path}: {e}")

    async def upload_missing_assets(self, missing_uids: list[str], local_assets: dict[str, str], session: aiohttp.ClientSession) -> bool:
        """Synchronizes missing assets to the remote engine, deduplicating parallel in-flight uploads across threads and event loops."""
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
                    waiter_trackers.append((asset_uid, tracker))
                else:
                    tracker = _TransferTracker()
                    self._active_uploads[asset_uid] = tracker
                    initiator_tasks.append((asset_uid, tracker, asset_path))

        async def _do_upload(asset_uid: str, tracker: _TransferTracker, asset_path: str) -> bool:
            success = False
            try:
                success = await self._upload_single_asset(asset_uid, asset_path, session)
                return success
            except Exception as e:
                print(f"[RemoteEngine] Error uploading asset '{asset_uid}': {e}")
                return False
            finally:
                with self._transfer_lock:
                    tracker.success = success
                    tracker.done_event.set()
                    if self._active_uploads.get(asset_uid) is tracker:
                        del self._active_uploads[asset_uid]

        upload_coros = [_do_upload(uid, tr, pth) for uid, tr, pth in initiator_tasks]
        wait_coros = [self._wait_for_transfer(tr, timeout=float(self.timeout)) for _, tr in waiter_trackers]

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
