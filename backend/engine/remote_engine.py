import aiohttp
import asyncio
import os
import json
from pathlib import Path
import tempfile
from typing import Any
from param_graph.elements.base_elements import GraphElement

from .engine import Engine
from param_graph.registry import resolve_element
from utils.uid import path_from_uid


class RemoteEngine(Engine):
    def __init__(self, remote_url: str, timeout: int = 300, data_root: str = None):
        super().__init__(data_root=data_root)
        self.remote_url = remote_url
        self.timeout = timeout
        self.cf_client_id = os.environ.get("CF_ACCESS_CLIENT_ID")
        self.cf_client_secret = os.environ.get("CF_ACCESS_CLIENT_SECRET")

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

    async def get_job_status(self, job_id: str) -> dict[str, Any]:
        """
        Polls the remote server for the status of a job.
        If the job is complete, it downloads the resulting file.
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

                        async with session.get(f"{self.remote_url}/download_asset/{result_element.id}") as file_response:
                            if file_response.status == 200:
                                file_data = await file_response.read()

                                # Save the file to a stable temporary location that won't be auto-deleted.
                                tmp_root = Path(__file__).parent.parent / "tmp"
                                tmp_root.mkdir(exist_ok=True)

                                # Construct the path from the UID to save locally.
                                base_path = path_from_uid(result_element.id)
                                local_path = tmp_root / base_path

                                local_path.parent.mkdir(parents=True, exist_ok=True)
                                local_path.write_bytes(file_data)

                                # Anchor the element's path to the root of our stable temp directory.
                                anchored_element = result_element.anchor(str(tmp_root), with_extension=False)

                                # Replace the dict result with the anchored element's dict representation.
                                status_info["result"] = anchored_element.to_dict()
                            else:
                                # If download fails, update status to reflect that
                                status_info["status"] = "failed"
                                status_info["error"] = f"Failed to download asset {result_element.id}"

                    return status_info
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            raise Exception("Cannot reach engine service. Please verify that the remote server is running.") from e

    async def upload_missing_assets(self, missing_uids: list[str], local_assets: dict[str, str], session: aiohttp.ClientSession) -> bool:
        paths_to_upload = []
        for asset_uid in missing_uids:
            asset_path = local_assets.get(asset_uid)
            if not asset_path:
                print(f"Warning: could not find path for missing uid {asset_uid}")
                return False
            paths_to_upload.append((asset_uid, asset_path))

        chunk_size = 10 * 1024 * 1024  # 10 MB
        for asset_uid, asset_path in paths_to_upload:
            temp_zip_path = None
            if os.path.isdir(asset_path):
                print(f"Archiving directory {asset_path} for upload...")
                import tempfile
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

            file_size = os.path.getsize(upload_path)
            total_chunks = max(1, (file_size + chunk_size - 1) // chunk_size)
            
            try:
                with open(upload_path, 'rb') as f:
                    for i in range(total_chunks):
                        chunk_data = f.read(chunk_size)
                        data = aiohttp.FormData()
                        data.add_field('uid', asset_uid)
                        data.add_field('chunk_index', str(i))
                        data.add_field('total_chunks', str(total_chunks))
                        data.add_field('total_size', str(file_size))
                        data.add_field('file', chunk_data, filename=asset_uid, content_type='application/octet-stream')

                        async with session.post(f"{self.remote_url}/upload", data=data) as response:
                            response.raise_for_status()
            finally:
                if temp_zip_path and os.path.exists(temp_zip_path):
                    try:
                        os.remove(temp_zip_path)
                    except Exception as e:
                        print(f"Warning: failed to clean up temp zip file {temp_zip_path}: {e}")
        
        return True

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
