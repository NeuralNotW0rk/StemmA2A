import aiohttp
import os
import json
from pathlib import Path
import tempfile
from typing import Any
from param_graph.elements.base_elements import GraphElement

from .engine import Engine
from param_graph.registry import resolve_element
from param_graph.utils import find_elements
from utils.uid import path_from_uid


class RemoteEngine(Engine):
    def __init__(self, remote_url: str, timeout: int = 300):
        super().__init__()
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
        all_elements = find_elements(kwargs)
        local_assets = {element_id: asset_path for element_id, asset_path in
                        (asset for e in all_elements.values() for asset in e.get_local_assets().items())}

        de_anchored_params = kwargs.copy()
        for name, element in all_elements.items():
            de_anchored_params[name] = element.de_anchor().to_dict()

        payload = {"operation": operation, "params": de_anchored_params}
        auth_headers = self._get_auth_headers()
        timeout = aiohttp.ClientTimeout(total=self.timeout)

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

    async def get_job_status(self, job_id: str) -> dict[str, Any]:
        """
        Polls the remote server for the status of a job.
        If the job is complete, it downloads the resulting file.
        """
        auth_headers = self._get_auth_headers()
        timeout = aiohttp.ClientTimeout(total=self.timeout)

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

                            # Creating a temporary directory that we can pass to the artifact
                            # so it gets cleaned up when the artifact is no longer in use.
                            tmp_root = Path(__file__).parent.parent / "tmp"
                            tmp_root.mkdir(exist_ok=True)
                            temp_dir = tempfile.TemporaryDirectory(dir=tmp_root)
                            temp_dir_path = Path(temp_dir.name)

                            # We need to reconstruct the path from the UID to save locally
                            base_path = path_from_uid(result_element.id)
                            local_path = temp_dir_path / base_path

                            local_path.parent.mkdir(parents=True, exist_ok=True)
                            local_path.write_bytes(file_data)

                            # Anchor the element to the new local path
                            anchored_element = result_element.anchor(str(temp_dir_path), with_extension=False)
                            # Attach the temporary directory reference for cleanup
                            anchored_element._temp_dir_ref = temp_dir

                            # Replace the dict result with the anchored element's dict representation
                            status_info["result"] = anchored_element.to_dict()
                        else:
                            # If download fails, update status to reflect that
                            status_info["status"] = "failed"
                            status_info["error"] = f"Failed to download asset {result_element.id}"

                return status_info

    async def upload_missing_assets(self, missing_uids: list[str], local_assets: dict[str, str], session: aiohttp.ClientSession) -> bool:
        paths_to_upload = []
        for asset_uid in missing_uids:
            asset_path = local_assets.get(asset_uid)
            if not asset_path:
                print(f"Warning: could not find path for missing uid {asset_uid}")
                return False
            paths_to_upload.append((asset_uid, asset_path))

        for asset_uid, asset_path in paths_to_upload:
            data = aiohttp.FormData()
            data.add_field('file',
                           open(asset_path, 'rb'),
                           filename=asset_uid,
                           content_type='application/octet-stream')

            async with session.post(f"{self.remote_url}/upload", data=data) as response:
                response.raise_for_status()
        
        return True
