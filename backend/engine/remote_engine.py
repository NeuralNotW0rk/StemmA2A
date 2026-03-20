import aiohttp
import os
import json
from pathlib import Path
import tempfile
from param_graph.elements.base_elements import GraphElement

from .engine import Engine
from param_graph.registry import resolve_element
from utils.uid_utils import path_from_uid

def find_elements(d: dict) -> dict[str, GraphElement]:
    elements = {}
    for k, v in d.items():
        if isinstance(v, GraphElement):
            elements[k] = v
        elif isinstance(v, dict):
            # For now, we won't recurse into dicts.
            # This can be expanded if we have nested elements.
            pass 
    return elements


class RemoteEngine(Engine):
    def __init__(self, remote_url: str):
        super().__init__()
        self.remote_url = remote_url
        self.cf_client_id = os.environ.get("CF_ACCESS_CLIENT_ID")
        self.cf_client_secret = os.environ.get("CF_ACCESS_CLIENT_SECRET")

    async def register_model(self, adapter_name: str, **kwargs) -> GraphElement:
        """Register a model by providing absolute paths to its files."""
        adapter_class = self._get_adapter_class(adapter_name)
        adapter_instance = adapter_class()
        model = adapter_instance.register_model(**kwargs)
        # The adapter_instance is temporary and will be garbage collected.
        # This should release any memory it was using.
        return model

    def _get_auth_headers(self) -> dict:
        headers = {}
        if self.cf_client_id and self.cf_client_secret:
            headers["CF-Access-Client-Id"] = self.cf_client_id
            headers["CF-Access-Client-Secret"] = self.cf_client_secret
        return headers

    async def execute(self, operation: str, **kwargs) -> GraphElement:
        # 1. Gather all local assets from all graph elements in the parameters.
        # This gives us a complete uid -> path mapping for anything we might need to upload.
        all_elements = find_elements(kwargs)
        local_assets = {}
        for element in all_elements.values():
            local_assets.update(element.get_local_assets())

        # 2. De-anchor all elements before serialization.
        # This removes local paths, which are not meaningful on the remote server.
        anchored_params = kwargs.copy()
        for name, element in all_elements.items():
            anchored_params[name] = element.de_anchor()

        # 3. Prepare the payload for the remote execution endpoint.
        payload = {
            "operation": operation,
            "params": anchored_params,
        }

        auth_headers = self._get_auth_headers()
        
        async with aiohttp.ClientSession(headers=auth_headers) as session:
            while True:
                # Use a custom json serializer that can handle dataclasses
                # This is a bit of a hack, we should probably have a proper serializer
                import json
                from dataclasses import is_dataclass, asdict
                class DataclassEncoder(json.JSONEncoder):
                    def default(self, o):
                        if is_dataclass(o):
                            return asdict(o)
                        return super().default(o)
                
                async with session.post(f"{self.remote_url}/execute", data=json.dumps(payload, cls=DataclassEncoder),
                                        headers={'Content-Type': 'application/json'}) as response:
                    if response.status == 422: # Missing assets
                        error_details = await response.json()
                        missing_uids = error_details.get("missing_uids", [])
                        if not missing_uids:
                            response.raise_for_status() # Re-raise if the error is not about missing assets
                        
                        can_retry = await self.upload_missing_assets(missing_uids, local_assets, session)
                        if not can_retry:
                            # If we couldn't upload all missing assets, we're in a bad state.
                            # Re-raise the original error instead of looping.
                            response.raise_for_status()
                        
                        # Retry the request after uploading
                        continue
                    
                    response.raise_for_status()

                    # 1. Get the de-anchored graph element from the header
                    element_json = response.headers.get('X-Graph-Element')
                    if not element_json:
                        raise Exception("Missing 'X-Graph-Element' header in response.")
                    element_dict = json.loads(element_json)
                    result_element = resolve_element(element_dict)
                    
                    # 2. Get the file data from the body
                    file_data = await response.read()

                    # 3. Save the file to a local temporary directory
                    temp_dir = tempfile.TemporaryDirectory()
                    temp_dir_path = Path(temp_dir.name)

                    # We assume the first UID is the one for the file we just downloaded
                    uid = result_element.get_uids()[0]
                    local_path = temp_dir_path / path_from_uid(uid)
                    local_path.parent.mkdir(parents=True, exist_ok=True)
                    local_path.write_bytes(file_data)
                    
                    # 4. Anchor the element to the new local path
                    anchored_element = result_element.anchor(temp_dir_path)

                    # 5. Store a reference to the TemporaryDirectory object to prevent
                    # it from being garbage collected and deleting the file.
                    anchored_element._temp_dir_ref = temp_dir

                    return anchored_element

    async def upload_missing_assets(self, missing_uids: list[str], local_assets: dict[str, str], session: aiohttp.ClientSession) -> bool:
        paths_to_upload = []
        for asset_uid in missing_uids:
            asset_path = local_assets.get(asset_uid)
            if not asset_path:
                print(f"Warning: could not find path for missing uid {asset_uid}")
                return False # Abort if any asset is missing.
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
