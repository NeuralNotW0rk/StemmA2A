import aiohttp

from param_graph.elements.models.base import Model

from .engine import Engine
from param_graph.elements.artifacts.audio import Audio


class RemoteEngine(Engine):
    def __init__(self, remote_url: str):
        self.remote_url = remote_url

    async def get_adapter_config(self, adapter_name: str) -> dict:
        """Get the form configuration for a specific adapter from the remote engine."""
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{self.remote_url}/adapter_config/{adapter_name}") as response:
                response.raise_for_status()
                return await response.json()

    async def register_model(self, adapter_name: str, **kwargs) -> Model:
        """Register a model on the remote engine."""
        payload = {
            "adapter": adapter_name,
            "params": kwargs
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(f"{self.remote_url}/register_model", json=payload) as response:
                response.raise_for_status()
                json_response = await response.json()
                return Model(**json_response)

    async def execute(self, model_element: Model, output_dir: str, **kwargs) -> Audio:
        # Remove local paths
        model_element = model_element.de_anchor()

        # Prepare the payload
        payload = {
            "adapter_params": model_element.to_dict(),
            "generation_params": kwargs,
            "output_dir": output_dir
        }

        async with aiohttp.ClientSession() as session:
            while True:
                try:
                    async with session.post(f"{self.remote_url}/execute", json=payload) as response:
                        response.raise_for_status()
                        json_response = await response.json()
                        return Audio(**json_response)
                except aiohttp.ClientResponseError as e:
                    if e.status == 422: # Missing assets
                        error_details = await e.json()
                        missing_hashes = error_details.get("missing_hashes", [])
                        if not missing_hashes:
                            raise e # Re-raise if the error is not about missing assets
                        
                        await self.upload_missing_assets(model_element, missing_hashes, session)
                        # Retry the request after uploading
                        continue
                    else:
                        raise e

    async def upload_missing_assets(self, model_element: Model, missing_hashes: list, session: aiohttp.ClientSession):
        # Create a mapping from hash to path from the model_element
        hash_to_path = {}
        # This could be more generic, but for now we'll handle the known asset types
        if hasattr(model_element, 'checkpoint_hash') and hasattr(model_element, 'checkpoint_path'):
            hash_to_path[model_element.checkpoint_hash] = model_element.checkpoint_path
        if hasattr(model_element, 'config_hash') and hasattr(model_element, 'config_path'):
            hash_to_path[model_element.config_hash] = model_element.config_path

        for asset_hash in missing_hashes:
            asset_path = hash_to_path.get(asset_hash)
            if asset_path:
                data = aiohttp.FormData()
                data.add_field('file',
                               open(asset_path, 'rb'),
                               filename=asset_hash,
                               content_type='application/octet-stream')

                async with session.post(f"{self.remote_url}/upload", data=data) as response:
                    response.raise_for_status()
            else:
                # Handle cases where the asset is not found in the graph
                # This could be an error condition
                print(f"Warning: could not find path for missing hash {asset_hash} in model_element")
                pass
