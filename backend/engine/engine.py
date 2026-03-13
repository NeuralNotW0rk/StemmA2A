from abc import ABC, abstractmethod
import aiohttp

from .model_adapter import ModelAdapter
from param_graph.graph import ParameterGraph
from param_graph.elements.artifacts.audio import Audio


class Engine(ABC):
    @abstractmethod
    async def execute(self, adapter: ModelAdapter, **kwargs):
        pass

class LocalEngine(Engine):
    async def execute(self, adapter: ModelAdapter, output_dir: str, **kwargs):
        # Directly call the adapter's generate method
        return adapter.generate(output_dir=output_dir, **kwargs)

class RemoteEngine(Engine):
    def __init__(self, remote_url: str, graph: ParameterGraph):
        self.remote_url = remote_url
        self.graph = graph

    async def execute(self, adapter: ModelAdapter, output_dir: str, **kwargs):
        # Prepare the payload
        payload = {
            "adapter": adapter.name,
            "adapter_params": adapter.model_info.to_dict(),
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
                        
                        await self.upload_missing_assets(missing_hashes, session)
                        # Retry the request after uploading
                        continue
                    else:
                        raise e

    async def upload_missing_assets(self, missing_hashes: list, session: aiohttp.ClientSession):
        for asset_hash in missing_hashes:
            asset_path = self.graph.get_path_from_name(asset_hash, relative=True)
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
                pass
