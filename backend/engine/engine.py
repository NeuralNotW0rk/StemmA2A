from abc import ABC, abstractmethod
from param_graph.elements.base_elements import GraphElement

from .model_adapters.stable_audio import StableAudioAdapter

class Engine(ABC):
    def __init__(self):
        self.adapter_registry = {'stable_audio_tools': StableAudioAdapter}

    def _get_adapter_class(self, adapter_name: str):
        adapter_class = self.adapter_registry.get(adapter_name)
        if not adapter_class:
            raise Exception(f"Adapter '{adapter_name}' not found in adapter_registry.")
        return adapter_class

    async def get_adapter_config(self, adapter_name: str) -> dict:
        """Get the form configuration for a specific adapter."""
        adapter_class = self._get_adapter_class(adapter_name)
        # Instantiate the adapter just to get the config
        adapter_instance = adapter_class()
        if not hasattr(adapter_instance, 'get_form_config'):
            raise Exception(f"Adapter '{adapter_name}' does not have a form configuration")

        return adapter_instance.get_form_config()

    @abstractmethod
    async def execute(self, operation_id: str, **kwargs) -> GraphElement:
        pass

    @abstractmethod
    async def register_model(self, adapter_name: str, **kwargs) -> GraphElement:
        pass
