import traceback

from param_graph.elements.models.base import Model

from .engine import Engine
from param_graph.elements.artifacts.audio import Audio
from .model_adapters.base import ModelAdapter
from .model_adapters.stable_audio import StableAudioAdapter


class LocalEngine(Engine):
    def __init__(self):
        self.adapter_registry = {'stable_audio_tools': StableAudioAdapter}
        self.active_adapter: ModelAdapter | None = None

    def _set_adapter(self, adapter_name: str) -> bool:
        """
        Sets the global adapter instance. Avoids re-initializing the adapter if it's already active.
        Returns True on success, False on failure.
        """
        if self.active_adapter and self.active_adapter.name == adapter_name:
            return True

        adapter_class = self.adapter_registry.get(adapter_name)
        if not adapter_class:
            print(f"Error: Adapter '{adapter_name}' not found in adapter_registry.")
            return False

        try:
            self.active_adapter = adapter_class()
            print(f"Adapter activated: {self.active_adapter.name}")
            return True
        except Exception as e:
            print(f"Error initializing adapter '{adapter_name}': {e}")
            traceback.print_exc()
            self.active_adapter = None
            return False

    async def get_adapter_config(self, adapter_name: str) -> dict:
        """Get the form configuration for a specific adapter."""
        if not self._set_adapter(adapter_name):
            raise Exception(f"Failed to set adapter '{adapter_name}'.")

        if not hasattr(self.active_adapter, 'get_form_config'):
            raise Exception(f"Adapter '{adapter_name}' does not have a form configuration")

        return self.active_adapter.get_form_config()

    async def register_model(self, adapter_name: str, **kwargs) -> Model:
        """Register a model by providing absolute paths to its files."""
        if not self._set_adapter(adapter_name):
            raise Exception(f"Failed to set adapter '{adapter_name}'.")

        return self.active_adapter.register_model(**kwargs)

    async def execute(self, model_element: Model, output_dir: str, **kwargs) -> Audio:
        """
        Executes the generation process.
        """
        if not self._set_adapter(model_element.adapter):
            raise Exception(f"Failed to set adapter '{model_element.adapter}'.")

        self.active_adapter.load_model(model_element)
        return self.active_adapter.generate(output_dir=output_dir, **kwargs)
