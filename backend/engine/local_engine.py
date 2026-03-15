import inspect
from param_graph.elements.base_elements import GraphElement
from param_graph.elements.models.base import Model

from .engine import Engine
from .model_cache import ModelCache


class LocalEngine(Engine):
    def __init__(self):
        super().__init__()
        self.model_cache = ModelCache()
        
    async def register_model(self, adapter_name: str, **kwargs) -> GraphElement:
        """
        Registers a model by creating a temporary adapter and using it to 
        create the model element. The adapter is not cached.
        """
        adapter_class = self._get_adapter_class(adapter_name)
        adapter = adapter_class()
        return adapter.register_model(**kwargs)

    async def generate(self, **kwargs) -> GraphElement:
        """
        Gets a cached model adapter and uses it to generate an output.
        """
        model_element = kwargs["model_element"]
        adapter_class = self._get_adapter_class(model_element.adapter)
        adapter = self.model_cache.get(model_element, adapter_class)
        return adapter.generate(**kwargs)

    async def execute(self, operation_id: str, **kwargs) -> GraphElement:
        """
        Dispatches the operation to a method on this class.
        """
        if not hasattr(self, operation_id):
            raise Exception(f"Operation '{operation_id}' not found on LocalEngine.")
        
        func = getattr(self, operation_id)

        # Basic validation: ensure it's a public method intended to be an operation.
        if operation_id.startswith('_') or not callable(func):
            raise Exception(f"Operation '{operation_id}' is not a valid operation.")

        # We assume all operations are async methods of this class.
        if inspect.iscoroutinefunction(func):
            return await func(**kwargs)
        else:
            # If we add non-async operations, we could handle them here.
            raise Exception(f"Operation '{operation_id}' is not an async function.")
