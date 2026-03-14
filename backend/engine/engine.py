from abc import ABC, abstractmethod
from param_graph.elements.models.base import Model


class Engine(ABC):
    @abstractmethod
    async def execute(self, model_element: Model, **kwargs):
        pass

    @abstractmethod
    async def register_model(self, adapter_name: str, **kwargs) -> Model:
        pass

    @abstractmethod
    async def get_adapter_config(self, adapter_name: str) -> dict:
        pass

