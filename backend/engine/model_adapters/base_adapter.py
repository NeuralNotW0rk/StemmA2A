from abc import ABC, abstractmethod
import json
import os
import inspect
import torch

from param_graph.elements.base_elements import GraphElement
from param_graph.elements.models.base_model_element import Model
from utils.uid import UIDGenerator, XXH3_64

def operation(name: str, is_standard: bool = True, description: str = "", initiator_types: list = None, context_overrides: dict = None):
    """Decorator to mark and register an adapter method as a supported operation."""
    def decorator(func):
        func._is_operation = True
        func._op_name = name
        func._op_is_standard = is_standard
        func._op_description = description
        func._op_initiator_types = initiator_types or []
        func._op_context_overrides = context_overrides or {}
        return func
    return decorator

class ModelAdapter(ABC):
    def __init__(self, uid_generator: UIDGenerator = None) -> None:
        if uid_generator is None:
            # Set default uid generator (XXH3_64)
            self.uid_generator = XXH3_64()
        else:
            self.uid_generator = uid_generator
        self.name = ""

    def get_form_config(self):
        engine_file = inspect.getfile(self.__class__)
        config_filename = f"{self.name}.json"
        config_path = os.path.join(os.path.dirname(engine_file), config_filename)
        with open(config_path, 'r') as f:
            return json.load(f)

    @abstractmethod
    def register_model(self, **kwargs) -> Model:
        pass
    
    @abstractmethod
    def load_model(self, info: Model, verify: bool = True):
        pass

    @abstractmethod
    def generate(self, **kwargs) -> tuple[GraphElement, torch.Tensor]:
        pass

    @abstractmethod
    def invert(self, **kwargs) -> tuple[GraphElement, torch.Tensor]:
        pass

    def cleanup(self):
        """Called to explicitly clean up resources when the adapter is removed from the cache."""
        pass
