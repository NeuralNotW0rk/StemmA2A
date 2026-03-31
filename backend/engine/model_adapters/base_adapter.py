from abc import ABC, abstractmethod
import json
import os
import inspect
import torch

from param_graph.elements.base_elements import GraphElement
from param_graph.elements.models.base_model_element import Model
from utils.uid import UIDGenerator, XXH3_64

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
