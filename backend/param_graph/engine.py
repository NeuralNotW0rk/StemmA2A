from abc import ABC, abstractmethod
from dataclasses import dataclass

from .elements.base_elements import Artifact
from .uid_gen import UIDGenerator, XXH3_64

@dataclass(kw_only=True)
class Model(Artifact):
    engine: str
    type: str = 'model'

class Engine(ABC):
    def __init__(self, uid_generator: UIDGenerator = None) -> None:
        if uid_generator is None:
            # Set default uid generator (XXH3_64)
            self.uid_generator = XXH3_64()
        else:
            self.uid_generator = uid_generator


    @abstractmethod
    def register_model(self, **kwargs) -> Model:
        pass

    @abstractmethod
    def generate(self) -> None:
        pass