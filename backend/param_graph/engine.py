from dataclasses import dataclass, field
from typing import Optional, List

from .elements.base_elements import Artifact
from .uid_gen import UIDGenerator, XXH3_64

@dataclass(kw_only=True)
class Model(Artifact):
    engine: str = field(init=False, default="")
    type: str = 'model'

class Engine:
    def __init__(self, uid_generator: UIDGenerator = None) -> None:
        if uid_generator is None:
            # Set default uid generator (XXH3_64)
            self.uid_generator = XXH3_64()
        else:
            self.uid_generator = uid_generator


    def register_model(self) -> Model:
        pass

    def register_model_default(self) -> Model:
        pass

    def generate(self) -> None:
        pass