from dataclasses import dataclass
from typing import Optional, List

from .elements.base_elements import Artifact

@dataclass(kw_only=True)
class Model(Artifact):
    engine: str

    def __post_init__(self):
        super().__post_init__()
        self.type = 'model'

class Engine:
    def __init__(self) -> None:
        pass

    def register_model(self) -> Model:
        pass

    def register_model_default(self) -> Model:
        pass

    def generate(self) -> None:
        pass