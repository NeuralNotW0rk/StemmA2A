from dataclasses import dataclass
from typing import Optional, List

from base_elements import Artifact

@dataclass(kw_only=True)
class Model(Artifact):
    config_path: Optional[str] = None
    engine: str

    def __post_init__(self):
        self.type = 'model'