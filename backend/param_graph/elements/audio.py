from dataclasses import dataclass, field
from typing import Optional, List

from base_elements import Artifact

@dataclass(kw_only=True)
class Model(Artifact):
    path: str
    type: str = 'audio'