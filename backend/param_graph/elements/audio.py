from dataclasses import dataclass
from typing import Optional, List

from base_elements import Artifact

@dataclass(kw_only=True)
class Audio(Artifact):
    tsne: Optional[List[float]] = None

    def __post_init__(self):
        self.type = 'audio'