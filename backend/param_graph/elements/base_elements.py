from dataclasses import dataclass, asdict, field
from typing import Optional, List
from time import time

@dataclass
class GraphElement:
    type: str = field(init=False)
    created: int = field(default_factory=lambda: int(time()))

    def to_dict(self):
        return asdict(self)
    
@dataclass(kw_only=True)
class Artifact(GraphElement):
    path: str
    uid: int
    uid_type: str
    alias: Optional[str] = None
    tags: Optional[List[str]] = None
    def __post_init__(self):
        self.type = 'artifact'

@dataclass(kw_only=True)
class ExternalSource(GraphElement):
    path: str

    def __post_init__(self):
        self.type = 'external'

@dataclass(kw_only=True)
class Set(GraphElement):
    alias: str

    def __post_init__(self):
        self.type = 'set'

@dataclass(kw_only=True)
class Batch(GraphElement):
    alias: str

    def __post_init__(self):
        self.type = 'batch'
