from dataclasses import dataclass, asdict, field
from typing import Optional, List
from time import time

@dataclass
class GraphElement:
    type: str
    created: int

    def __post_init__(self):
        self.created = int(time())

    def to_dict(self):
        return asdict(self)
    
@dataclass(kw_only=True)
class Artifact(GraphElement):
    uid: int
    alias: Optional[str] = None

    def __post_init__(self):
        super().__post_init__()
        self.type = 'artifact'

@dataclass(kw_only=True)
class ExternalSource(GraphElement):
    path: str

    def __post_init__(self):
        super().__post_init__()
        self.type = 'external'

@dataclass(kw_only=True)
class Set(GraphElement):
    alias: str

    def __post_init__(self):
        super().__post_init__()
        self.type = 'set'

@dataclass(kw_only=True)
class Batch(GraphElement):
    alias: str

    def __post_init__(self):
        super().__post_init__()
        self.type = 'batch'
