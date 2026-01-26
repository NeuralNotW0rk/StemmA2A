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
class Audio(GraphElement):
    path: str
    parent: Optional[str] = None
    alias: Optional[str] = None
    batch_index: Optional[int] = None
    tags: Optional[List[str]] = None
    tsne: Optional[List[float]] = None

    def __post_init__(self):
        self.type = 'audio'

@dataclass(kw_only=True)
class Model(GraphElement):
    # Let's assume some attributes for now, based on typical model tracking
    model_path: str
    parameters: Optional[dict] = None
    version: Optional[str] = None

    def __post_init__(self):
        self.type = 'model'

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
