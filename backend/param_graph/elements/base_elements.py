from dataclasses import dataclass, asdict, field
from typing import Optional, List
from time import time

@dataclass(kw_only=True)
class GraphElement:
    type: str
    created: int = field(default_factory=lambda: int(time()))

    def to_dict(self):
        return asdict(self)
    
@dataclass(kw_only=True)
class Artifact(GraphElement):
    uid: str
    uid_type: str
    uid_version: str
    name: str
    type: str = 'artifact'

@dataclass(kw_only=True)
class ExternalSource(GraphElement):
    path: str
    type: str = 'external'

@dataclass(kw_only=True)
class Set(GraphElement):
    alias: str
    type: str = 'set'

@dataclass(kw_only=True)
class Batch(GraphElement):
    alias: str
    type: str = 'batch'
