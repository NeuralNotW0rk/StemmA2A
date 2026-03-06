from dataclasses import dataclass, asdict, field
from typing import Optional, List
from time import time
from ..registry import register

@dataclass(kw_only=True)
class GraphElement:
    type: str
    created: int = field(default_factory=lambda: int(time()))

    def to_dict(self):
        return asdict(self)
    
@dataclass(kw_only=True)
class Artifact(GraphElement):
    id: str
    name: str
    type: str = 'artifact'

@register('external')
@dataclass(kw_only=True)
class ExternalSource(GraphElement):
    path: str
    type: str = 'external'

@register('set')
@dataclass(kw_only=True)
class Set(GraphElement):
    alias: str
    type: str = 'set'

@register('batch')
@dataclass(kw_only=True)
class Batch(GraphElement):
    alias: str
    type: str = 'batch'
