from dataclasses import dataclass, field

from .collection_element import Collection

@dataclass(kw_only=True)
class Batch(Collection):
    shared_context: dict = field(default_factory=dict)
    type: str = 'batch'