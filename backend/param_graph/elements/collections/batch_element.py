from dataclasses import dataclass, field

from param_graph.elements.base_elements import Collection
from param_graph.registry import register

@register('batch')
@dataclass(kw_only=True)
class Batch(Collection):
    member_type: str
    type: str = 'batch'