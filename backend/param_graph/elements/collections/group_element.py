from dataclasses import dataclass

from param_graph.elements.base_elements import Collection
from param_graph.registry import register

@register('group')
@dataclass(kw_only=True)
class Group(Collection):
    member_type: str | None = None
    type: str = 'group'
