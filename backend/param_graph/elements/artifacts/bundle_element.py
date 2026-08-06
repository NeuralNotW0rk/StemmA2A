from dataclasses import dataclass, field

from param_graph.elements.base_elements import Artifact
from param_graph.registry import register

@register('bundle')
@dataclass(kw_only=True)
class Bundle(Artifact):
    member_ids: list[str] = field(default_factory=list)
    member_type: str | None = None
    type: str = 'bundle'
