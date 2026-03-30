from dataclasses import dataclass, field

from param_graph.elements.base_elements import GraphElement
from param_graph.registry import register

@dataclass(kw_only=True)
class Collection(GraphElement):
    children: dict[str, GraphElement] = field(default_factory=dict)
    type: str = 'collection'
