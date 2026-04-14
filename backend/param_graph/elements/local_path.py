from dataclasses import dataclass

from param_graph.elements.base_elements import GraphElement
from param_graph.registry import register

@register('local_path')
@dataclass(kw_only=True)
class LocalPath(GraphElement):
    """
    Represents a local path (folder or file). 
    Can be expanded into a connected Directory compound node.
    """
    name: str
    path: str
    type: str = 'local_path'