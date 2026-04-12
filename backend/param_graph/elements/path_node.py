from dataclasses import dataclass

from param_graph.elements.base_elements import GraphElement
from param_graph.registry import register

@register('path_node')
@dataclass(kw_only=True)
class PathNode(GraphElement):
    """
    Represents a local path (folder or file). 
    Can be expanded into a connected Directory compound node.
    """
    name: str
    path: str
    type: str = 'path_node'