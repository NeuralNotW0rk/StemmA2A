from dataclasses import dataclass

from param_graph.elements.base_elements import Collection
from param_graph.registry import register

@register('directory')
@dataclass(kw_only=True)
class Directory(Collection):
    """
    Represents an external directory. 
    Can act as a compound parent node in the frontend visualization.
    """
    name: str
    path: str
    type: str = 'directory'