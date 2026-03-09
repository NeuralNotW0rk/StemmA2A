from dataclasses import dataclass

from param_graph.elements.base_elements import Artifact
from param_graph.registry import register

@register('audio')
@dataclass(kw_only=True)
class Audio(Artifact):
    path: str
    type: str = 'audio'
