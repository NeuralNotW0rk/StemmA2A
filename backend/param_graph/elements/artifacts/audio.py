from dataclasses import dataclass

from param_graph.elements.base_elements import Artifact, Asset
from param_graph.registry import register

@register('audio')
@dataclass(kw_only=True)
class Audio(Artifact):
    file: Asset
    sample_rate: int = 44100
    type: str = 'audio'
