from dataclasses import dataclass, field

from param_graph.elements.base_elements import Artifact, Asset
from param_graph.registry import register

@register('audio')
@dataclass(kw_only=True)
class Audio(Artifact):
    file: Asset
    sample_rate: int = 44100
    embeddings: dict = field(default_factory=dict)
    type: str = 'audio'
