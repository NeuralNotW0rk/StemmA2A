from dataclasses import dataclass, field

from param_graph.elements.base_elements import Artifact, Asset
from param_graph.registry import register

@register('image')
@dataclass(kw_only=True)
class Image(Artifact):
    file: Asset
    width: int = 512
    height: int = 512
    embeddings: dict = field(default_factory=dict)
    type: str = 'image'
