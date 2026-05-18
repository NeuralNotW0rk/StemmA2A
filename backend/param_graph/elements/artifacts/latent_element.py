from dataclasses import dataclass

from param_graph.elements.base_elements import Artifact, Asset
from param_graph.registry import register

@register('latent')
@dataclass(kw_only=True)
class Latent(Artifact):
    """
    Represents a latent tensor resulting from DDIM inversion or other encodings.
    """
    file: Asset
    type: str = 'latent'