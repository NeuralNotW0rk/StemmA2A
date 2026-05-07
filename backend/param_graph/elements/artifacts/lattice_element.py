from dataclasses import dataclass, field
from typing import List, Dict, Any

from param_graph.registry import register
from param_graph.elements.base_elements import Artifact, Asset

@register('lattice')
@dataclass(kw_only=True)
class Lattice(Artifact):
    """
    Represents a Diffracture Lattice, which is a collection of 'prisms'
    that apply transformations to model layers. This can be used to
    represent things like LoRAs.
    """
    # The asset file containing the lattice weights (e.g., a .safetensors file for a LoRA).
    file: Asset
    # The ID of the base model this lattice is designed to be applied to.
    # This helps ensure compatibility.
    base_model_id: str
    # The internal structure of the lattice, as a list of prism configurations.
    prisms: List[Dict[str, Any]] = field(default_factory=list)
    type: str = 'lattice'