from dataclasses import dataclass, field
from typing import List, Dict, Any

from param_graph.registry import register
from param_graph.elements.base_elements import Artifact, Asset

@register('grating')
@dataclass(kw_only=True)
class Grating(Artifact):
    """
    Represents a Diffracture Grating, which is a collection of 'elements'
    that apply transformations to model layers. This can be used to
    represent things like LoRAs.
    """
    # The asset file containing the grating weights (e.g., a .safetensors file for a LoRA).
    file: Asset
    # The ID of the base model this grating is designed to be applied to.
    # This helps ensure compatibility.
    base_model_id: str
    # The internal structure of the grating, as a list of element configurations.
    elements: List[Dict[str, Any]] = field(default_factory=list)
    type: str = 'grating'
