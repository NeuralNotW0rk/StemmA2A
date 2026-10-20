from dataclasses import dataclass, field
from param_graph.elements.base_elements import Artifact, Asset
from param_graph.registry import register

@register('individual')
@dataclass(kw_only=True)
class Individual(Artifact):
    """
    Represents an individual in an evolutionary population.
    Stores the genotype itself as a safetensors asset.
    """
    # The asset file containing the serialized LoRAGenome.
    file: Asset
    # The ID of the base model this individual corresponds to.
    base_model_id: str
    # The ID of the baseline grating this individual mutated from.
    baseline_grating_id: str | None = None
    # The generation number/index.
    generation: int = 0
    # Optional fitness score/evaluation.
    fitness: float | None = None
    type: str = 'individual'
