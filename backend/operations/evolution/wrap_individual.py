"""Agnostic operation for wrapping precursor artifacts into evolutionary Individual nodes."""

from typing import Any, Optional
from param_graph.elements.artifacts.individual_element import Individual
from param_graph.elements.base_elements import Asset


def wrap_artifact_as_individual(
    individual_id: str,
    name: str,
    asset_path: str,
    base_model_id: str,
    baseline_grating_id: Optional[str] = None,
    generation: int = 0,
    fitness: Optional[float] = None,
    context: Optional[dict[str, Any]] = None,
    extension: str = ".safetensors",
) -> Individual:
    """
    Constructs a graph-level Individual artifact representing a baseline or precursor individual.

    Args:
        individual_id: Deterministic or unique ID for the individual node.
        name: Human-readable display name.
        asset_path: File path where the serialized genotype asset is stored.
        base_model_id: ID of the base model this individual binds to.
        baseline_grating_id: Optional ID of the baseline grating node.
        generation: Initial generation number (defaults to 0 for baseline).
        fitness: Initial fitness evaluation (None if unevaluated).
        context: Metadata context dictionary preserving provenance.
        extension: File extension of the genotype asset (.safetensors).

    Returns:
        An Individual graph artifact instance ready to be added to ParameterGraph.
    """
    return Individual(
        id=individual_id,
        name=name,
        file=Asset(path=asset_path, uid=individual_id, extension=extension),
        base_model_id=base_model_id,
        baseline_grating_id=baseline_grating_id,
        generation=generation,
        fitness=fitness,
        context=context or {},
    )
