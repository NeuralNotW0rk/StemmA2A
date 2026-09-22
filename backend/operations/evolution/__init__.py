"""Agnostic evolutionary operations package."""

import json
from pathlib import Path

from .wrap_individual import wrap_artifact_as_individual
from .mutation import (
    MutatedOffspring,
    mutate_offspring,
)
from .recombination import (
    LineageRecord,
    RecombinedOffspring,
    ReproducedOffspring,
    recombine_offspring,
    breed_offspring,
    build_selection_strategy,
    build_crossover_strategy,
    build_mutation_strategy,
    build_replacement_strategy,
    build_pipeline,
)


def get_evolution_operations() -> list[dict]:
    """Returns the list of host-level evolutionary operations loaded from their JSON definitions."""
    dir_path = Path(__file__).parent
    operations: list[dict] = []
    for filename in ["wrap_individual.json", "mutate.json", "recombine.json"]:
        json_path = dir_path / filename
        if not json_path.exists():
            raise FileNotFoundError(f"Evolution operation config not found at: {json_path}")
        with open(json_path, "r", encoding="utf-8") as f:
            operations.append(json.load(f))
    return operations


__all__ = [
    "get_evolution_operations",
    "wrap_artifact_as_individual",
    "MutatedOffspring",
    "mutate_offspring",
    "LineageRecord",
    "RecombinedOffspring",
    "ReproducedOffspring",
    "recombine_offspring",
    "breed_offspring",
    "build_selection_strategy",
    "build_crossover_strategy",
    "build_mutation_strategy",
    "build_replacement_strategy",
    "build_pipeline",
]
