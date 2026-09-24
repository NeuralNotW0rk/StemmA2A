"""Agnostic evolutionary operations package."""

import json
from pathlib import Path

from .wrap_individual import (
    WrapIndividualOperation,
    wrap_artifact_as_individual,
    wrap_precursor_as_individual,
)
from .mutate import (
    MutateOperation,
    MutatedOffspring,
    mutate_offspring,
    mutate_evolution_task,
    run_mutate_task,
    dispatch_mutate_operation,
)
from .recombine import (
    RecombineOperation,
    LineageRecord,
    RecombinedOffspring,
    recombine_offspring,
    build_selection_strategy,
    build_crossover_strategy,
    build_mutation_strategy,
    build_replacement_strategy,
    build_pipeline,
    recombine_evolution_task,
    run_recombine_task,
    dispatch_recombine_operation,
)
from .expression import express_individual_to_grating_artifact
from .resolution import extract_individual_parent_ids


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
    "WrapIndividualOperation",
    "wrap_artifact_as_individual",
    "wrap_precursor_as_individual",
    "MutateOperation",
    "MutatedOffspring",
    "mutate_offspring",
    "mutate_evolution_task",
    "run_mutate_task",
    "dispatch_mutate_operation",
    "RecombineOperation",
    "LineageRecord",
    "RecombinedOffspring",
    "recombine_offspring",
    "build_selection_strategy",
    "build_crossover_strategy",
    "build_mutation_strategy",
    "build_replacement_strategy",
    "build_pipeline",
    "recombine_evolution_task",
    "run_recombine_task",
    "dispatch_recombine_operation",
    "express_individual_to_grating_artifact",
    "extract_individual_parent_ids",
]
