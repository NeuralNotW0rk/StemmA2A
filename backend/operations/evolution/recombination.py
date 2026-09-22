"""Agnostic evolutionary recombination / crossover operators and breeding coordinators."""

from dataclasses import dataclass
from typing import Optional, Union

from neutral_selection.representation.individual import Individual
from neutral_selection.representation.population import Population
from neutral_selection.variation.selection import SelectionStrategy
from neutral_selection.variation.recombination import RecombinationStrategy
from neutral_selection.variation.mutation import MutationStrategy
from neutral_selection.pipeline import step
from neutral_selection.builders import (
    build_selection_strategy,
    build_crossover_strategy,
    build_mutation_strategy,
    build_replacement_strategy,
    build_pipeline,
)


@dataclass
class LineageRecord:
    """Tracks the lineage and recombination history of an offspring individual."""
    parent_ids: list[str]
    crossover_applied: bool
    mutated: bool


@dataclass
class RecombinedOffspring:
    """Wraps a newly bred/recombined Individual alongside its lineage metadata."""
    individual: Individual
    lineage: LineageRecord


# Backward-compatibility alias
ReproducedOffspring = RecombinedOffspring


def recombine_offspring(
    parents: Union[Population, list[Individual]],
    offspring_count: int,
    selection_strategy: SelectionStrategy,
    crossover_strategy: Optional[RecombinationStrategy] = None,
    mutation_strategy: Optional[MutationStrategy] = None,
    crossover_prob: float = 0.8,
    elitism: int = 0,
    parent_ids: Optional[list[str]] = None,
) -> list[RecombinedOffspring]:
    """
    Generates `offspring_count` child Individuals from the given parent population
    by coordinating selection, crossover, mutation, and elitism via neutral_selection.

    This operation is representation-agnostic and works with any Individual genotype.

    Args:
        parents: Population or list of parent Individuals.
        offspring_count: The total number of offspring to produce.
        selection_strategy: The SelectionStrategy used to select mating pairs.
        crossover_strategy: Optional RecombinationStrategy to combine parent pairs.
        mutation_strategy: Optional MutationStrategy to mutate child genotypes.
        crossover_prob: Probability of applying crossover to a selected pair (0.0 to 1.0).
        elitism: Number of top-fitness parent individuals to carry over unchanged.
        parent_ids: Optional list of ID strings corresponding to `parents` in order.

    Returns:
        A list of RecombinedOffspring items containing the child Individuals and lineage records.
    """
    bred_population: Population = step(
        parents=parents,
        selection_strategy=selection_strategy,
        crossover_strategy=crossover_strategy,
        mutation_strategy=mutation_strategy,
        offspring_count=offspring_count,
        crossover_prob=crossover_prob,
        elitism=elitism,
        parent_ids=parent_ids,
    )

    offspring: list[RecombinedOffspring] = []
    for ind in bred_population:
        lineage_record = LineageRecord(
            parent_ids=list(ind.lineage.parent_ids) if ind.lineage else [],
            crossover_applied=ind.lineage.crossover_applied if ind.lineage else False,
            mutated=ind.lineage.mutated if ind.lineage else False,
        )
        offspring.append(RecombinedOffspring(individual=ind, lineage=lineage_record))

    return offspring


# Backward-compatibility alias
breed_offspring = recombine_offspring
