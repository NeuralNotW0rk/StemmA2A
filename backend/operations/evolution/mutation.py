"""Agnostic evolutionary mutation operators and mutation coordinators."""

from dataclasses import dataclass
from typing import Union, Optional

from neutral_selection.representation.individual import Individual
from neutral_selection.representation.population import Population
from neutral_selection.representation.lineage import Lineage
from neutral_selection.variation.mutation import MutationStrategy, mutate


@dataclass
class MutatedOffspring:
    """Wraps a newly mutated Individual alongside its parent lineage tracking."""
    individual: Individual
    parent_id: str
    mutated: bool


def mutate_offspring(
    parents: Union[Population, list[Individual]],
    mutation_strategy: MutationStrategy,
    offspring_per_parent: int = 1,
    parent_ids: Optional[list[str]] = None,
) -> list[MutatedOffspring]:
    """
    Mutates each parent individual in the given collection by `offspring_per_parent` times
    using the supplied MutationStrategy.

    This operation is representation-agnostic and works with any Individual genotype.

    Args:
        parents: Population or list of parent Individuals.
        mutation_strategy: The MutationStrategy to apply to each genotype.
        offspring_per_parent: Number of mutant variants to create per parent.
        parent_ids: Optional list of ID strings corresponding to parents.

    Returns:
        A list of MutatedOffspring containing the mutated Individuals and parent lineage.
    """
    parent_list: list[Individual] = list(parents)
    results: list[MutatedOffspring] = []

    for idx, parent in enumerate(parent_list):
        pid = parent_ids[idx] if (parent_ids and idx < len(parent_ids)) else (
            parent.metadata.get("id") or f"parent_{idx}"
        )

        for _ in range(offspring_per_parent):
            cloned_ind = parent.clone(deep=True)
            mutated_genotype = mutate(cloned_ind.genotype, mutation_strategy)
            cloned_ind.genotype = mutated_genotype
            cloned_ind.fitness = None
            cloned_ind.clear_expression()
            cloned_ind.lineage = Lineage(
                parent_ids=[pid],
                crossover_applied=False,
                mutated=True
            )
            results.append(MutatedOffspring(
                individual=cloned_ind,
                parent_id=pid,
                mutated=True
            ))

    return results
