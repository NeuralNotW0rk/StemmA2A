from typing import Type, Callable, Any, Optional
from neutral_selection.representation.genome import Genome
from neutral_selection.variation.mutation import MutationStrategy
from neutral_selection.variation.recombination import RecombinationStrategy


_GENOME_REGISTRY: dict[str, Type[Genome]] = {}
_EXPRESSION_REGISTRY: dict[str, Callable[[Any, Any], Any]] = {}
_DEFAULT_MUTATION_FACTORIES: dict[str, Callable[..., MutationStrategy]] = {}
_DEFAULT_CROSSOVER_FACTORIES: dict[str, Callable[..., RecombinationStrategy]] = {}


def register_representation(
    name: str,
    genome_cls: Type[Genome],
    expression_fn: Callable[[Any, Any], Any],
    default_mutation_factory: Optional[Callable[..., MutationStrategy]] = None,
    default_crossover_factory: Optional[Callable[..., RecombinationStrategy]] = None,
) -> None:
    """Registers an evolutionary genome representation and its decoding/operator helpers."""
    key = name.lower()
    _GENOME_REGISTRY[key] = genome_cls
    _EXPRESSION_REGISTRY[key] = expression_fn
    if default_mutation_factory is not None:
        _DEFAULT_MUTATION_FACTORIES[key] = default_mutation_factory
    if default_crossover_factory is not None:
        _DEFAULT_CROSSOVER_FACTORIES[key] = default_crossover_factory


def get_genome_class(name: str) -> Type[Genome]:
    """Retrieves the Genome subclass registered for the given representation name."""
    key = name.lower()
    if key not in _GENOME_REGISTRY:
        raise ValueError(f"Unknown genome representation '{name}'. Available: {list(_GENOME_REGISTRY.keys())}")
    return _GENOME_REGISTRY[key]


def get_expression_function(name: str) -> Callable[[Any, Any], Any]:
    """Retrieves the expression/decoder function registered for the given representation name."""
    key = name.lower()
    if key not in _EXPRESSION_REGISTRY:
        raise ValueError(f"Unknown expression representation '{name}'. Available: {list(_EXPRESSION_REGISTRY.keys())}")
    return _EXPRESSION_REGISTRY[key]


def get_default_mutation_factory(name: str) -> Optional[Callable[..., MutationStrategy]]:
    """Retrieves the default mutation strategy factory for the given representation name."""
    return _DEFAULT_MUTATION_FACTORIES.get(name.lower())


def get_default_crossover_factory(name: str) -> Optional[Callable[..., RecombinationStrategy]]:
    """Retrieves the default crossover strategy factory for the given representation name."""
    return _DEFAULT_CROSSOVER_FACTORIES.get(name.lower())
