import random
import copy
import json
from dataclasses import dataclass
from typing import Callable, Any, Union
from pathlib import Path

import torch
from diffracture.topology.grating import Grating
from neutral_selection.representation.genome import Genome


@dataclass
class PerturbationGene:
    """
    A topology-agnostic gene representing a single LoRA perturbation site.
    Stores lora_down and lora_up weights directly.
    """
    address: str
    lora_down: torch.Tensor
    lora_up: torch.Tensor
    active: bool

    def __post_init__(self) -> None:
        if not isinstance(self.address, str):
            raise TypeError("address must be a string")
        if not isinstance(self.lora_down, torch.Tensor):
            raise TypeError("lora_down must be a torch.Tensor")
        if not isinstance(self.lora_up, torch.Tensor):
            raise TypeError("lora_up must be a torch.Tensor")
        if not isinstance(self.active, bool):
            raise TypeError("active must be a boolean")

    @classmethod
    def from_tensors(
        cls,
        address: str,
        lora_down: torch.Tensor,
        lora_up: torch.Tensor,
        active: bool = True
    ) -> "PerturbationGene":
        """
        Creates a PerturbationGene by cloning the given weight tensors.
        """
        if not isinstance(lora_down, torch.Tensor) or not isinstance(lora_up, torch.Tensor):
            raise TypeError("lora_down and lora_up must be torch.Tensors")

        return cls(
            address=address,
            lora_down=lora_down.clone(),
            lora_up=lora_up.clone(),
            active=active
        )

    def to_tensors(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Reconstructs the lora_down and lora_up tensors.
        """
        return self.lora_down, self.lora_up


class LoRAGenome(Genome):
    """
    A 1D sequence representing LoRA/DoRA perturbations ordered from input to output.
    """
    def __init__(self, items: list[PerturbationGene]) -> None:
        super().__init__(items)

    def get_state_dict(self) -> dict[str, torch.Tensor]:
        """
        Returns a dict of tensors representing the genome's state (lora_down, lora_up, active flags).
        This can be used to compute a deterministic content-addressable UID.
        """
        state_dict: dict[str, torch.Tensor] = {}
        for gene in self._items:
            state_dict[f"{gene.address}.lora_down"] = gene.lora_down
            state_dict[f"{gene.address}.lora_up"] = gene.lora_up
            state_dict[f"{gene.address}.active"] = torch.tensor(
                gene.active, dtype=torch.bool, device=gene.lora_down.device
            )
        return state_dict

    def save(self, path: Union[str, Path]) -> None:
        """
        Saves the genome to a safetensors file.
        Only serializes the gene names, weights (lora_down, lora_up), and basic active state,
        avoiding duplicating the full topology (ranks, alpha, dimensions) which is stored in the base grating.
        """
        from safetensors.torch import save_file

        state_dict = {}
        gene_metadata = {}
        addresses = []

        for gene in self._items:
            addresses.append(gene.address)
            # Store directions in the state dict
            state_dict[f"{gene.address}.lora_down"] = gene.lora_down
            state_dict[f"{gene.address}.lora_up"] = gene.lora_up
            # Store scalars/states in the metadata
            gene_metadata[gene.address] = {
                "active": bool(gene.active)
            }

        metadata = {
            "addresses": json.dumps(addresses),
            "gene_metadata": json.dumps(gene_metadata)
        }

        save_file(state_dict, str(path), metadata=metadata)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "LoRAGenome":
        """
        Loads a genome from a safetensors file.
        """
        from safetensors import safe_open

        genes = []
        with safe_open(str(path), framework="pt", device="cpu") as f:
            metadata = f.metadata()
            if not metadata or "addresses" not in metadata or "gene_metadata" not in metadata:
                raise ValueError(
                    f"Incompatible or missing metadata in genome file: {path}. "
                    "Make sure it is a serialized LoRAGenome safetensors file."
                )

            addresses = json.loads(metadata["addresses"])
            gene_metadata = json.loads(metadata["gene_metadata"])

            for address in addresses:
                lora_down = f.get_tensor(f"{address}.lora_down")
                lora_up = f.get_tensor(f"{address}.lora_up")
                meta = gene_metadata[address]
                active = meta["active"]

                gene = PerturbationGene(
                    address=address,
                    lora_down=lora_down,
                    lora_up=lora_up,
                    active=active
                )
                genes.append(gene)

        return cls(genes)

    @classmethod
    def from_grating(cls, grating: Grating) -> "LoRAGenome":
        """
        Constructs a LoRAGenome from the LoRA/DoRA elements inside the given Grating.
        """
        genes = []
        # Use grating.nodes to get elements in their stored order
        for address, element in grating.nodes.items():
            if "lora_down" in element.params and "lora_up" in element.params:
                gene = PerturbationGene.from_tensors(
                    address=address,
                    lora_down=element.params["lora_down"].data,
                    lora_up=element.params["lora_up"].data,
                    active=element.active
                )
                genes.append(gene)
        return cls(genes)

    @classmethod
    def create_initial_population(
        cls,
        base_grating: Grating,
        population_size: int,
        lora_noise: float,
        active_flip_prob: float
    ) -> list["LoRAGenome"]:
        """
        Creates an initial population of LoRAGenomes from a base grating.
        All individuals in the population are mutated copies of the base genome.
        """
        from neutral_selection.variation.mutation import mutate, UniformMutation, attribute_mutator, bit_flip_mutator

        base_genome = cls.from_grating(base_grating)
        population: list[LoRAGenome] = []

        mutation_strategy = get_lora_mutation_strategy(
            lora_noise=lora_noise,
            active_flip_prob=active_flip_prob,
            mutation_rate=1.0
        )

        for _ in range(population_size):
            child_genome = copy.deepcopy(base_genome)
            child_genome = mutate(child_genome, mutation_strategy)
            
            if not isinstance(child_genome, cls):
                child_genome = cls(list(child_genome))
            population.append(child_genome)

        return population


def express_to_grating(genome: LoRAGenome, base_grating: Union[Grating, str, Path]) -> Grating:
    """
    Rebuilds a Grating, populating it with elements expressed from the genome.
    If base_grating is a string/Path, loads the Grating from that path first.
    """
    from diffracture.registry import get_element

    if isinstance(base_grating, (str, Path)):
        base_grating = Grating.load(str(base_grating))

    new_grating = Grating()
    gene_dict = {gene.address: gene for gene in genome}

    for address, element in base_grating.nodes.items():
        element_cls = get_element(element.kernel_type)
        new_el = element_cls.from_metadata(address, element.metadata)
        new_el.multiplier = element.multiplier

        if address in gene_dict:
            gene = gene_dict[address]
            new_el.active = gene.active
            if gene.active:
                lora_down, lora_up = gene.to_tensors()
                
                # Check shapes match to enforce Fail-Fast
                if lora_down.shape != new_el.params["lora_down"].shape:
                    raise ValueError(
                        f"Shape mismatch for '{address}' lora_down: "
                        f"genome shape {lora_down.shape} != target shape {new_el.params['lora_down'].shape}"
                    )
                if lora_up.shape != new_el.params["lora_up"].shape:
                    raise ValueError(
                        f"Shape mismatch for '{address}' lora_up: "
                        f"genome shape {lora_up.shape} != target shape {new_el.params['lora_up'].shape}"
                    )

                new_el.params["lora_down"].data.copy_(
                    lora_down.to(
                        device=new_el.params["lora_down"].device,
                        dtype=new_el.params["lora_down"].dtype
                    )
                )
                new_el.params["lora_up"].data.copy_(
                    lora_up.to(
                        device=new_el.params["lora_up"].device,
                        dtype=new_el.params["lora_up"].dtype
                    )
                )
            else:
                new_el.params["lora_down"].data.zero_()
                new_el.params["lora_up"].data.zero_()
        else:
            # Fallback copying standard values
            new_el.active = element.active
            new_el.params["lora_down"].data.copy_(element.params["lora_down"])
            new_el.params["lora_up"].data.copy_(element.params["lora_up"])

        new_grating.add_element(new_el)

    return new_grating


def lora_gaussian_noise_mutator(std: float, mean: float = 0.0) -> Callable[[Any], Any]:
    """
    Returns a mutator function that adds Gaussian noise to numeric values,
    with custom LoRA-specific scaling initialization for zero-initialized torch Tensors.
    """
    from neutral_selection.variation.mutation import gaussian_noise_mutator

    base_mutator = gaussian_noise_mutator(std, mean=mean)

    def mutate_fn(val: Any) -> Any:
        if hasattr(val, "device") and hasattr(val, "dtype") and hasattr(val, "clone"):
            # Check for torch Tensor
            if torch.all(val == 0.0):
                # Standard LoRA-style initialization: scale std by 1 / sqrt(rank)
                rank = val.size(0)
                init_std = 1.0 / (rank ** 0.5) if rank > 0 else 1.0
                return torch.randn_like(val) * init_std * std + mean
        return base_mutator(val)

    return mutate_fn


def get_lora_mutation_strategy(
    lora_noise: float = 0.05,
    active_flip_prob: float = 0.05,
    mutation_rate: float = 1.0
) -> Any:
    """
    Returns a UniformMutation strategy tailored for LoRAGenome instances.
    """
    from neutral_selection.variation.mutation import UniformMutation, attribute_mutator, bit_flip_mutator

    return UniformMutation(
        mutation_rate=mutation_rate,
        mutation_fn=attribute_mutator({
            "lora_down": lora_gaussian_noise_mutator(std=lora_noise),
            "lora_up": lora_gaussian_noise_mutator(std=lora_noise),
            "active": bit_flip_mutator(prob=active_flip_prob)
        })
    )


def get_lora_crossover_strategy(num_cut_points: int = 1) -> Any:
    """
    Returns a RandomNPointCrossover strategy for LoRAGenome instances.
    """
    from neutral_selection.variation.recombination import RandomNPointCrossover

    return RandomNPointCrossover(num_cut_points=num_cut_points)


# Register representation with global registry
try:
    from evolution.registry import register_representation
    register_representation(
        "lora",
        LoRAGenome,
        express_to_grating,
        get_lora_mutation_strategy,
        get_lora_crossover_strategy
    )
except ImportError:
    pass
