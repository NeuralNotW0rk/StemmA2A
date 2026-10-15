"""LoRA / Grating genome representation, mutation, crossover, and phenotypic expression."""

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Union, Optional, Any, Callable

import torch
from diffracture.topology.grating import Grating
from neutral_selection.representation.genome import Genome
from neutral_selection.variation.mutation import MutationStrategy


@dataclass
class LoRAGene:
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
    ) -> "LoRAGene":
        """
        Creates a LoRAGene by cloning the given weight tensors.
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


# Backward-compatibility alias
PerturbationGene = LoRAGene


# --- LoRA Mutation Operators ---

def lora_noise_reference(val: torch.Tensor, floor: Optional[float] = None) -> float:
    """
    Returns the scale a tensor's noise is measured against: its own RMS, floored at `floor`
    (by default the 1/sqrt(fan_in) init scale). The floor keeps zero-initialized factors able to
    move and sizes steps by layer width; the RMS keeps steps proportional once weights have grown.
    """
    if floor is None:
        fan_in = val.numel() // val.size(0) if val.dim() > 0 and val.size(0) > 0 else 1
        floor = 1.0 / max(fan_in, 1) ** 0.5
    rms = val.detach().float().pow(2).mean().sqrt().item() if val.numel() > 0 else 0.0
    return max(rms, floor)


def lora_scaled_noise_mutator(noise_scale: float, floor: Optional[float] = None) -> Callable[[torch.Tensor], torch.Tensor]:
    """
    Returns a mutator that adds Gaussian noise with std noise_scale * lora_noise_reference(tensor, floor),
    so noise_scale is a step size relative to each tensor rather than an absolute weight delta.
    """
    def mutate_fn(val: torch.Tensor) -> torch.Tensor:
        std = noise_scale * lora_noise_reference(val, floor)
        return val + (torch.randn(val.shape, device=val.device) * std).to(val.dtype)

    return mutate_fn


def lora_up_noise_floor(lora_down: torch.Tensor) -> Optional[float]:
    """
    Returns the lora_up noise floor at which its contribution to dW = up @ down has an RMS of about
    noise_scale / sqrt(in_features), the init scale of the base layer's weights. That contribution
    scales with ||down||_F, so a fixed fan-in floor would overshoot once lora_down has grown.
    Returns None (the fan-in default) when lora_down is zero.
    """
    down_norm = lora_down.detach().float().norm().item()
    return 1.0 / down_norm if down_norm > 0.0 else None


class LoRAMutation(MutationStrategy):
    """
    Mutation strategy for LoRAGenomes.
    Each active gene is perturbed with probability mutation_rate using relative Gaussian noise on both
    factors, with lora_up's floor set from lora_down (see lora_up_noise_floor). Separately, each gene's
    active flag flips with probability active_flips / len(genome), so the expected number of flips per
    offspring stays fixed regardless of model size.
    Inactive genes are frozen: a gene switched off keeps its weights, so switching it back on restores
    the state it was selected in rather than an unheard random drift.
    """
    def __init__(self, noise_scale: float, active_flips: float, mutation_rate: float) -> None:
        if noise_scale < 0.0:
            raise ValueError("noise_scale must be non-negative.")
        if active_flips < 0.0:
            raise ValueError("active_flips must be non-negative.")
        if not (0.0 <= mutation_rate <= 1.0):
            raise ValueError("mutation_rate must be between 0.0 and 1.0 inclusive.")
        self.noise_scale = float(noise_scale)
        self.active_flips = float(active_flips)
        self.mutation_rate = float(mutation_rate)

    def __call__(self, genome: Genome) -> Genome:
        if not isinstance(genome, Genome):
            raise TypeError("genome must be an instance of Genome.")

        genes = list(genome)
        flip_prob = min(1.0, self.active_flips / len(genes)) if genes else 0.0
        mutated: list[LoRAGene] = []
        for gene in genes:
            if not isinstance(gene, LoRAGene):
                raise TypeError(f"LoRAMutation expects LoRAGene items, got {type(gene).__name__}.")
            flip = random.random() < flip_prob
            # Only genes that are expressed both before and after this step are perturbed
            perturb = (
                gene.active and not flip
                and self.noise_scale > 0.0 and random.random() < self.mutation_rate
            )
            if not perturb and not flip:
                # Genes are never modified in place, so the parent's gene can be shared
                mutated.append(gene)
                continue
            mutated.append(LoRAGene(
                address=gene.address,
                lora_down=self._perturb(gene.lora_down) if perturb else gene.lora_down,
                lora_up=self._perturb(gene.lora_up, lora_up_noise_floor(gene.lora_down)) if perturb else gene.lora_up,
                active=(not gene.active) if flip else gene.active
            ))
        return genome.__class__(mutated)

    def _perturb(self, val: torch.Tensor, floor: Optional[float] = None) -> torch.Tensor:
        return lora_scaled_noise_mutator(self.noise_scale, floor)(val)


def get_lora_mutation_strategy(
    noise_scale: float = 0.1,
    active_flips: float = 1.0,
    mutation_rate: float = 1.0
) -> Any:
    """
    Returns a LoRAMutation strategy for LoRAGenome instances.
    """
    return LoRAMutation(noise_scale=noise_scale, active_flips=active_flips, mutation_rate=mutation_rate)


# --- LoRA Crossover / Recombination Operators ---

def lora_gene_blend_crossover_fn(gene_a: Any, gene_b: Any, blend_factor: float = 0.5) -> Any:
    """
    Blends two LoRAGene instances by interpolating their lora_down and lora_up weights.
    """
    if not isinstance(gene_a, LoRAGene) or not isinstance(gene_b, LoRAGene):
        return gene_a if random.random() < 0.5 else gene_b

    alpha = float(blend_factor)
    blended_down = (1.0 - alpha) * gene_a.lora_down + alpha * gene_b.lora_down
    blended_up = (1.0 - alpha) * gene_a.lora_up + alpha * gene_b.lora_up
    active = gene_a.active if random.random() < (1.0 - alpha) else gene_b.active

    return LoRAGene(
        address=gene_a.address,
        lora_down=blended_down,
        lora_up=blended_up,
        active=active
    )


def get_lora_crossover_strategy(
    strategy_type: str = "two_point",
    num_cut_points: int = 1,
    swap_prob: float = 0.5,
    blend_factor: float = 0.5,
    max_depth: Optional[int] = None,
) -> Any:
    """
    Returns a RecombinationStrategy tailored for LoRAGenome instances.
    Supports:
      - "two_point": Two-point multi-scale crossover across the flattened hierarchy (Default).
      - "one_point": One-point multi-scale crossover across the flattened hierarchy.
      - "random_n_point" / "n_point": Random N-point multi-scale crossover across the flattened hierarchy.
      - "layer_crossover": Wholesale layer splicing (constrained to max_depth=1).
      - "uniform_crossover" / "uniform": Independent per-element coin-flip swap.
      - "blend_crossover" / "blend" / "arithmetic": Continuous tensor weight blending.
    """
    from neutral_selection.variation.recombination import (
        OnePointCrossover,
        TwoPointCrossover,
        RandomNPointCrossover,
        UniformCrossover,
        ElementwiseCrossover,
    )

    s_type = str(strategy_type).lower().strip()
    if s_type in ("two_point", "2point", "2_point"):
        return TwoPointCrossover(max_depth=max_depth)
    elif s_type in ("one_point", "1point", "1_point"):
        return OnePointCrossover(max_depth=max_depth)
    elif s_type in ("layer_crossover", "wholesale_layer"):
        return RandomNPointCrossover(num_cut_points=num_cut_points, max_depth=1)
    elif s_type in ("blend_crossover", "blend", "arithmetic", "elementwise"):
        return ElementwiseCrossover(blend_factor=blend_factor, crossover_fn=lora_gene_blend_crossover_fn)
    elif s_type in ("uniform_crossover", "uniform"):
        return UniformCrossover(swap_prob=swap_prob, max_depth=max_depth)
    else:
        return RandomNPointCrossover(num_cut_points=num_cut_points, max_depth=max_depth)


# --- LoRA Genome Representation ---

class LoRAGenome(Genome):
    """
    A 1D sequence representing LoRA/DoRA perturbations ordered from input to output.
    """
    def __init__(self, items: list[LoRAGene]) -> None:
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

                gene = LoRAGene(
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
                gene = LoRAGene.from_tensors(
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
        noise_scale: float,
        active_flips: float
    ) -> list["LoRAGenome"]:
        """
        Creates an initial population of LoRAGenomes from a base grating.
        All individuals in the population are mutated copies of the base genome.
        """
        from neutral_selection.variation.mutation import mutate

        base_genome = cls.from_grating(base_grating)
        population: list[LoRAGenome] = []

        mutation_strategy = get_lora_mutation_strategy(
            noise_scale=noise_scale,
            active_flips=active_flips,
            mutation_rate=1.0
        )

        for _ in range(population_size):
            # Mutation builds a new genome and never modifies the base genome's tensors in place
            child_genome = mutate(base_genome, mutation_strategy)
            
            if not isinstance(child_genome, cls):
                child_genome = cls(list(child_genome))
            population.append(child_genome)

        return population


# --- Phenotypic Expression ---

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
