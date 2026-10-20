"""LoRA / Grating genome representation, mutation, crossover, and phenotypic expression."""

import dataclasses
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Union, Optional, Any

import torch
from diffracture.topology.grating import Grating
from neutral_selection.representation.genome import Genome
from neutral_selection.variation.mutation import MutationStrategy


# Element settings a gene carries beyond its tensors, with the values assumed when a file omits them
TOPOLOGY_DEFAULTS: dict[str, Any] = {
    "kernel_type": "lora",
    "alpha": 1.0,
    "param": "weight",
    "layout": None,
    "reference_scale": 1.0,
    "multiplier": 1.0,
}
# Element metadata recoverable from a gene's tensor shapes, so never stored
SHAPE_DERIVED_KEYS = ("rank", "in_features", "out_features", "kernel_size")


@dataclass
class LoRAGene:
    """
    A gene representing a single LoRA perturbation site, complete enough to express on its own.

    lora_down and lora_up hold genotype values. Expression multiplies each by a scale derived from its
    shape (see lora_gene_scales), so variation operators can add plain, uniform noise and still move
    wide and narrow layers comparably. topology holds the element's remaining settings (kernel type,
    alpha, ...); its rank and dimensions come from the tensor shapes. Variation never changes topology.

    activation encodes the on/off state as a one-element tensor, +1 for on and -1 for off, so it is an
    ordinary leaf that crossover inherits like the weights; active decodes it (on when positive). A bool
    is accepted in its place and encoded. Operators that blend values can leave it between the two;
    LoRAMutation snaps it back to +1 or -1.
    """
    address: str
    lora_down: torch.Tensor
    lora_up: torch.Tensor
    activation: Union[torch.Tensor, bool]
    topology: dict = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.address, str):
            raise TypeError("address must be a string")
        if not isinstance(self.lora_down, torch.Tensor):
            raise TypeError("lora_down must be a torch.Tensor")
        if not isinstance(self.lora_up, torch.Tensor):
            raise TypeError("lora_up must be a torch.Tensor")
        if isinstance(self.activation, bool):
            self.activation = encode_activation(self.activation, self.lora_down)
        if not isinstance(self.activation, torch.Tensor) or self.activation.numel() != 1:
            raise TypeError("activation must be a bool or a one-element torch.Tensor")
        if not isinstance(self.topology, dict):
            raise TypeError("topology must be a dict")
        self.topology = {**TOPOLOGY_DEFAULTS, **self.topology}

    @property
    def active(self) -> bool:
        """Whether the gene is expressed, decoded from its activation."""
        return bool(self.activation.item() > 0.0)

    def compact_topology(self) -> dict:
        """Returns the topology without its default values, as stored in files and hashed into the uid."""
        return {k: v for k, v in self.topology.items() if k not in TOPOLOGY_DEFAULTS or TOPOLOGY_DEFAULTS[k] != v}

    @classmethod
    def from_tensors(
        cls,
        address: str,
        lora_down: torch.Tensor,
        lora_up: torch.Tensor,
        active: bool = True,
        topology: Optional[dict] = None
    ) -> "LoRAGene":
        """
        Creates a LoRAGene by cloning the given genotype tensors.
        """
        if not isinstance(lora_down, torch.Tensor) or not isinstance(lora_up, torch.Tensor):
            raise TypeError("lora_down and lora_up must be torch.Tensors")

        return cls(
            address=address,
            lora_down=lora_down.clone(),
            lora_up=lora_up.clone(),
            activation=active,
            topology=dict(topology or {})
        )

    @classmethod
    def from_element(cls, element: Any) -> "LoRAGene":
        """
        Encodes a grating element's weights as a gene, keeping its topology.
        """
        lora_down = element.params["lora_down"].data
        lora_up = element.params["lora_up"].data
        down_scale, up_scale = lora_gene_scales(lora_down, lora_up)
        topology = {k: v for k, v in element.metadata.items() if k not in SHAPE_DERIVED_KEYS}
        return cls(
            address=element.address,
            lora_down=lora_down / down_scale,
            lora_up=lora_up / up_scale,
            activation=bool(element.active),
            topology={**topology, "kernel_type": element.kernel_type, "multiplier": element.multiplier}
        )

    def to_element(self) -> Any:
        """
        Expresses the gene as a grating element, rebuilding it from its topology and tensor shapes and
        decoding its genotype values into weights. An inactive gene's weights are zeroed.
        """
        from diffracture.registry import get_element

        rank, in_features, *kernel_size = self.lora_down.shape
        metadata = {k: v for k, v in self.topology.items() if k not in ("kernel_type", "multiplier")}
        metadata.update(
            rank=rank,
            in_features=in_features,
            out_features=self.lora_up.shape[0],
            kernel_size=tuple(kernel_size) or None,
        )
        element = get_element(self.topology["kernel_type"]).from_metadata(self.address, metadata)
        element.multiplier = self.topology["multiplier"]
        element.active = self.active

        if self.active:
            down_scale, up_scale = lora_gene_scales(self.lora_down, self.lora_up)
            for name, value in (("lora_down", self.lora_down * down_scale), ("lora_up", self.lora_up * up_scale)):
                target = element.params[name]
                target.data.copy_(value.to(device=target.device, dtype=target.dtype))
        else:
            element.params["lora_down"].data.zero_()
            element.params["lora_up"].data.zero_()
        return element

    def to_tensors(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns the genotype lora_down and lora_up tensors.
        """
        return self.lora_down, self.lora_up


# Backward-compatibility alias
PerturbationGene = LoRAGene


def encode_activation(active: bool, like: torch.Tensor) -> torch.Tensor:
    """Encodes an on/off state as a one-element activation tensor (+1 or -1) matching like's dtype and device."""
    return torch.full((1,), 1.0 if active else -1.0, dtype=like.dtype, device=like.device)


def lora_gene_scales(lora_down: torch.Tensor, lora_up: torch.Tensor) -> tuple[float, float]:
    """
    Returns the (down_scale, up_scale) that one genotype unit maps to in weight space. Only the tensors'
    shapes are read, so genotype or weight tensors of a gene give the same scales.
    down_scale is lora_down's 1/sqrt(fan_in) init scale. up_scale is sqrt(3/rank), the inverse of the
    expected ||lora_down||_F under its Kaiming-uniform init, so lora_up's contribution to dW = up @ down
    has an RMS of about 1/sqrt(in_features), the init scale of the base layer's weights.
    """
    def fan_in(t: torch.Tensor) -> int:
        return max(t.numel() // t.size(0), 1) if t.dim() > 0 and t.size(0) > 0 else 1

    # lora_up's fan-in is the rank
    return 1.0 / fan_in(lora_down) ** 0.5, (3.0 / fan_in(lora_up)) ** 0.5


# --- LoRA Mutation Operators ---

class LoRAMutation(MutationStrategy):
    """
    Mutation strategy for LoRAGenomes.
    Each gene is perturbed with probability mutation_rate by adding Gaussian noise with std noise_scale
    to both genotype factors; expression scales that into a width-appropriate step in weight space.
    Separately, each gene's on/off state flips with probability active_flips / len(genome), so the
    expected number of flips per offspring stays fixed regardless of model size, and every mutated
    gene's activation is snapped to +1 or -1.
    With freeze_inactive (the default), only genes that are on both before and after this step are
    perturbed: a gene switched off keeps its weights, so switching it back on restores the state it was
    selected in rather than an unheard random drift. Without it, switched-off genes drift too.
    """
    def __init__(self, noise_scale: float, active_flips: float, mutation_rate: float, freeze_inactive: bool = True) -> None:
        if noise_scale < 0.0:
            raise ValueError("noise_scale must be non-negative.")
        if active_flips < 0.0:
            raise ValueError("active_flips must be non-negative.")
        if not (0.0 <= mutation_rate <= 1.0):
            raise ValueError("mutation_rate must be between 0.0 and 1.0 inclusive.")
        self.noise_scale = float(noise_scale)
        self.active_flips = float(active_flips)
        self.mutation_rate = float(mutation_rate)
        self.freeze_inactive = bool(freeze_inactive)

    def __call__(self, genome: Genome) -> Genome:
        if not isinstance(genome, Genome):
            raise TypeError("genome must be an instance of Genome.")

        genes = list(genome)
        flip_prob = min(1.0, self.active_flips / len(genes)) if genes else 0.0
        mutated: list[LoRAGene] = []
        for gene in genes:
            if not isinstance(gene, LoRAGene):
                raise TypeError(f"LoRAMutation expects LoRAGene items, got {type(gene).__name__}.")
            was_active = gene.active
            flip = random.random() < flip_prob
            now_active = was_active != flip
            perturb = (
                (not self.freeze_inactive or (was_active and now_active))
                and self.noise_scale > 0.0 and random.random() < self.mutation_rate
            )
            # An activation left between +1 and -1 by a blending crossover is snapped to its decoded state
            snapped = abs(gene.activation.item()) == 1.0
            if not perturb and not flip and snapped:
                # Genes are never modified in place, so the parent's gene can be shared
                mutated.append(gene)
                continue
            mutated.append(dataclasses.replace(
                gene,
                lora_down=self._perturb(gene.lora_down) if perturb else gene.lora_down,
                lora_up=self._perturb(gene.lora_up) if perturb else gene.lora_up,
                activation=encode_activation(now_active, gene.activation)
            ))
        return genome.__class__(mutated)

    def _perturb(self, val: torch.Tensor) -> torch.Tensor:
        return val + (torch.randn(val.shape, device=val.device) * self.noise_scale).to(val.dtype)


def get_lora_mutation_strategy(
    noise_scale: float = 0.1,
    active_flips: float = 1.0,
    mutation_rate: float = 1.0,
    freeze_inactive: bool = True
) -> Any:
    """
    Returns a LoRAMutation strategy for LoRAGenome instances.
    """
    return LoRAMutation(
        noise_scale=noise_scale,
        active_flips=active_flips,
        mutation_rate=mutation_rate,
        freeze_inactive=freeze_inactive,
    )


# --- LoRA Crossover / Recombination Operators ---

def lora_gene_blend_crossover_fn(gene_a: Any, gene_b: Any, blend_factor: float = 0.5) -> Any:
    """
    Blends two LoRAGene instances by interpolating their genotype lora_down and lora_up values.
    The child keeps gene_a's address.
    """
    if not isinstance(gene_a, LoRAGene) or not isinstance(gene_b, LoRAGene):
        return gene_a if random.random() < 0.5 else gene_b

    alpha = float(blend_factor)
    blended_down = (1.0 - alpha) * gene_a.lora_down + alpha * gene_b.lora_down
    blended_up = (1.0 - alpha) * gene_a.lora_up + alpha * gene_b.lora_up
    activation = gene_a.activation if random.random() < (1.0 - alpha) else gene_b.activation

    return dataclasses.replace(gene_a, lora_down=blended_down, lora_up=blended_up, activation=activation)


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
      - "layer_crossover": Wholesale layer splicing (gene granularity: each LoRAGene is one leaf).
      - "uniform_crossover" / "uniform": Independent per-element coin-flip swap.
      - "blend_crossover" / "blend" / "arithmetic": Continuous tensor weight blending, gene by gene.
    """
    from neutral_selection.variation.recombination import (
        OnePointCrossover,
        TwoPointCrossover,
        RandomNPointCrossover,
        UniformCrossover,
        FunctionCrossover,
    )

    s_type = str(strategy_type).lower().strip()
    if s_type in ("two_point", "2point", "2_point"):
        return TwoPointCrossover(max_depth=max_depth)
    elif s_type in ("one_point", "1point", "1_point"):
        return OnePointCrossover(max_depth=max_depth)
    elif s_type in ("layer_crossover", "wholesale_layer"):
        return RandomNPointCrossover(num_cut_points=num_cut_points, granularity="gene")
    elif s_type in ("blend_crossover", "blend", "arithmetic", "elementwise"):
        return FunctionCrossover(lora_gene_blend_crossover_fn, blend_factor=blend_factor, granularity="gene", num_children=1)
    elif s_type in ("uniform_crossover", "uniform"):
        return UniformCrossover(swap_prob=swap_prob, max_depth=max_depth)
    else:
        return RandomNPointCrossover(num_cut_points=num_cut_points, max_depth=max_depth)


# --- LoRA Genome Representation ---

# Version 2 stores genotype values (decoded with scales from their shapes), activations and each
# gene's topology; unversioned files store phenotype weights and boolean active flags.
GENOME_FORMAT = "2"

class LoRAGenome(Genome):
    """
    A 1D sequence representing LoRA/DoRA perturbations ordered from input to output.
    """
    def __init__(self, items: list[LoRAGene]) -> None:
        super().__init__(items)

    def get_state_dict(self) -> dict[str, torch.Tensor]:
        """
        Returns a dict representing the genome's state (genotype values, activations and topology).
        This can be used to compute a deterministic content-addressable UID.
        """
        state_dict: dict[str, torch.Tensor] = {}
        for gene in self._items:
            state_dict[f"{gene.address}.lora_down"] = gene.lora_down
            state_dict[f"{gene.address}.lora_up"] = gene.lora_up
            state_dict[f"{gene.address}.activation"] = gene.activation
            state_dict[f"{gene.address}.topology"] = json.dumps(gene.compact_topology(), sort_keys=True)
        return state_dict

    def save(self, path: Union[str, Path]) -> None:
        """
        Saves the genome to a safetensors file.
        Serializes the gene names, genotype values (lora_down, lora_up, activation) and topology, so the
        file can be expressed on its own. Topology is stored without its defaults.
        """
        from safetensors.torch import save_file

        state_dict = {}
        gene_metadata = {}
        addresses = []

        for gene in self._items:
            addresses.append(gene.address)
            state_dict[f"{gene.address}.lora_down"] = gene.lora_down
            state_dict[f"{gene.address}.lora_up"] = gene.lora_up
            state_dict[f"{gene.address}.activation"] = gene.activation
            gene_metadata[gene.address] = gene.compact_topology()

        metadata = {
            "format": GENOME_FORMAT,
            "addresses": json.dumps(addresses),
            "gene_metadata": json.dumps(gene_metadata)
        }

        save_file(state_dict, str(path), metadata=metadata)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "LoRAGenome":
        """
        Loads a genome from a safetensors file.
        Files without a format version store phenotype weights and must first be converted against
        their baseline grating (utils.migrations.migrate_genome_encoding, run when a project loads).
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

            if metadata.get("format") != GENOME_FORMAT:
                raise ValueError(
                    f"Genome file {path} uses format {metadata.get('format') or 'legacy (unversioned)'}, "
                    f"expected {GENOME_FORMAT}. Reopen its project to migrate it."
                )

            addresses = json.loads(metadata["addresses"])
            gene_metadata = json.loads(metadata["gene_metadata"])

            for address in addresses:
                gene = LoRAGene(
                    address=address,
                    lora_down=f.get_tensor(f"{address}.lora_down"),
                    lora_up=f.get_tensor(f"{address}.lora_up"),
                    activation=f.get_tensor(f"{address}.activation"),
                    topology=gene_metadata[address]
                )
                genes.append(gene)

        return cls(genes)

    @classmethod
    def from_grating(cls, grating: Grating) -> "LoRAGenome":
        """
        Constructs a LoRAGenome from the LoRA/DoRA elements inside the given Grating, encoding their
        weights as genotype values and keeping their topology.
        """
        # Use grating.nodes to get elements in their stored order
        return cls([
            LoRAGene.from_element(element)
            for element in grating.nodes.values()
            if "lora_down" in element.params and "lora_up" in element.params
        ])

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

def express_to_grating(genome: LoRAGenome, base_grating: Union[Grating, str, Path, None] = None) -> Grating:
    """
    Builds a Grating from the genome, which carries every gene's topology (see LoRAGene.to_element).
    If base_grating (or a path to one) is given, its elements that are not genes are copied over
    unchanged, and genes are checked to match the shapes of its elements at the same addresses.
    """
    from diffracture.registry import get_element

    if isinstance(base_grating, (str, Path)):
        base_grating = Grating.load(str(base_grating))
    base_nodes = base_grating.nodes if base_grating is not None else {}

    new_grating = Grating()
    gene_addresses = set()
    for gene in genome:
        gene_addresses.add(gene.address)
        base_el = base_nodes.get(gene.address)
        if base_el is not None:
            # Fail fast when the genome was bred against a different topology
            for name, value in (("lora_down", gene.lora_down), ("lora_up", gene.lora_up)):
                if value.shape != base_el.params[name].shape:
                    raise ValueError(
                        f"Shape mismatch for '{gene.address}' {name}: "
                        f"genome shape {value.shape} != target shape {base_el.params[name].shape}"
                    )
        new_grating.add_element(gene.to_element())

    for address, element in base_nodes.items():
        if address in gene_addresses:
            continue
        new_el = get_element(element.kernel_type).from_metadata(address, element.metadata)
        new_el.multiplier = element.multiplier
        new_el.active = element.active
        for name, value in element.params.items():
            new_el.params[name].data.copy_(value)
        new_grating.add_element(new_el)

    return new_grating
