import random
import copy
import json
from dataclasses import dataclass
from typing import Callable, Any, Union
from pathlib import Path

import torch
from diffracture.topology.grating import Grating
from neutral_selection.representation.genome import Genome
from neutral_selection.variation.recombination import RandomNPointCrossover


@dataclass
class PerturbationGene:
    """
    A topology-agnostic gene representing a single LoRA perturbation site.
    Decoupled into normalized directions and a scalar magnitude.
    """
    address: str
    a_dir: torch.Tensor
    b_dir: torch.Tensor
    magnitude: float
    active: bool

    def __post_init__(self) -> None:
        if not isinstance(self.address, str):
            raise TypeError("address must be a string")
        if not isinstance(self.a_dir, torch.Tensor):
            raise TypeError("a_dir must be a torch.Tensor")
        if not isinstance(self.b_dir, torch.Tensor):
            raise TypeError("b_dir must be a torch.Tensor")
        if not isinstance(self.magnitude, (int, float)):
            raise TypeError("magnitude must be a float or int")
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
        Creates a PerturbationGene by normalizing the given weight tensors.
        """
        if not isinstance(lora_down, torch.Tensor) or not isinstance(lora_up, torch.Tensor):
            raise TypeError("lora_down and lora_up must be torch.Tensors")

        norm_a = float(torch.linalg.norm(lora_down).item())
        norm_b = float(torch.linalg.norm(lora_up).item())
        magnitude = norm_a * norm_b

        # Normalize a_dir
        if norm_a > 1e-9:
            a_dir = lora_down.clone() / norm_a
        else:
            a_dir = torch.zeros_like(lora_down)
            if a_dir.numel() > 0:
                a_dir = torch.randn_like(lora_down)
                norm_a_rand = torch.linalg.norm(a_dir)
                if norm_a_rand > 1e-9:
                    a_dir = a_dir / norm_a_rand

        # Normalize b_dir
        if norm_b > 1e-9:
            b_dir = lora_up.clone() / norm_b
        else:
            b_dir = torch.zeros_like(lora_up)
            if b_dir.numel() > 0:
                b_dir = torch.randn_like(lora_up)
                norm_b_rand = torch.linalg.norm(b_dir)
                if norm_b_rand > 1e-9:
                    b_dir = b_dir / norm_b_rand

        return cls(
            address=address,
            a_dir=a_dir,
            b_dir=b_dir,
            magnitude=magnitude,
            active=active
        )

    def to_tensors(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Reconstructs the scaled lora_down and lora_up tensors.
        """
        factor = torch.sqrt(
            torch.tensor(
                max(0.0, self.magnitude),
                dtype=self.a_dir.dtype,
                device=self.a_dir.device
            )
        )
        return factor * self.a_dir, factor * self.b_dir


class LoRAGenome(Genome):
    """
    A 1D sequence representing LoRA/DoRA perturbations ordered from input to output.
    """
    def __init__(self, items: list[PerturbationGene]) -> None:
        super().__init__(items)

    def save(self, path: Union[str, Path]) -> None:
        """
        Saves the genome to a safetensors file.
        Only serializes the gene names, weights (a_dir, b_dir), and basic scalar states,
        avoiding duplicating the full topology (ranks, alpha, dimensions) which is stored in the base grating.
        """
        from safetensors.torch import save_file

        state_dict = {}
        gene_metadata = {}
        addresses = []

        for gene in self._items:
            addresses.append(gene.address)
            # Store directions in the state dict
            state_dict[f"{gene.address}.a_dir"] = gene.a_dir
            state_dict[f"{gene.address}.b_dir"] = gene.b_dir
            # Store scalars/states in the metadata
            gene_metadata[gene.address] = {
                "magnitude": float(gene.magnitude),
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
                a_dir = f.get_tensor(f"{address}.a_dir")
                b_dir = f.get_tensor(f"{address}.b_dir")
                meta = gene_metadata[address]
                magnitude = meta["magnitude"]
                active = meta["active"]

                gene = PerturbationGene(
                    address=address,
                    a_dir=a_dir,
                    b_dir=b_dir,
                    magnitude=magnitude,
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



def mutate_perturbation_gene(
    gene: PerturbationGene,
    direction_noise: float = 0.05,
    magnitude_noise: float = 0.1,
    active_flip_prob: float = 0.05
) -> PerturbationGene:
    """
    Creates a mutated copy of the given gene.
    """
    new_gene = copy.deepcopy(gene)

    if new_gene.active:
        # Mutate direction vector a_dir
        if direction_noise > 0.0:
            noise_a = torch.randn_like(new_gene.a_dir) * direction_noise
            new_a = new_gene.a_dir + noise_a
            norm_a = torch.linalg.norm(new_a)
            if norm_a > 1e-9:
                new_gene.a_dir = new_a / norm_a

        # Mutate direction vector b_dir
        if direction_noise > 0.0:
            noise_b = torch.randn_like(new_gene.b_dir) * direction_noise
            new_b = new_gene.b_dir + noise_b
            norm_b = torch.linalg.norm(new_b)
            if norm_b > 1e-9:
                new_gene.b_dir = new_b / norm_b

        # Mutate magnitude
        if magnitude_noise > 0.0:
            noise_mag = float(torch.randn(1).item()) * magnitude_noise
            new_gene.magnitude = max(0.0, new_gene.magnitude + noise_mag)

    # Mutate active status
    if random.random() < active_flip_prob:
        new_gene.active = not new_gene.active

    return new_gene


def genome_from_grating(grating: Grating) -> LoRAGenome:
    """
    Deprecated alias. Use LoRAGenome.from_grating(grating) instead.
    """
    return LoRAGenome.from_grating(grating)


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
