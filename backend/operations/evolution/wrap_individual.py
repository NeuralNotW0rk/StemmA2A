"""Agnostic operation for wrapping precursor artifacts into evolutionary Individual nodes."""

from typing import Any, Optional
import os
import copy
import uuid
import threading
from coolname import generate_slug

from param_graph.elements.artifacts.individual_element import Individual
from param_graph.elements.artifacts.grating_element import Grating
from param_graph.elements.artifacts.audio_element import Audio
from param_graph.elements.models.base_model_element import Model
from param_graph.elements.base_elements import Asset
from diffracture.topology.grating import Grating as DiffractureGrating
from evolution.genome import LoRAGenome
from utils.uid import XXH3_64


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


async def wrap_precursor_as_individual(
    data: dict[str, Any],
    param_graph: Any,
    engine_provider: Any,
    graph_lock: Optional[threading.Lock] = None,
    uid_generator: Optional[Any] = None,
) -> tuple[dict[str, Any], int]:
    """
    Synchronously creates a baseline Individual node from a precursor artifact in ParameterGraph.

    Args:
        data: Request payload containing precursor identifier and optional parameters.
        param_graph: The active ParameterGraph instance.
        engine_provider: The active EngineProvider instance.
        graph_lock: Optional threading.Lock for graph synchronization.
        uid_generator: Optional UID generator (defaults to XXH3_64).

    Returns:
        A tuple of (response_dict, status_code).
    """
    if param_graph is None or engine_provider is None:
        return {"error": "No project loaded"}, 400

    params = data.get("params", {}) or {}
    uid_gen = uid_generator or XXH3_64()

    # 1. Resolve precursor node ID
    initiator = data.get("initiator") or params.get("initiator") or {}
    initiator_id = initiator.get("id") if isinstance(initiator, dict) else (str(initiator).strip() if initiator else None)

    precursor_val = (
        data.get("precursor_id")
        or data.get("precursor")
        or data.get("precursor_audio_id")
        or data.get("source_node")
        or data.get("source_node_id")
        or data.get("source_audio")
        or data.get("source_audio_id")
        or data.get("source_id")
        or data.get("artifact_id")
        or data.get("node_id")
        or params.get("precursor_id")
        or params.get("precursor")
        or params.get("precursor_audio_id")
        or params.get("source_node")
        or params.get("source_node_id")
        or params.get("source_audio")
        or params.get("source_audio_id")
        or params.get("source_id")
        or params.get("artifact_id")
        or initiator_id
    )
    if isinstance(precursor_val, dict):
        precursor_node_id = precursor_val.get("id")
    else:
        precursor_node_id = str(precursor_val).strip() if precursor_val else None

    if not precursor_node_id:
        return {"error": "Precursor artifact is required to wrap an Individual."}, 400

    def _get_precursor_and_model() -> tuple[Any, Any, str, Any, Any, int, float, Optional[dict[str, Any]], Optional[int]]:
        precursor_node = param_graph.get_element(precursor_node_id)
        if not precursor_node:
            return None, None, "", None, None, 0, 0.0, {"error": f"Precursor artifact '{precursor_node_id}' not found."}, 400

        # Resolve model_id
        model_val = data.get("model_id") or data.get("model") or params.get("model") or params.get("model_id")
        if isinstance(model_val, dict):
            model_id = model_val.get("id")
        else:
            model_id = str(model_val).strip() if model_val else None

        if not model_id:
            ctx = getattr(precursor_node, "context", {}) or {}
            model_id = ctx.get("model_id")

        if not model_id and hasattr(precursor_node, "base_model_id"):
            model_id = precursor_node.base_model_id

        if not model_id:
            incomers = param_graph.get_incomers(precursor_node.id)
            for inc in incomers:
                if isinstance(inc, Model):
                    model_id = inc.id
                    break

        if not model_id:
            node_name = getattr(precursor_node, "name", precursor_node_id)
            return None, None, "", None, None, 0, 0.0, {
                "error": f"Precursor node '{node_name}' has no associated generative model and cannot be wrapped as an Individual."
            }, 400

        model_element = param_graph.get_element(model_id)
        if not isinstance(model_element, Model):
            return None, None, "", None, None, 0, 0.0, {"error": f"Associated model '{model_id}' was not found in the project graph."}, 400

        baseline_grating_id = data.get("baseline_grating_id") or params.get("baseline_grating_id")
        elements_input = data.get("elements") or params.get("elements")
        lora_rank = int(data.get("lora_rank") or params.get("lora_rank") or 1)
        lora_alpha = float(data.get("lora_alpha") or params.get("lora_alpha") or 1.0)

        return precursor_node, model_element, model_id, baseline_grating_id, elements_input, lora_rank, lora_alpha, None, None

    if graph_lock is not None:
        with graph_lock:
            precursor_node, model_element, model_id, baseline_grating_id, elements_input, lora_rank, lora_alpha, err_dict, err_code = _get_precursor_and_model()
    else:
        precursor_node, model_element, model_id, baseline_grating_id, elements_input, lora_rank, lora_alpha, err_dict, err_code = _get_precursor_and_model()

    if err_dict is not None:
        return err_dict, err_code  # type: ignore

    # If grating node exists
    if baseline_grating_id:
        if graph_lock is not None:
            with graph_lock:
                baseline_grating = param_graph.get_element(baseline_grating_id)
                if not isinstance(baseline_grating, Grating):
                    return {"error": f"Node '{baseline_grating_id}' is not a valid grating."}, 400
                base_grating_path = baseline_grating.file.path
                baseline_elements = baseline_grating.elements
                baseline_name = baseline_grating.name
                diff_base_grating = DiffractureGrating.load(base_grating_path)
        else:
            baseline_grating = param_graph.get_element(baseline_grating_id)
            if not isinstance(baseline_grating, Grating):
                return {"error": f"Node '{baseline_grating_id}' is not a valid grating."}, 400
            base_grating_path = baseline_grating.file.path
            baseline_elements = baseline_grating.elements
            baseline_name = baseline_grating.name
            diff_base_grating = DiffractureGrating.load(base_grating_path)
    elif isinstance(precursor_node, Grating):
        base_grating_path = precursor_node.file.path
        baseline_elements = precursor_node.elements
        baseline_name = precursor_node.name
        diff_base_grating = DiffractureGrating.load(base_grating_path)
        baseline_grating_id = precursor_node.id
    else:
        # If not provided, inspect model linear layers
        if not elements_input:
            engine = engine_provider.get_engine()
            layers = await engine.get_model_layers(model_element)
            linear_layers = [l for l in layers if "Linear" in l.get("type", "")]
            if not linear_layers:
                return {"error": f"No compatible Linear layers found on model '{model_id}' for LoRA genome."}, 400

            elements_input = [
                {
                    "address": l["address"],
                    "kernel_type": "lora",
                    "params": {
                        "rank": lora_rank,
                        "alpha": lora_alpha,
                        "in_features": l.get("in_features", 0) or 0,
                        "out_features": l.get("out_features", 0) or 0,
                        "kernel_size": None
                    },
                    "indices": [],
                    "perform_clustering": False,
                    "num_clusters": None,
                    "cluster": None
                }
                for l in linear_layers
            ]

        engine = engine_provider.get_engine()
        grating_name = f"baseline_{uuid.uuid4().hex[:8]}"
        baseline_grating = await engine.create_grating(model_element, grating_name, elements_input)
        base_grating_path = baseline_grating.file.path
        baseline_elements = baseline_grating.elements
        baseline_name = precursor_node.name or precursor_node.alias or "Artifact"
        diff_base_grating = DiffractureGrating.load(base_grating_path)

    # Build baseline genome from base grating
    baseline_genome = LoRAGenome.from_grating(diff_base_grating)

    # Compute deterministic UID from baseline genome
    genome_state_dict = baseline_genome.get_state_dict()
    genome_uid = uid_gen.from_state_dict(genome_state_dict)
    individual_id = f"individual_{genome_uid}"

    output_dir = param_graph.root / "generate"
    os.makedirs(output_dir, exist_ok=True)
    genome_path = output_dir / f"{individual_id}.safetensors"
    baseline_genome.save(str(genome_path))

    # Prepare context
    ind_context = copy.deepcopy(getattr(precursor_node, "context", {}) or {})
    ind_context["model_id"] = model_id
    ind_context["baseline_elements"] = baseline_elements
    ind_context["baseline_file_path"] = str(base_grating_path)
    ind_context["precursor_artifact_id"] = precursor_node_id

    slug = generate_slug(2)
    custom_name = data.get("name") or params.get("name") or f"{baseline_name} - Baseline ({slug})"

    individual_node = wrap_artifact_as_individual(
        individual_id=individual_id,
        name=custom_name,
        asset_path=str(genome_path),
        base_model_id=model_id,
        baseline_grating_id=baseline_grating_id,
        generation=0,
        context=ind_context,
    )

    def _link_and_save() -> None:
        param_graph.add_element(individual_node)
        param_graph.link(model_element, individual_node, relation='binds_to')
        param_graph.link(precursor_node, individual_node, relation='precursor')

        # If the precursor is an audio artifact, set its parent to the individual so it serves as exemplar
        if isinstance(precursor_node, Audio):
            param_graph.update_element(precursor_node.id, {"parent": individual_id})

        param_graph.save()

    if graph_lock is not None:
        with graph_lock:
            _link_and_save()
    else:
        _link_and_save()

    return {
        "success": True,
        "message": f"Artifact '{precursor_node_id}' wrapped as Individual '{individual_id}'",
        "individual": individual_node.to_dict(),
        "node_id": individual_id
    }, 200


from ..base import Operation
from ..registry import register


@register
class WrapIndividualOperation(Operation):
    @property
    def name(self) -> str:
        return "wrap_individual"

    @property
    def description(self) -> str:
        return "Wraps a precursor artifact into a baseline Individual node with generation 0."

    @property
    def category(self) -> str:
        return "evolution"

    @property
    def execution(self) -> str:
        return "immediate"

    @property
    def initiator_types(self) -> list[str]:
        return ["audio", "image", "grating", "latent", "individual"]

    async def execute_async(self, **kwargs: Any) -> tuple[dict[str, Any], int]:
        return await wrap_precursor_as_individual(
            data=kwargs.get("data", kwargs),
            param_graph=kwargs.get("param_graph"),
            engine_provider=kwargs.get("engine_provider"),
            graph_lock=kwargs.get("graph_lock"),
            uid_generator=kwargs.get("uid_generator"),
        )

