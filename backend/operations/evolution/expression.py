"""Agnostic helper for expressing evolutionary Individuals to Grating artifacts in ParameterGraph."""

from typing import Any, Optional
import os
import uuid
import threading

from param_graph.elements.artifacts.individual_element import Individual
from param_graph.elements.artifacts.grating_element import Grating
from param_graph.elements.base_elements import Asset
from evolution.genome import LoRAGenome, express_to_grating
from utils.uid import XXH3_64


def express_individual_to_grating_artifact(
    individual_id: str,
    param_graph: Any,
    graph_lock: Optional[threading.Lock] = None,
    uid_generator: Optional[Any] = None,
) -> tuple[dict[str, Any], int]:
    """
    Expresses the genotype of an Individual node to a full Grating node in the parameter graph,
    linking it as a child of the individual.

    Args:
        individual_id: ID of the Individual node to express.
        param_graph: The active ParameterGraph instance.
        graph_lock: Optional threading.Lock for graph concurrency.
        uid_generator: Optional UID generator (defaults to XXH3_64).

    Returns:
        A tuple of (response_dict, status_code).
    """
    if param_graph is None:
        return {"error": "No project loaded"}, 400

    if not individual_id:
        return {"error": "individual_id is required"}, 400

    uid_gen = uid_generator or XXH3_64()

    def _execute_expression() -> tuple[dict[str, Any], int]:
        individual_node = param_graph.get_element(individual_id)
        if not isinstance(individual_node, Individual):
            return {"error": f"Node '{individual_id}' is not a valid individual."}, 400

        baseline_grating_id = individual_node.baseline_grating_id
        baseline_elements = individual_node.context.get("baseline_elements")
        baseline_file_path = individual_node.context.get("baseline_file_path")

        if baseline_grating_id:
            baseline_grating_node = param_graph.get_element(baseline_grating_id)
            if not isinstance(baseline_grating_node, Grating):
                return {"error": f"Baseline grating '{baseline_grating_id}' not found."}, 400
            base_elements = baseline_grating_node.elements
            base_path = baseline_grating_node.file.path
        elif baseline_elements and baseline_file_path:
            base_elements = baseline_elements
            base_path = baseline_file_path
        else:
            return {"error": "Unable to resolve baseline grating configuration for this individual."}, 400

        # Load LoRAGenome from individual file
        genome_path = param_graph.get_path_from_id(individual_node.id) or individual_node.file.path
        genome = LoRAGenome.load(genome_path)

        # Express to new Diffracture Grating
        expressed_diff_grating = express_to_grating(genome, base_path)

        grating_id = f"grating_{uid_gen.from_string(str(uuid.uuid4()))}"
        output_dir = param_graph.root / "generate"
        os.makedirs(output_dir, exist_ok=True)
        expressed_grating_path = output_dir / f"{grating_id}.safetensors"
        expressed_diff_grating.save(str(expressed_grating_path))

        # Create Grating Artifact
        grating_artifact = Grating(
            id=grating_id,
            name=f"Expressed {individual_node.name}",
            file=Asset(path=str(expressed_grating_path), uid=grating_id, extension=".safetensors"),
            base_model_id=individual_node.base_model_id,
            elements=base_elements,
            context=individual_node.context or {}
        )

        param_graph.add_element(grating_artifact)
        param_graph.update_element(grating_artifact.id, {"parent": individual_id})

        model_element = param_graph.get_element(individual_node.base_model_id)
        if model_element:
            param_graph.link(model_element, grating_artifact, relation='binds_to')
        param_graph.link(individual_node, grating_artifact, relation='expressed_to')

        param_graph.save()

        return {
            "success": True,
            "message": "Individual expressed to grating successfully",
            "grating": grating_artifact.to_dict()
        }, 200

    if graph_lock is not None:
        with graph_lock:
            return _execute_expression()
    else:
        return _execute_expression()


async def express_individuals(
    data: dict[str, Any],
    param_graph: Any,
    graph_lock: Optional[threading.Lock] = None,
    uid_generator: Optional[Any] = None,
) -> tuple[dict[str, Any], int]:
    """
    Expresses one or more Individuals (or groups of Individuals) to Grating artifacts.

    Args:
        data: Request payload containing the individuals to express.
        param_graph: The active ParameterGraph instance.
        graph_lock: Optional threading.Lock for graph concurrency.
        uid_generator: Optional UID generator (defaults to XXH3_64).

    Returns:
        A tuple of (response_dict, status_code).
    """
    if param_graph is None:
        return {"error": "No project loaded"}, 400

    params = data.get("params", {}) or {}
    initiator = data.get("initiator") or params.get("initiator")
    individual_input = (
        data.get("individuals")
        or params.get("individuals")
        or data.get("individual_id")
        or data.get("parent_ids")
        or data.get("parent_group_id")
        or initiator
    )
    individual_ids = extract_individual_parent_ids(individual_input, param_graph, graph_lock)
    if not individual_ids:
        return {"error": "At least one Individual is required to express."}, 400

    gratings: list[dict[str, Any]] = []
    for individual_id in individual_ids:
        resp, code = express_individual_to_grating_artifact(
            individual_id=individual_id,
            param_graph=param_graph,
            graph_lock=graph_lock,
            uid_generator=uid_generator,
        )
        if code != 200:
            return resp, code
        gratings.append(resp["grating"])

    return {
        "success": True,
        "status": "completed",
        "message": f"Expressed {len(gratings)} individual(s) to gratings",
        "artifact": gratings[0],
        "artifacts": gratings,
        "node_id": gratings[0]["id"],
    }, 200


from ..base import Operation
from ..registry import register
from .resolution import extract_individual_parent_ids


@register
class ExpressOperation(Operation):
    @property
    def name(self) -> str:
        return "express"

    @property
    def description(self) -> str:
        return "Expresses an Individual's genotype into a Grating phenotype."

    @property
    def category(self) -> str:
        return "evolution"

    @property
    def execution(self) -> str:
        return "immediate"

    async def execute_async(self, **kwargs: Any) -> tuple[dict[str, Any], int]:
        return await express_individuals(
            data=kwargs.get("data", kwargs),
            param_graph=kwargs.get("param_graph"),
            graph_lock=kwargs.get("graph_lock"),
            uid_generator=kwargs.get("uid_generator"),
        )
