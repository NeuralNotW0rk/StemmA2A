"""Scatter operation: spreads variant offspring around seed individuals, plus background task execution."""

from typing import Any, Optional, Callable, Union
from dataclasses import dataclass
import os
import copy
import contextvars
import uuid
import threading
import traceback
import asyncio
from coolname import generate_slug

from neutral_selection.representation.individual import Individual as NSIndividual
from neutral_selection.representation.population import Population
from neutral_selection.representation.lineage import Lineage
from neutral_selection.variation.mutation import MutationStrategy, mutate

from param_graph.elements.artifacts.individual_element import Individual
from param_graph.elements.artifacts.grating_element import Grating
from param_graph.elements.models.base_model_element import Model
from param_graph.elements.collections.group_element import Group
from param_graph.elements.base_elements import Asset
from evolution.lora_genome import LoRAGenome, get_lora_mutation_strategy
from utils.uid import XXH3_64

from ..base import Operation
from ..registry import register
from .resolution import extract_individual_parent_ids, resolve_exemplar_context
from .wrap_individual import wrap_precursor_as_individual



@dataclass
class ScatteredOffspring:
    """Wraps a newly scattered Individual alongside its parent lineage tracking."""
    individual: NSIndividual
    parent_id: str
    mutated: bool


def scatter_offspring(
    parents: Union[Population, list[NSIndividual]],
    mutation_strategy: MutationStrategy,
    offspring_per_parent: int = 1,
    parent_ids: Optional[list[str]] = None,
) -> list[ScatteredOffspring]:
    """
    Scatters `offspring_per_parent` variants around each parent individual in the given collection,
    perturbing each with the supplied MutationStrategy.
    """
    parent_list: list[NSIndividual] = list(parents)
    results: list[ScatteredOffspring] = []

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
            results.append(ScatteredOffspring(
                individual=cloned_ind,
                parent_id=pid,
                mutated=True
            ))

    return results


def scatter_evolution_task(
    parent_job_id: str,
    parent_ids: list[str],
    offspring_size: int,
    noise_scale: float,
    active_flips: float,
    mutation_rate: float = 1.0,
    freeze_inactive: bool = True,
    generation_context: Optional[dict[str, Any]] = None,
    generate_exemplars: bool = False,
    param_graph: Any = None,
    graph_lock: Optional[threading.Lock] = None,
    engine_provider: Optional[Any] = None,
    local_jobs: Optional[dict[str, Any]] = None,
    active_jobs: Optional[dict[str, Any]] = None,
    uid_generator: Optional[Any] = None,
) -> None:
    """
    Background worker that scatters variant offspring around parent individuals,
    creates child nodes in the parameter graph with lineage relationships,
    and groups offspring in a scatter Group.
    """
    uid_gen = uid_generator or XXH3_64()

    try:
        if local_jobs is not None and parent_job_id in local_jobs:
            local_jobs[parent_job_id]["status"] = "running"
            local_jobs[parent_job_id]["progress"] = {
                "value": 0,
                "total": offspring_size,
                "description": "Starting scatter..."
            }

        def _resolve_parents_and_group() -> tuple[list[Individual], Any, Any, Any, Any, int, dict[str, Any], str, dict[str, Any], dict[str, Any], Group]:
            parent_nodes: list[Individual] = []
            for pid in parent_ids:
                if param_graph.G.has_node(pid):
                    node = param_graph.get_element(pid)
                    if isinstance(node, Individual) and node not in parent_nodes:
                        parent_nodes.append(node)

            if not parent_nodes:
                raise ValueError("No valid parent individuals found in graph.")

            first_parent = parent_nodes[0]
            model_id = first_parent.base_model_id
            baseline_grating_id = first_parent.baseline_grating_id
            baseline_elements = first_parent.context.get("baseline_elements")
            baseline_file_path = first_parent.context.get("baseline_file_path")

            model_element = param_graph.get_element(model_id)
            if not model_element:
                raise ValueError(f"Base model '{model_id}' not found in parameter graph.")

            # Load baseline Diffracture Grating if needed
            base_grating = None
            if baseline_grating_id:
                base_grating = param_graph.get_element(baseline_grating_id)
            elif baseline_file_path:
                base_grating = Grating(
                    id=f"grating_{uuid.uuid4().hex[:8]}",
                    name="Baseline Grating",
                    context={},
                    file=Asset(path=str(baseline_file_path), uid=f"grating_{uuid.uuid4().hex[:8]}", extension=".safetensors"),
                    base_model_id=model_id,
                    elements=baseline_elements or []
                )

            max_gen = max(getattr(p, "generation", 0) for p in parent_nodes)
            next_generation = max_gen + 1

            parent_base_ctx = copy.deepcopy(getattr(parent_nodes[0], "context", {}) or {})
            merged_context = {**parent_base_ctx, **generation_context}
            merged_context["model_id"] = model_id
            merged_context["baseline_grating_id"] = baseline_grating_id
            merged_context["generation"] = next_generation

            operation = "generate"
            node_engine_args: dict[str, Any] = {}
            dumped_params: dict[str, Any] = {}

            if generate_exemplars and engine_provider is not None:
                engine = engine_provider.get_engine()
                # Run async adapter config inspection
                form_config = asyncio.run(engine.get_adapter_config(model_element.adapter))
                form_config = form_config.get(operation, []) if isinstance(form_config, dict) else form_config

                # Resolve parent exemplar context
                parent_exemplar_context = resolve_exemplar_context(
                    first_parent,
                    param_graph=param_graph,
                    graph_lock=None
                )
                effective_generation_params = {**parent_exemplar_context, **generation_context}


                from utils.form import create_dynamic_model
                DynamicArgsModel = create_dynamic_model(form_config)
                validated_params = DynamicArgsModel.model_validate(effective_generation_params)
                dumped_params = validated_params.model_dump()


                for field in form_config:
                    if field.get("type") == "node":
                        field_name = field.get("name")
                        node_id = (
                            dumped_params.pop(field_name, None)
                            or effective_generation_params.get(f"{field_name}_id")
                            or effective_generation_params.get(field_name)
                        )
                        if node_id:
                            element = param_graph.get_element(node_id)
                            if element:
                                node_engine_args[f"{field_name}_element"] = element

            # Create visual container Group
            mut_group_id = f"group_{uid_gen.from_string(str(uuid.uuid4()))}"
            scatter_group = Group(
                id=mut_group_id,
                member_ids=[],
                member_type='individual'
            )
            param_graph.add_element(scatter_group)
            param_graph.update_element(scatter_group.id, {"alias": f"Scatter (Gen {next_generation})"})
            param_graph.save()

            return (
                parent_nodes, model_element, model_id, baseline_grating_id,
                base_grating, next_generation, merged_context, operation,
                node_engine_args, dumped_params, scatter_group
            )

        if graph_lock is not None:
            with graph_lock:
                (
                    parent_nodes, model_element, model_id, baseline_grating_id,
                    base_grating, next_generation, merged_context, operation,
                    node_engine_args, dumped_params, scatter_group
                ) = _resolve_parents_and_group()
        else:
            (
                parent_nodes, model_element, model_id, baseline_grating_id,
                base_grating, next_generation, merged_context, operation,
                node_engine_args, dumped_params, scatter_group
            ) = _resolve_parents_and_group()

        group_id = scatter_group.id

        mutation_strategy = get_lora_mutation_strategy(
            noise_scale=noise_scale,
            active_flips=active_flips,
            mutation_rate=mutation_rate,
            freeze_inactive=freeze_inactive
        )

        job_ids: list[str] = []
        individual_ids: list[str] = []
        parent_genomes: dict[str, LoRAGenome] = {}

        for i in range(offspring_size):
            if local_jobs is not None and parent_job_id in local_jobs:
                local_jobs[parent_job_id]["progress"] = {
                    "value": i,
                    "total": offspring_size,
                    "description": f"Scattering individual {i+1} of {offspring_size}..."
                }

            # Select parent (round-robin across supplied parent individuals)
            selected_parent = parent_nodes[i % len(parent_nodes)]
            parent_genome = parent_genomes.get(selected_parent.id)
            if parent_genome is None:
                genome_path = param_graph.get_path_from_id(selected_parent.id) or selected_parent.file.path
                parent_genome = parent_genomes[selected_parent.id] = LoRAGenome.load(genome_path)

            # Mutation builds a new genome and never modifies the parent's tensors in place
            child_genome = mutate(parent_genome, mutation_strategy)
            if not isinstance(child_genome, LoRAGenome):
                child_genome = LoRAGenome(list(child_genome))

            child_ind_id = uid_gen.from_state_dict(child_genome.get_state_dict())

            if child_ind_id in individual_ids:
                print(
                    f"[scatter_evolution_task] Discarding duplicate scattered individual '{child_ind_id}' "
                    f"already generated in current batch."
                )
                continue

            output_dir = param_graph.root / "generate"
            os.makedirs(output_dir, exist_ok=True)
            saved_genome_path = output_dir / f"{child_ind_id}.safetensors"

            child_context = copy.deepcopy(merged_context)
            child_context["scatter_operation"] = {
                "noise_scale": noise_scale,
                "active_flips": active_flips,
                "mutation_rate": mutation_rate,
                "freeze_inactive": freeze_inactive,
                "parent_id": selected_parent.id,
                "generation": next_generation
            }
            child_context["lineage"] = {
                "parent_ids": [selected_parent.id],
                "crossover_applied": False,
                "mutated": True
            }

            slug = generate_slug(2)
            child_node = Individual(
                id=child_ind_id,
                name=f"Gen {next_generation} - Scatter {i+1} ({slug})",
                file=Asset(path=str(saved_genome_path), uid=child_ind_id, extension=".safetensors"),
                base_model_id=model_id,
                baseline_grating_id=baseline_grating_id,
                generation=next_generation,
                context=child_context
            )

            def _add_child_and_link() -> bool:
                if not param_graph.add_element(child_node, allow_duplicates=False):
                    return False
                child_genome.save(str(saved_genome_path))
                param_graph.update_element(child_node.id, {"parent": group_id})
                param_graph.link(model_element, child_node, relation='binds_to')
                if selected_parent.id != child_ind_id:
                    param_graph.link(selected_parent, child_node, relation='parent')

                scatter_group.member_ids.append(child_ind_id)
                param_graph.update_element(scatter_group.id, {"member_ids": scatter_group.member_ids})
                return True

            if graph_lock is not None:
                with graph_lock:
                    added = _add_child_and_link()
            else:
                added = _add_child_and_link()

            if not added:
                print(
                    f"[scatter_evolution_task] Discarding duplicate scattered individual '{child_ind_id}' "
                    f"which already exists in the parameter graph."
                )
                continue

            individual_ids.append(child_ind_id)

            # Optional automatic exemplar generation
            if generate_exemplars and engine_provider is not None:
                try:
                    engine = engine_provider.get_engine()
                    sub_job_id = f"job_exemplar_{uuid.uuid4().hex[:8]}"

                    child_engine_args = dict(node_engine_args)
                    child_engine_args["model_element"] = model_element
                    child_engine_args["individual_elements"] = [child_node]
                    child_engine_args["individual_strengths"] = [1.0]
                    if base_grating:
                        child_engine_args["baseline_grating"] = base_grating

                    if active_jobs is not None:
                        active_jobs[sub_job_id] = {
                            "parent_id": child_ind_id,
                            "group_id": None,
                            "linked_elements": [child_node],
                            "validated_params": {
                                **dumped_params,
                                "model_id": model_id,
                                "operation": operation,
                                "individuals": [{"id": child_ind_id, "strength": 1.0}]
                            },
                            "operation": operation
                        }

                    asyncio.run(engine.execute(operation, job_id=sub_job_id, **child_engine_args, **dumped_params))
                    job_ids.append(sub_job_id)
                    print(f"[scatter_evolution_task] Queued exemplar generation job {sub_job_id} for individual {child_ind_id}")
                except Exception as ex:
                    print(f"[scatter_evolution_task] Warning: Failed to queue exemplar generation for individual {child_ind_id}: {ex}")
                    traceback.print_exc()

        # Persist the whole batch in one write, since every save re-serializes the entire graph
        if graph_lock is not None:
            with graph_lock:
                param_graph.save()
        else:
            param_graph.save()

        if local_jobs is not None and parent_job_id in local_jobs:
            local_jobs[parent_job_id]["status"] = "completed"
            local_jobs[parent_job_id]["progress"] = {
                "value": offspring_size,
                "total": offspring_size,
                "description": "Scatter complete."
            }
            local_jobs[parent_job_id]["result"] = {
                "individual_ids": individual_ids,
                "group_id": group_id,
                "generation": next_generation,
                "job_ids": job_ids
            }
        print(f"[scatter_evolution_task] Scatter completed successfully for job {parent_job_id}")

    except Exception as e:
        print(f"Failed in async evolution scatter: {e}")
        traceback.print_exc()
        if local_jobs is not None and parent_job_id in local_jobs:
            local_jobs[parent_job_id]["status"] = "failed"
            local_jobs[parent_job_id]["progress"] = None
            local_jobs[parent_job_id]["error"] = str(e)
            local_jobs[parent_job_id]["traceback"] = traceback.format_exc()


run_scatter_task = scatter_evolution_task


async def dispatch_scatter_operation(
    data: dict[str, Any],
    param_graph: Any,
    graph_lock: Optional[threading.Lock] = None,
    engine_provider: Optional[Any] = None,
    local_jobs: Optional[dict[str, Any]] = None,
    active_jobs: Optional[dict[str, Any]] = None,
    uid_generator: Optional[Any] = None,
) -> tuple[dict[str, Any], int]:
    """
    Validates and spawns a background scatter operation for seed individual(s).
    """
    if param_graph is None or engine_provider is None:
        return {"error": "No project loaded"}, 400

    params = data.get("params", {}) or {}
    initiator = data.get("initiator") or params.get("initiator") or {}
    initiator_id = initiator.get("id") if isinstance(initiator, dict) else (str(initiator).strip() if initiator else None)
    initiator_type = initiator.get("type") if isinstance(initiator, dict) else None

    raw_parent_input = (
        data.get("parent_ids")
        or params.get("parent_ids")
        or data.get("parents")
        or params.get("parents")
        or data.get("individual_id")
        or params.get("individual_id")
        or data.get("target_node")
        or params.get("target_node")
    )
    parent_ids = extract_individual_parent_ids(raw_parent_input, param_graph=param_graph, graph_lock=graph_lock)

    parent_group_id = (
        data.get("parent_group_id")
        or params.get("parent_group_id")
        or (initiator_id if initiator_type == "group" else None)
    )

    if not parent_ids and parent_group_id:
        def _resolve_grp() -> list[str]:
            grp = param_graph.get_element(parent_group_id)
            if grp and hasattr(grp, "member_ids") and grp.member_ids:
                return extract_individual_parent_ids(grp.member_ids, param_graph=param_graph)
            return []

        if graph_lock is not None:
            with graph_lock:
                parent_ids = _resolve_grp()
        else:
            parent_ids = _resolve_grp()

    if not parent_ids and initiator_id and initiator_type == "individual":
        parent_ids = extract_individual_parent_ids([initiator_id], param_graph=param_graph, graph_lock=graph_lock)

    if not parent_ids:
        precursor_val = (
            data.get("source_audio")
            or data.get("source_audio_id")
            or data.get("precursor_audio_id")
            or data.get("source_node")
            or data.get("source_node_id")
            or (initiator_id if initiator_type == "audio" else None)
        )
        if precursor_val:
            wrap_json, wrap_code = await wrap_precursor_as_individual(
                data=data,
                param_graph=param_graph,
                engine_provider=engine_provider,
                graph_lock=graph_lock,
                uid_generator=uid_generator,
            )
            if wrap_code == 200:
                parent_ids = [wrap_json["node_id"]]
            else:
                return wrap_json, wrap_code

    if not parent_ids:
        return {"error": "Either parent_ids, parents, or an Individual/Group initiator must contain valid individual nodes."}, 400

    offspring_size = int(
        data.get("offspring_size")
        or params.get("offspring_size")
        or data.get("population_size")
        or params.get("population_size")
        or 5
    )
    noise_scale = float(
        data.get("noise_scale")
        if data.get("noise_scale") is not None
        else (
            params.get("noise_scale")
            if params.get("noise_scale") is not None
            else 0.1
        )
    )
    active_flips = float(
        data.get("active_flips")
        if data.get("active_flips") is not None
        else (
            params.get("active_flips")
            if params.get("active_flips") is not None
            else 1.0
        )
    )
    mutation_rate = float(
        data.get("mutation_rate")
        if data.get("mutation_rate") is not None
        else (
            params.get("mutation_rate")
            if params.get("mutation_rate") is not None
            else 1.0
        )
    )
    freeze_inactive = data.get("freeze_inactive")
    if freeze_inactive is None:
        freeze_inactive = params.get("freeze_inactive", True)
    if isinstance(freeze_inactive, str):
        freeze_inactive = freeze_inactive.lower() in ("true", "1", "yes")
    freeze_inactive = bool(freeze_inactive)
    generation_context = data.get("generation_context") or params.get("generation_context") or {}
    merged_generation_context = {**params, **generation_context}

    generate_exemplars = (
        data.get("generate_exemplars")
        if data.get("generate_exemplars") is not None
        else (
            params.get("generate_exemplars")
            if params.get("generate_exemplars") is not None
            else (
                data.get("auto_generate_exemplars")
                if data.get("auto_generate_exemplars") is not None
                else params.get("auto_generate_exemplars", False)
            )
        )
    )
    if isinstance(generate_exemplars, str):
        generate_exemplars = generate_exemplars.lower() in ("true", "1", "yes")
    else:
        generate_exemplars = bool(generate_exemplars)

    parent_job_id = data.get("job_id") or f"evolution_scatter_{uuid.uuid4().hex[:12]}"
    if local_jobs is not None:
        local_jobs[parent_job_id] = {
            "status": "pending",
            "progress": None,
            "result": None,
            "error": None
        }

    # Spawn background task
    # Copying the context keeps the request's room placement in the worker
    thread = threading.Thread(
        target=contextvars.copy_context().run,
        args=(run_scatter_task,),
        kwargs=dict(
            parent_job_id=parent_job_id,
            parent_ids=parent_ids,
            offspring_size=offspring_size,
            noise_scale=noise_scale,
            active_flips=active_flips,
            mutation_rate=mutation_rate,
            freeze_inactive=freeze_inactive,
            generation_context=merged_generation_context,
            generate_exemplars=generate_exemplars,
            param_graph=param_graph,
            graph_lock=graph_lock,
            engine_provider=engine_provider,
            local_jobs=local_jobs,
            active_jobs=active_jobs,
            uid_generator=uid_generator,
        ),
        daemon=True
    )
    thread.start()

    return {
        "success": True,
        "job_id": parent_job_id,
        "status": "pending",
        "message": "Scatter job initialized"
    }, 202


@register
class ScatterOperation(Operation):
    @property
    def name(self) -> str:
        return "scatter"

    @property
    def description(self) -> str:
        return "Scatters variant offspring across the parameter space from seed individual(s)."

    @property
    def category(self) -> str:
        return "evolution"

    @property
    def execution(self) -> str:
        return "queued"

    def execute_task(self, job_id: str, **kwargs: Any) -> None:
        scatter_evolution_task(
            parent_job_id=job_id,
            parent_ids=kwargs.get("parent_ids", []),
            offspring_size=kwargs.get("offspring_size", 5),
            noise_scale=kwargs.get("noise_scale", 0.1),
            active_flips=kwargs.get("active_flips", 1.0),
            mutation_rate=kwargs.get("mutation_rate", 1.0),
            freeze_inactive=kwargs.get("freeze_inactive", True),
            generation_context=kwargs.get("generation_context", {}),
            generate_exemplars=kwargs.get("generate_exemplars", False),
            param_graph=kwargs.get("param_graph"),
            graph_lock=kwargs.get("graph_lock"),
            engine_provider=kwargs.get("engine_provider"),
            local_jobs=kwargs.get("local_jobs"),
            active_jobs=kwargs.get("active_jobs"),
            uid_generator=kwargs.get("uid_generator"),
        )
