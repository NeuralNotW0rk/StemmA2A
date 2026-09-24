"""Agnostic evolutionary recombination / breeding operation, operators, and background task execution."""

from typing import Any, Optional, Union
from dataclasses import dataclass
import os
import copy
import uuid
import threading
import traceback
import asyncio
from coolname import generate_slug

from neutral_selection.representation.individual import Individual as NSIndividual
from neutral_selection.representation.population import Population
from neutral_selection.representation.lineage import Lineage
from neutral_selection.variation.selection import SelectionStrategy
from neutral_selection.variation.recombination import RecombinationStrategy
from neutral_selection.variation.mutation import MutationStrategy
from neutral_selection.pipeline import step, EvolutionPipeline
from neutral_selection.builders import (
    build_selection_strategy,
    build_crossover_strategy,
    build_mutation_strategy,
    build_replacement_strategy,
    build_pipeline,
)

from param_graph.elements.artifacts.individual_element import Individual
from param_graph.elements.artifacts.grating_element import Grating
from param_graph.elements.artifacts.bundle_element import Bundle
from param_graph.elements.models.base_model_element import Model
from param_graph.elements.collections.group_element import Group
from param_graph.elements.base_elements import Asset
from evolution.lora_genome import (
    LoRAGenome,
    get_lora_mutation_strategy,
    get_lora_crossover_strategy,
)
from utils.uid import XXH3_64

from ..base import Operation
from ..registry import register
from .resolution import extract_individual_parent_ids


@dataclass
class LineageRecord:
    """Tracks the lineage and recombination history of an offspring individual."""
    parent_ids: list[str]
    crossover_applied: bool
    mutated: bool


@dataclass
class RecombinedOffspring:
    """Wraps a newly bred/recombined Individual alongside its lineage metadata."""
    individual: NSIndividual
    lineage: LineageRecord


def recombine_offspring(
    parents: Union[Population, list[NSIndividual]],
    offspring_count: int,
    selection_strategy: SelectionStrategy,
    crossover_strategy: Optional[RecombinationStrategy] = None,
    mutation_strategy: Optional[MutationStrategy] = None,
    crossover_prob: float = 0.8,
    elitism: int = 0,
    parent_ids: Optional[list[str]] = None,
) -> list[RecombinedOffspring]:
    """
    Breeds an offspring population from given parents using NeutralSelection's composable operators.
    """
    parent_pop = parents if isinstance(parents, Population) else Population(list(parents))
    if len(parent_pop) == 0:
        raise ValueError("Cannot breed offspring from an empty parent population.")

    pipeline = EvolutionPipeline(
        selection_strategy=selection_strategy,
        crossover_strategy=crossover_strategy,
        mutation_strategy=mutation_strategy,
        crossover_prob=crossover_prob,
        elitism=elitism,
    )

    offspring_pop = pipeline.step(
        parents=parent_pop,
        offspring_count=offspring_count,
        target_size=offspring_count,
        parent_ids=parent_ids,
    )

    results: list[RecombinedOffspring] = []
    for off in offspring_pop:
        off.clear_expression()
        p_ids = list(off.lineage.parent_ids) if off.lineage and off.lineage.parent_ids else []
        c_app = bool(off.lineage.crossover_applied) if off.lineage else False
        mut = bool(off.lineage.mutated) if off.lineage else False
        results.append(RecombinedOffspring(
            individual=off,
            lineage=LineageRecord(
                parent_ids=p_ids,
                crossover_applied=c_app,
                mutated=mut,
            )
        ))

    return results


def recombine_evolution_task(
    parent_job_id: str,
    parent_ids: list[str] | None,
    parent_bundle_id: str | None,
    offspring_size: int | None,
    selection_cfg: dict,
    crossover_cfg: dict,
    mutation_cfg: dict,
    crossover_prob: float,
    elitism: int,
    generation_context: dict,
    default_fitness: float = 0.0,
    generate_exemplars: bool = False,
    param_graph: Any = None,
    graph_lock: Optional[threading.Lock] = None,
    engine_provider: Optional[Any] = None,
    local_jobs: Optional[dict[str, Any]] = None,
    active_jobs: Optional[dict[str, Any]] = None,
    uid_generator: Optional[Any] = None,
) -> None:
    """
    Background worker that breeds a new generation from parent individuals,
    creates child nodes in the parameter graph with lineage relationships,
    and groups offspring in a generation Group.
    """
    uid_gen = uid_generator or XXH3_64()

    try:
        if local_jobs is not None and parent_job_id in local_jobs:
            local_jobs[parent_job_id]["status"] = "running"
        print(f"[_reproduce_evolution_task] Started reproduction task for job {parent_job_id}")

        def _setup_recombination() -> tuple[
            list[Individual], Any, str, Any, Any, int, int, dict[str, Any], str,
            dict[str, Any], dict[str, Any], list[RecombinedOffspring], Group
        ]:
            parent_nodes: list[Individual] = []
            if parent_ids:
                for pid in parent_ids:
                    if param_graph.G.has_node(pid):
                        node = param_graph.get_element(pid)
                        if isinstance(node, Individual) and node not in parent_nodes:
                            parent_nodes.append(node)
            elif parent_bundle_id:
                bundle_node = param_graph.get_element(parent_bundle_id)
                if isinstance(bundle_node, Bundle):
                    for mid in bundle_node.member_ids:
                        if param_graph.G.has_node(mid):
                            node = param_graph.get_element(mid)
                            if isinstance(node, Individual) and node not in parent_nodes:
                                parent_nodes.append(node)

            if not parent_nodes:
                raise ValueError("No valid parent individuals could be resolved.")

            first_parent = parent_nodes[0]
            model_id = first_parent.base_model_id
            baseline_grating_id = first_parent.baseline_grating_id
            baseline_elements = first_parent.context.get("baseline_elements")
            baseline_file_path = first_parent.context.get("baseline_file_path")

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

            model_element = param_graph.get_element(model_id)
            if not model_element:
                raise ValueError(f"Base model '{model_id}' not found in parameter graph.")

            max_gen = max(getattr(p, "generation", 0) for p in parent_nodes)
            next_generation = max_gen + 1
            target_offspring_count = offspring_size or len(parent_nodes)

            first_parent_ctx = copy.deepcopy(getattr(first_parent, "context", {}) or {})
            merged_context = {**first_parent_ctx, **generation_context}
            merged_context["model_id"] = model_id
            merged_context["baseline_grating_id"] = baseline_grating_id
            merged_context["generation"] = next_generation

            operation = "generate"
            node_engine_args: dict[str, Any] = {}
            dumped_params: dict[str, Any] = {}

            if generate_exemplars and engine_provider is not None:
                engine = engine_provider.get_engine()
                form_config = asyncio.run(engine.get_adapter_config(model_element.adapter))
                form_config = form_config.get(operation, []) if isinstance(form_config, dict) else form_config

                from utils.form import create_dynamic_model
                DynamicArgsModel = create_dynamic_model(form_config)
                validated_params = DynamicArgsModel.model_validate(generation_context)
                dumped_params = validated_params.model_dump()

                for field in form_config:
                    if field.get("type") == "node":
                        field_name = field.get("name")
                        node_id = dumped_params.pop(field_name, None)
                        if node_id:
                            element = param_graph.get_element(node_id)
                            if element:
                                node_engine_args[f"{field_name}_element"] = element

            # Construct NeutralSelection Population
            parent_ns_individuals: list[NSIndividual] = []
            parent_id_list: list[str] = []

            for p_node in parent_nodes:
                gpath = param_graph.get_path_from_id(p_node.id) or p_node.file.path
                genome = LoRAGenome.load(gpath)
                fit_val = p_node.fitness if p_node.fitness is not None else default_fitness

                ns_ind = NSIndividual(
                    genotype=genome,
                    fitness=float(fit_val),
                    lineage=Lineage(parent_ids=[p_node.id]),
                    metadata={"id": p_node.id, "name": p_node.name}
                )
                parent_ns_individuals.append(ns_ind)
                parent_id_list.append(p_node.id)

            selection_strat = build_selection_strategy(selection_cfg)
            crossover_type = crossover_cfg.get("type", "two_point") if isinstance(crossover_cfg, dict) else str(crossover_cfg)
            crossover_strat = get_lora_crossover_strategy(
                strategy_type=crossover_type,
                num_cut_points=int(crossover_cfg.get("num_cut_points", 1)) if isinstance(crossover_cfg, dict) else 1,
                swap_prob=float(crossover_cfg.get("swap_prob", 0.5)) if isinstance(crossover_cfg, dict) else 0.5,
                blend_factor=float(crossover_cfg.get("blend_factor", 0.5)) if isinstance(crossover_cfg, dict) else 0.5,
            )

            lora_noise = float(mutation_cfg.get("lora_noise", merged_context.get("lora_noise", 0.05)))
            active_flip_prob = float(mutation_cfg.get("active_flip_prob", merged_context.get("active_flip_prob", 0.05)))
            mutation_rate = float(mutation_cfg.get("mutation_rate", 0.1))
            mutation_strat = get_lora_mutation_strategy(
                lora_noise=lora_noise,
                active_flip_prob=active_flip_prob,
                mutation_rate=mutation_rate
            )

            offspring_records: list[RecombinedOffspring] = recombine_offspring(
                parents=parent_ns_individuals,
                offspring_count=target_offspring_count,
                selection_strategy=selection_strat,
                crossover_strategy=crossover_strat,
                mutation_strategy=mutation_strat,
                crossover_prob=crossover_prob,
                elitism=elitism,
                parent_ids=parent_id_list,
            )

            gen_group_id = f"group_{uid_gen.from_string(str(uuid.uuid4()))}"
            gen_group = Group(
                id=gen_group_id,
                member_ids=[],
                member_type='individual'
            )

            param_graph.add_element(gen_group)
            param_graph.update_element(gen_group.id, {"alias": f"Recombination (Gen {next_generation})"})
            param_graph.save()

            return (
                parent_nodes, model_element, model_id, baseline_grating_id,
                base_grating, next_generation, target_offspring_count, merged_context,
                operation, node_engine_args, dumped_params, offspring_records, gen_group
            )

        if graph_lock is not None:
            with graph_lock:
                (
                    parent_nodes, model_element, model_id, baseline_grating_id,
                    base_grating, next_generation, target_offspring_count, merged_context,
                    operation, node_engine_args, dumped_params, offspring_records, gen_group
                ) = _setup_recombination()
        else:
            (
                parent_nodes, model_element, model_id, baseline_grating_id,
                base_grating, next_generation, target_offspring_count, merged_context,
                operation, node_engine_args, dumped_params, offspring_records, gen_group
            ) = _setup_recombination()

        baseline_elements = parent_nodes[0].context.get("baseline_elements")
        baseline_file_path = parent_nodes[0].context.get("baseline_file_path")
        gen_group_id = gen_group.id
        parent_id_list = [p.id for p in parent_nodes]
        mutation_rate = float(mutation_cfg.get("mutation_rate", 0.1))

        job_ids: list[str] = []
        individual_ids: list[str] = []

        for i, record in enumerate(offspring_records):
            if local_jobs is not None and parent_job_id in local_jobs:
                local_jobs[parent_job_id]["progress"] = {
                    "value": i,
                    "total": target_offspring_count,
                    "description": f"Processing offspring {i+1} of {target_offspring_count}..."
                }

            child_genome = record.individual.genotype
            genome_state_dict = child_genome.get_state_dict()
            genome_uid = uid_gen.from_state_dict(genome_state_dict)
            child_ind_id = f"individual_{genome_uid}"

            output_dir = param_graph.root / "generate"
            os.makedirs(output_dir, exist_ok=True)
            child_genome_path = output_dir / f"{child_ind_id}.safetensors"
            child_genome.save(str(child_genome_path))

            child_context = copy.deepcopy(merged_context)
            child_context["baseline_elements"] = baseline_elements
            child_context["baseline_file_path"] = str(baseline_file_path) if baseline_file_path else None
            child_context["recombine_operation"] = {
                "crossover_type": crossover_cfg.get("type", "hierarchical") if isinstance(crossover_cfg, dict) else str(crossover_cfg),
                "selection_type": selection_cfg.get("type", "tournament") if isinstance(selection_cfg, dict) else str(selection_cfg),
                "default_fitness": default_fitness,
                "mutation_rate": mutation_rate,
                "crossover_prob": crossover_prob,
                "elitism": elitism,
                "selection_cfg": selection_cfg,
                "crossover_cfg": crossover_cfg,
                "mutation_cfg": mutation_cfg,
                "offspring_size": target_offspring_count,
                "parent_ids": parent_id_list,
                "model_id": model_id,
                "baseline_grating_id": baseline_grating_id,
                "generation": next_generation
            }
            child_context["lineage"] = {
                "parent_ids": record.lineage.parent_ids,
                "crossover_applied": record.lineage.crossover_applied,
                "mutated": record.lineage.mutated
            }

            slug = generate_slug(2)
            child_node = Individual(
                id=child_ind_id,
                name=f"Gen {next_generation} - Ind {i+1} ({slug})",
                file=Asset(path=str(child_genome_path), uid=child_ind_id, extension=".safetensors"),
                base_model_id=model_id,
                baseline_grating_id=baseline_grating_id,
                generation=next_generation,
                context=child_context
            )

            def _add_child() -> None:
                param_graph.add_element(child_node)
                param_graph.update_element(child_node.id, {"parent": gen_group_id})
                param_graph.link(model_element, child_node, relation='binds_to')

                for p_id in record.lineage.parent_ids:
                    if param_graph.G.has_node(p_id):
                        p_elem = param_graph.get_element(p_id)
                        param_graph.link(p_elem, child_node, relation='parent')

                gen_group.member_ids.append(child_ind_id)
                param_graph.update_element(gen_group.id, {"member_ids": gen_group.member_ids})
                param_graph.save()

            if graph_lock is not None:
                with graph_lock:
                    _add_child()
            else:
                _add_child()

            individual_ids.append(child_ind_id)

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
                    print(f"[_recombine_evolution_task] Queued exemplar generation job {sub_job_id} for individual {child_ind_id}")
                except Exception as ex:
                    print(f"[_recombine_evolution_task] Warning: Failed to queue exemplar generation for individual {child_ind_id}: {ex}")
                    traceback.print_exc()

        if local_jobs is not None and parent_job_id in local_jobs:
            local_jobs[parent_job_id]["status"] = "completed"
            local_jobs[parent_job_id]["progress"] = {
                "value": target_offspring_count,
                "total": target_offspring_count,
                "description": "Recombination complete."
            }
            local_jobs[parent_job_id]["result"] = {
                "individual_ids": individual_ids,
                "group_id": gen_group_id,
                "generation": next_generation,
                "job_ids": job_ids
            }
        print(f"[_recombine_evolution_task] Evolution recombination completed successfully for job {parent_job_id}")

    except Exception as e:
        print(f"Failed in async evolution reproduction: {e}")
        traceback.print_exc()
        if local_jobs is not None and parent_job_id in local_jobs:
            local_jobs[parent_job_id]["status"] = "failed"
            local_jobs[parent_job_id]["progress"] = None
            local_jobs[parent_job_id]["error"] = str(e)
            local_jobs[parent_job_id]["traceback"] = traceback.format_exc()


run_recombine_task = recombine_evolution_task


async def dispatch_recombine_operation(
    data: dict[str, Any],
    param_graph: Any,
    graph_lock: Optional[threading.Lock] = None,
    engine_provider: Optional[Any] = None,
    local_jobs: Optional[dict[str, Any]] = None,
    active_jobs: Optional[dict[str, Any]] = None,
    uid_generator: Optional[Any] = None,
) -> tuple[dict[str, Any], int]:
    """
    Validates and spawns a background recombination operation for parent individuals or group.
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

    parent_bundle_id = data.get("parent_bundle_id") or params.get("parent_bundle_id")
    parent_group_id = (
        data.get("parent_group_id")
        or params.get("parent_group_id")
        or (initiator_id if initiator_type in ("group", "bundle") else None)
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

    if not parent_ids and not parent_bundle_id and initiator_id and initiator_type == "individual":
        parent_ids = extract_individual_parent_ids([initiator_id], param_graph=param_graph, graph_lock=graph_lock)

    if not parent_ids and not parent_bundle_id:
        return {"error": "Either parent_ids, parent_bundle_id, or an Individual/Group initiator must contain valid individuals."}, 400

    offspring_size = data.get("offspring_size") or params.get("offspring_size")
    offspring_size = int(offspring_size) if offspring_size is not None else None

    s_type = data.get("selection_type") or params.get("selection_type") or "tournament"
    c_type = data.get("crossover_type") or params.get("crossover_type") or "two_point"

    selection_cfg = data.get("selection") or params.get("selection") or {"type": s_type, "tournament_size": 2}
    crossover_cfg = data.get("crossover") or params.get("crossover") or {"type": c_type}
    mutation_cfg = data.get("mutation") or params.get("mutation") or {
        "mutation_rate": float(data.get("mutation_rate") or params.get("mutation_rate") or 0.1),
        "lora_noise": float(data.get("lora_noise") or params.get("lora_noise") or 0.05),
        "active_flip_prob": float(data.get("active_flip_prob") or params.get("active_flip_prob") or 0.05)
    }

    crossover_prob = float(data.get("crossover_prob") or params.get("crossover_prob") or 1.0)
    elitism = int(data.get("elitism") or params.get("elitism") or 0)
    default_fitness = float(data.get("default_fitness") or params.get("default_fitness") or 0.0)

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

    parent_job_id = data.get("job_id") or f"evolution_recombine_{uuid.uuid4().hex[:12]}"
    if local_jobs is not None:
        local_jobs[parent_job_id] = {
            "status": "pending",
            "progress": None,
            "result": None,
            "error": None
        }

    thread = threading.Thread(
        target=run_recombine_task,
        kwargs=dict(
            parent_job_id=parent_job_id,
            parent_ids=parent_ids,
            parent_bundle_id=parent_bundle_id,
            offspring_size=offspring_size,
            selection_cfg=selection_cfg,
            crossover_cfg=crossover_cfg,
            mutation_cfg=mutation_cfg,
            crossover_prob=crossover_prob,
            elitism=elitism,
            generation_context=merged_generation_context,
            default_fitness=default_fitness,
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
        "message": "Recombination job initialized"
    }, 202


@register
class RecombineOperation(Operation):
    @property
    def name(self) -> str:
        return "recombine"

    @property
    def description(self) -> str:
        return "Breeds a new generation of individuals from parent individuals or group."

    @property
    def category(self) -> str:
        return "evolution"

    @property
    def execution(self) -> str:
        return "queued"

    @property
    def initiator_types(self) -> list[str]:
        return ["individual", "group", "bundle"]

    def execute_task(self, job_id: str, **kwargs: Any) -> None:
        recombine_evolution_task(
            parent_job_id=job_id,
            parent_ids=kwargs.get("parent_ids"),
            parent_bundle_id=kwargs.get("parent_bundle_id"),
            offspring_size=kwargs.get("offspring_size"),
            selection_cfg=kwargs.get("selection_cfg", {"type": "tournament", "tournament_size": 2}),
            crossover_cfg=kwargs.get("crossover_cfg", {"type": "two_point"}),
            mutation_cfg=kwargs.get("mutation_cfg", {"mutation_rate": 0.1}),
            crossover_prob=kwargs.get("crossover_prob", 1.0),
            elitism=kwargs.get("elitism", 0),
            generation_context=kwargs.get("generation_context", {}),
            default_fitness=kwargs.get("default_fitness", 0.0),
            generate_exemplars=kwargs.get("generate_exemplars", False),
            param_graph=kwargs.get("param_graph"),
            graph_lock=kwargs.get("graph_lock"),
            engine_provider=kwargs.get("engine_provider"),
            local_jobs=kwargs.get("local_jobs"),
            active_jobs=kwargs.get("active_jobs"),
            uid_generator=kwargs.get("uid_generator"),
        )
