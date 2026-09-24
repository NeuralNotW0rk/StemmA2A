"""Unified operation dispatcher coordinating immediate DSP, evolutionary, and generative operations."""

from typing import Any, Optional, Callable
from pathlib import Path
from dataclasses import replace
import uuid
import traceback
import torch
from pydantic import ValidationError

from param_graph.graph import ParameterGraph
from param_graph.elements.artifacts.audio_element import Audio
from param_graph.elements.artifacts.image_element import Image
from param_graph.elements.artifacts.grating_element import Grating
from param_graph.elements.artifacts.latent_element import Latent
from param_graph.elements.artifacts.individual_element import Individual
from param_graph.elements.models.base_model_element import Model
from param_graph.elements.collections.group_element import Group
from param_graph.elements.base_elements import Asset
from param_graph.utils import save_artifact_asset
from utils.audio import save_audio
from utils.form import create_dynamic_model
from utils.uid import XXH3_64, path_from_uid

from .registry import operation_registry
from .task_manager import task_manager, TaskManager
from .evolution.wrap_individual import wrap_precursor_as_individual
from .evolution.mutate import dispatch_mutate_operation
from .evolution.recombine import dispatch_recombine_operation
from .grating.create_grating import dispatch_create_grating_operation


async def dispatch_operation(
    payload: dict[str, Any],
    param_graph: ParameterGraph,
    engine_provider: Optional[Any],
    graph_lock: Any,
    uid_generator: Optional[Any] = None,
    device_accelerator: Optional[torch.device] = None,
    sample_rate: int = 48000,
    active_jobs: Optional[dict[str, Any]] = None,
    local_jobs: Optional[dict[str, Any]] = None,
    task_mgr: Optional[TaskManager] = None,
    trigger_embedding_update_fn: Optional[Callable[..., Any]] = None,
    cache_used_audio_fn: Optional[Callable[[str], Any]] = None,
    resolve_audio_path_fn: Optional[Callable[[str], Any]] = None,
    update_group_labels_fn: Optional[Callable[[str], Any]] = None,
    data_cache_root: Optional[Path] = None,
) -> tuple[dict[str, Any], int]:
    """
    Unified Dispatcher for all operations across DSP, Evolution, and Generative Engine.

    Args:
        payload: Request payload dictionary containing 'operation' and parameters.
        param_graph: Active ParameterGraph instance.
        engine_provider: Active EngineProvider instance.
        graph_lock: Threading lock for graph mutations.
        uid_generator: XXH3_64 UID generator.
        device_accelerator: Torch device for DSP.
        sample_rate: Default sample rate for DSP processing.
        active_jobs: Context tracking for async generative engine jobs.
        local_jobs: Job dictionary for task manager.
        task_mgr: TaskManager instance.
        trigger_embedding_update_fn: Optional callback to trigger embedding updates.
        cache_used_audio_fn: Optional callback to cache audio before processing.
        resolve_audio_path_fn: Optional callback to resolve local audio path.
        update_group_labels_fn: Optional callback to recursively label group members.
        data_cache_root: Path to data cache directory.

    Returns:
        A tuple of (response_dict, status_code).
    """
    if param_graph is None:
        return {"error": "No project loaded"}, 400

    operation_name = payload.get("operation")
    if not operation_name:
        return {"error": "'operation' is required"}, 400

    uid_gen = uid_generator or XXH3_64()
    mgr = task_mgr or task_manager
    jobs_dict = local_jobs if local_jobs is not None else mgr.jobs
    device = device_accelerator or torch.device("cpu")
    cache_root = data_cache_root or (Path.home() / ".stemma2a_data")

    # ---------------------------------------------------------
    # 1. Registered Host Operations (DSP & Evolutionary Tasks)
    # ---------------------------------------------------------
    if operation_registry.has(operation_name):
        op = operation_registry.get(operation_name)
        if op is None:
            return {"error": f"Operation '{operation_name}' could not be resolved."}, 400

        # Immediate In-Band Operations
        if op.execution == "immediate":
            if operation_name in ("wrap_individual", "create_individual"):
                return await wrap_precursor_as_individual(
                    data=payload,
                    param_graph=param_graph,
                    engine_provider=engine_provider,
                    graph_lock=graph_lock,
                    uid_generator=uid_gen,
                )

            # DSP Audio / Synchronous Processing
            try:
                params = payload.get("params", {}) or {}
                form_config = op.get_form_config()
                DynamicArgsModel = create_dynamic_model(form_config)

                validation_target = params if params else payload
                validated_params = DynamicArgsModel.model_validate(validation_target).model_dump()

                # Dynamically resolve all requested node links from the graph
                node_args: dict[str, Any] = {}
                source_elements = []
                for field in form_config:
                    if field.get("type") == "node":
                        field_name = field.get("name")
                        node_id = validated_params.pop(field_name, None)
                        if node_id:
                            element = param_graph.get_element(node_id)
                            if element:
                                node_args[f"{field_name}_element"] = element
                                source_elements.append(element)

                # Resolve and cache audio
                for arg_name, element in list(node_args.items()):
                    if isinstance(element, Audio):
                        if cache_used_audio_fn:
                            cache_used_audio_fn(element.id)
                        if resolve_audio_path_fn:
                            valid_path = resolve_audio_path_fn(element.id)
                            if valid_path and str(valid_path) != element.file.path:
                                element.file = replace(element.file, path=str(valid_path))
                                node_args[arg_name] = element

                process_results = op.execute(
                    device=device,
                    sample_rate=sample_rate,
                    **node_args,
                    **validated_params
                )

                if isinstance(process_results, tuple) and len(process_results) == 2 and isinstance(process_results[0], dict):
                    return process_results  # type: ignore

                output_dir = param_graph.root / "process"
                output_dir.mkdir(parents=True, exist_ok=True)

                final_artifacts = []
                is_batch = len(process_results) > 1

                for artifact_blueprint, raw_data in process_results:
                    local_path = cache_root / path_from_uid(artifact_blueprint.id)
                    local_path.parent.mkdir(parents=True, exist_ok=True)

                    if isinstance(artifact_blueprint, Audio):
                        save_audio(raw_data.cpu(), local_path, artifact_blueprint.sample_rate, format="wav")
                    elif isinstance(artifact_blueprint, Latent):
                        torch.save(raw_data.cpu(), local_path)
                    elif isinstance(artifact_blueprint, Image):
                        from torchvision.utils import save_image
                        save_image(raw_data, local_path, format="png", normalize=True, value_range=(-1, 1))

                    artifact_blueprint.file = replace(artifact_blueprint.file, path=str(local_path))
                    final_artifact = save_artifact_asset(artifact_blueprint, output_dir, asset_name="file")
                    final_artifacts.append(final_artifact)

                # Integrate into graph
                collection_dict = None
                req_group_id = (
                    payload.get("group_id")
                    or payload.get("batch_id")
                    or params.get("group_id")
                    or params.get("batch_id")
                )

                with graph_lock:
                    for artifact in final_artifacts:
                        param_graph.add_element(artifact)
                        for source_elem in source_elements:
                            param_graph.link(source_elem, artifact, relation='source')

                    if req_group_id:
                        if not param_graph.G.has_node(req_group_id):
                            group_node = Group(id=req_group_id, member_ids=[], member_type=final_artifacts[0].type if final_artifacts else None)
                            param_graph.add_element(group_node)
                            for el in source_elements:
                                if param_graph.G.has_node(el.id):
                                    el_pos = param_graph.G.nodes[el.id].get('position')
                                    if el_pos:
                                        param_graph.update_element(req_group_id, {"position": {"x": el_pos["x"] + 80, "y": el_pos["y"] + 80}})
                                        break

                        for artifact in final_artifacts:
                            param_graph.update_element(artifact.id, {"parent": req_group_id})
                            group_node_attrs = param_graph.G.nodes[req_group_id]
                            if 'member_ids' not in group_node_attrs or not isinstance(group_node_attrs['member_ids'], list):
                                group_node_attrs['member_ids'] = []
                            if artifact.id not in group_node_attrs['member_ids']:
                                group_node_attrs['member_ids'].append(artifact.id)

                        if update_group_labels_fn:
                            update_group_labels_fn(req_group_id)
                        group_elem = param_graph.get_element(req_group_id)
                        if group_elem:
                            collection_dict = group_elem.to_dict()

                    elif is_batch:
                        member_ids = [a.id for a in final_artifacts]
                        group_id = uid_gen.from_uids(member_ids)
                        group = Group(id=group_id, member_ids=member_ids, member_type=final_artifacts[0].type if final_artifacts else None)
                        param_graph.add_element(group)

                        for el in source_elements:
                            if param_graph.G.has_node(el.id):
                                el_pos = param_graph.G.nodes[el.id].get('position')
                                if el_pos:
                                    param_graph.update_element(group_id, {"position": {"x": el_pos["x"] + 80, "y": el_pos["y"] + 80}})
                                    break

                        for m_id in member_ids:
                            param_graph.update_element(m_id, {"parent": group_id})

                        if update_group_labels_fn:
                            update_group_labels_fn(group_id)
                        group_elem = param_graph.get_element(group_id)
                        if group_elem:
                            collection_dict = group_elem.to_dict()

                    param_graph.save()

                if trigger_embedding_update_fn:
                    trigger_embedding_update_fn()

                response_data = {
                    "message": f"Operation '{op.name}' completed successfully.",
                    "status": "completed",
                    "success": True,
                }
                if req_group_id:
                    response_data["artifact"] = final_artifacts[0].to_dict() if final_artifacts else None
                    response_data["node_id"] = final_artifacts[0].id if final_artifacts else None
                    if collection_dict:
                        response_data["collection"] = collection_dict
                elif is_batch:
                    response_data["artifacts"] = [a.to_dict() for a in final_artifacts]
                    if collection_dict:
                        response_data["collection"] = collection_dict
                        response_data["node_id"] = collection_dict["id"]
                else:
                    response_data["artifact"] = final_artifacts[0].to_dict() if final_artifacts else None
                    response_data["node_id"] = final_artifacts[0].id if final_artifacts else None

                return response_data, 200

            except (ValidationError, ValueError) as e:
                return {"error": "Invalid request", "details": str(e)}, 400
            except Exception as e:
                traceback.print_exc()
                return {"error": str(e), "traceback": traceback.format_exc()}, 500

        # Queued Background Host Tasks
        elif op.execution == "queued":
            if operation_name == "mutate":
                return await dispatch_mutate_operation(
                    data=payload,
                    param_graph=param_graph,
                    graph_lock=graph_lock,
                    engine_provider=engine_provider,
                    local_jobs=jobs_dict,
                    active_jobs=active_jobs,
                    uid_generator=uid_gen,
                )
            elif operation_name in ("recombine", "reproduce"):
                return await dispatch_recombine_operation(
                    data=payload,
                    param_graph=param_graph,
                    graph_lock=graph_lock,
                    engine_provider=engine_provider,
                    local_jobs=jobs_dict,
                    active_jobs=active_jobs,
                    uid_generator=uid_gen,
                )
            elif operation_name == "create_grating":
                return await dispatch_create_grating_operation(
                    data=payload,
                    param_graph=param_graph,
                    engine_provider=engine_provider,
                    graph_lock=graph_lock,
                    local_jobs=jobs_dict,
                )
            else:
                job_id = payload.get("job_id") or f"{operation_name}_{uuid.uuid4().hex[:12]}"
                mgr.submit_task(
                    op.execute_task,
                    job_id=job_id,
                    name=op.name.capitalize(),
                    **payload
                )
                return {
                    "success": True,
                    "job_id": job_id,
                    "status": "pending",
                    "message": f"Task '{op.name}' initiated."
                }, 202

    # ---------------------------------------------------------
    # 2. Generative Neural Engine Operations (Tier 3)
    # ---------------------------------------------------------
    if engine_provider is None:
        return {"error": "Engine provider not initialized"}, 400

    engine = engine_provider.get_engine()
    if engine is None:
        return {"error": "Engine could not be initialized"}, 400

    try:
        model_id = payload.get("model_id")
        job_id = payload.get("job_id")
        params = payload.get("params", {}) or {}

        initiator = payload.get("initiator") or params.get("initiator") or {}
        initiator_id = initiator.get("id") if isinstance(initiator, dict) else (str(initiator).strip() if initiator else None)
        initiator_type = initiator.get("type") if isinstance(initiator, dict) else None

        # Resolve targeted Individual elements (if any)
        target_individual_nodes: list[Individual] = []
        target_individual_strengths: list[float] = []

        raw_individuals = payload.get("individuals") or params.get("individuals")
        individual_id = (
            payload.get("individual_id")
            or params.get("individual_id")
            or (initiator_id if initiator_type == "individual" else None)
        )

        with graph_lock:
            if raw_individuals and isinstance(raw_individuals, list):
                for ind_item in raw_individuals:
                    ind_id = ind_item.get("id") if isinstance(ind_item, dict) else ind_item
                    ind_str = ind_item.get("strength", 1.0) if isinstance(ind_item, dict) else 1.0
                    if ind_id and param_graph.G.has_node(ind_id):
                        ind_elem = param_graph.get_element(ind_id)
                        if isinstance(ind_elem, Individual) and ind_elem not in target_individual_nodes:
                            target_individual_nodes.append(ind_elem)
                            target_individual_strengths.append(float(ind_str))
            elif individual_id and param_graph.G.has_node(individual_id):
                ind_elem = param_graph.get_element(individual_id)
                if isinstance(ind_elem, Individual):
                    target_individual_nodes.append(ind_elem)
                    target_individual_strengths.append(1.0)

            # Auto-infer model_id from target individual if not explicitly specified
            if not model_id and target_individual_nodes:
                model_id = target_individual_nodes[0].base_model_id

        if not job_id:
            job_id = str(uuid.uuid4())
        if not model_id:
            return {"error": "A base model must be linked to execute generative operations."}, 400

        model_element = param_graph.get_element(model_id)
        if not isinstance(model_element, Model):
            return {"error": f"Node '{model_id}' is not a valid model."}, 400

        adapter_config = await engine.get_adapter_config(model_element.adapter)
        form_config = adapter_config.get(operation_name, [])
        if not form_config:
            # Check if adapter_config is itself a list
            if isinstance(adapter_config, list):
                form_config = adapter_config
            else:
                return {"error": f"Adapter '{model_element.adapter}' has no '{operation_name}' configuration"}, 404

        DynamicArgsModel = create_dynamic_model(form_config)
        validation_target = params if params else payload
        validated_params = DynamicArgsModel.model_validate(validation_target)
        dumped_params = validated_params.model_dump()

        node_engine_args: dict[str, Any] = {}
        for field in form_config:
            if field.get("type") == "node":
                field_name = field.get("name")
                node_id = dumped_params.pop(field_name, None)
                if node_id:
                    element = param_graph.get_element(node_id)
                    if element:
                        node_engine_args[f"{field_name}_element"] = element

        # Gratings
        grating_strengths = []
        gratings = payload.get("gratings") or params.get("gratings")
        if gratings:
            grating_elements = []
            for g_conf in gratings:
                g_id = g_conf.get("id") if isinstance(g_conf, dict) else g_conf
                g_element = param_graph.get_element(g_id)
                if not isinstance(g_element, Grating):
                    return {"error": f"Node '{g_id}' is not a valid grating."}, 400
                grating_elements.append(g_element)
                grating_strengths.append(float(g_conf.get("strength", 1.0)) if isinstance(g_conf, dict) else 1.0)
            node_engine_args["grating_elements"] = grating_elements

        # Individuals and Baseline Grating
        if target_individual_nodes:
            node_engine_args["individual_elements"] = target_individual_nodes
            first_ind = target_individual_nodes[0]
            base_grating_elem = None
            if first_ind.baseline_grating_id and param_graph.G.has_node(first_ind.baseline_grating_id):
                base_grating_elem = param_graph.get_element(first_ind.baseline_grating_id)
            elif first_ind.context.get("baseline_file_path"):
                base_grating_elem = Grating(
                    id=f"grating_{uuid.uuid4().hex[:8]}",
                    name="Baseline Grating",
                    context={},
                    file=Asset(path=str(first_ind.context.get("baseline_file_path")), uid=f"grating_{uuid.uuid4().hex[:8]}", extension=".safetensors"),
                    base_model_id=model_id,
                    elements=first_ind.context.get("baseline_elements") or []
                )
            if base_grating_elem:
                node_engine_args["baseline_grating"] = base_grating_elem

        source_audio_id = payload.get("source_audio_id") or params.get("source_audio_id")
        if source_audio_id:
            source_audio_element = param_graph.get_element(source_audio_id)
            if not isinstance(source_audio_element, Audio):
                return {"error": f"Node '{source_audio_id}' is not a valid audio artifact."}, 400
            node_engine_args["source_audio_element"] = source_audio_element

        resolved_elements = []
        for val in node_engine_args.values():
            if isinstance(val, list):
                resolved_elements.extend(val)
            else:
                resolved_elements.append(val)

        # Cache external audio and validate artifacts
        for arg_name, element in list(node_engine_args.items()):
            if isinstance(element, list):
                for el in element:
                    if not isinstance(el, (Audio, Model, Grating, Latent, Individual)):
                        return {"error": f"Node '{el.id}' is not a valid artifact."}, 400
            else:
                if isinstance(element, Audio):
                    if cache_used_audio_fn:
                        cache_used_audio_fn(element.id)
                    if resolve_audio_path_fn:
                        valid_path = resolve_audio_path_fn(element.id)
                        if valid_path and str(valid_path) != element.file.path:
                            element.file = replace(element.file, path=str(valid_path))
                            node_engine_args[arg_name] = element

                if not isinstance(element, (Audio, Model, Grating, Latent, Individual)):
                    field_name = arg_name.removesuffix("_element")
                    return {"error": f"Node '{element.id}' for field '{field_name}' is not a valid artifact."}, 400

        engine_args = {"model_element": model_element, **node_engine_args}
        if gratings:
            engine_args["grating_strengths"] = grating_strengths
            engine_args["gratings"] = gratings
        if target_individual_nodes:
            engine_args["individual_strengths"] = target_individual_strengths

        parent_id = None
        if target_individual_nodes:
            parent_id = target_individual_nodes[0].id
            linked_elements = [target_individual_nodes[0], *[el for el in resolved_elements if el.id != target_individual_nodes[0].id]]
        elif "grating_elements" in node_engine_args and node_engine_args["grating_elements"]:
            linked_elements = [*resolved_elements]
        else:
            linked_elements = [model_element, *resolved_elements]

        group_id = payload.get("group_id") or payload.get("batch_id")
        if group_id:
            with graph_lock:
                if not param_graph.G.has_node(group_id):
                    group_element = Group(id=group_id, member_type="audio" if operation_name != "invert" else "latent")
                    param_graph.add_element(group_element)
                    for el in linked_elements:
                        if param_graph.G.has_node(el.id):
                            el_pos = param_graph.G.nodes[el.id].get('position')
                            if el_pos:
                                param_graph.update_element(group_id, {"position": {"x": el_pos["x"] + 80, "y": el_pos["y"] + 80}})
                                break
                    param_graph.save()

        returned_job_id = await engine.execute(operation_name, job_id=job_id, **engine_args, **dumped_params)
        if returned_job_id:
            job_id = returned_job_id

        v_params = dumped_params
        v_params["model_id"] = model_id
        v_params["operation"] = operation_name
        if gratings:
            v_params["gratings"] = gratings
        if target_individual_nodes:
            v_params["individuals"] = [
                {"id": ind.id, "strength": str_val}
                for ind, str_val in zip(target_individual_nodes, target_individual_strengths)
            ]

        if active_jobs is not None:
            active_jobs[job_id] = {
                "parent_id": parent_id,
                "group_id": group_id,
                "linked_elements": linked_elements,
                "validated_params": v_params,
                "operation": operation_name
            }

        return {
            "message": "Job started successfully.",
            "job_id": job_id,
            "status": "pending",
            "success": True,
        }, 202

    except (ValidationError, ValueError) as e:
        return {"error": "Invalid request", "details": str(e)}, 400
    except Exception as e:
        traceback.print_exc()
        return {"error": str(e), "traceback": traceback.format_exc()}, 500
