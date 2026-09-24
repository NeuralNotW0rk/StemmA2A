"""Dynamic Bending Grating creation operation and background task execution."""

from typing import Any, Optional
import uuid
import threading
import traceback
import asyncio

from param_graph.graph import ParameterGraph
from param_graph.elements.models.base_model_element import Model
from operations.base import Operation
from operations.registry import register


async def create_grating_task(
    job_id: str,
    data: dict[str, Any],
    param_graph: ParameterGraph,
    engine_provider: Any,
    graph_lock: Optional[threading.Lock] = None,
    local_jobs: Optional[dict[str, Any]] = None,
) -> None:
    """
    Background worker that dynamically creates a Diffracture Grating via the active engine,
    attaches it to the model node, and saves it into the parameter graph.
    """
    try:
        if local_jobs is not None and job_id in local_jobs:
            local_jobs[job_id]["status"] = "running"
            local_jobs[job_id]["progress"] = {
                "value": 15,
                "total": 100,
                "description": "Validating model and layer configuration..."
            }

        params = data.get("params", {}) or {}
        model_id = data.get("model_id") or params.get("model_id")
        grating_name = data.get("name") or params.get("name") or "New Grating"
        elements_input = data.get("elements") or params.get("elements", [])
        context_data = data.get("context") or params.get("context", {})

        if not model_id or not elements_input:
            raise ValueError("model_id and elements list are required")

        model_element = param_graph.get_element(model_id)
        if not isinstance(model_element, Model):
            raise ValueError(f"Node '{model_id}' is not a valid model.")

        if local_jobs is not None and job_id in local_jobs:
            local_jobs[job_id]["progress"] = {
                "value": 45,
                "total": 100,
                "description": "Constructing grating topology..."
            }

        engine = engine_provider.get_engine()
        grating_artifact = await engine.create_grating(model_element, grating_name, elements_input)
        grating_artifact.context = context_data

        if local_jobs is not None and job_id in local_jobs:
            local_jobs[job_id]["progress"] = {
                "value": 85,
                "total": 100,
                "description": "Saving grating artifact to parameter graph..."
            }

        def _link_and_save() -> None:
            param_graph.add_element(grating_artifact)
            param_graph.link(model_element, grating_artifact, relation='binds_to')
            param_graph.save()

        if graph_lock is not None:
            with graph_lock:
                _link_and_save()
        else:
            _link_and_save()

        if local_jobs is not None and job_id in local_jobs:
            local_jobs[job_id]["status"] = "completed"
            local_jobs[job_id]["progress"] = {
                "value": 100,
                "total": 100,
                "description": "Grating created successfully."
            }
            local_jobs[job_id]["result"] = {
                "message": "Grating created successfully",
                "grating": grating_artifact.to_dict(),
                "node_id": grating_artifact.id,
                "id": grating_artifact.id
            }

    except Exception as e:
        print(f"Failed in async grating creation: {e}")
        traceback.print_exc()
        if local_jobs is not None and job_id in local_jobs:
            local_jobs[job_id]["status"] = "failed"
            local_jobs[job_id]["progress"] = None
            local_jobs[job_id]["error"] = str(e)
            local_jobs[job_id]["traceback"] = traceback.format_exc()


async def dispatch_create_grating_operation(
    data: dict[str, Any],
    param_graph: ParameterGraph,
    engine_provider: Any,
    graph_lock: Optional[threading.Lock] = None,
    local_jobs: Optional[dict[str, Any]] = None,
) -> tuple[dict[str, Any], int]:
    """
    Validates and spawns a background grating creation job.
    """
    if param_graph is None or engine_provider is None:
        return {"error": "No project loaded"}, 400

    job_id = data.get("job_id") or f"create_grating_{uuid.uuid4().hex[:12]}"
    if local_jobs is not None:
        local_jobs[job_id] = {
            "status": "pending",
            "progress": {"value": 0, "total": 100, "description": "Starting grating creation..."},
            "result": None,
            "error": None,
            "traceback": None,
            "name": "Create Grating"
        }

    threading.Thread(
        target=lambda: asyncio.run(create_grating_task(
            job_id=job_id,
            data=data,
            param_graph=param_graph,
            engine_provider=engine_provider,
            graph_lock=graph_lock,
            local_jobs=local_jobs,
        )),
        daemon=True
    ).start()

    return {
        "success": True,
        "job_id": job_id,
        "message": "Grating creation job started"
    }, 202


@register
class CreateGratingOperation(Operation):
    @property
    def name(self) -> str:
        return "create_grating"

    @property
    def description(self) -> str:
        return "Dynamically constructs a new Bending Grating and attaches it to a base model."

    @property
    def category(self) -> str:
        return "grating"

    @property
    def execution(self) -> str:
        return "queued"

    @property
    def initiator_types(self) -> list[str]:
        return ["model"]

    def execute_task(self, job_id: str, **kwargs: Any) -> None:
        asyncio.run(create_grating_task(
            job_id=job_id,
            data=kwargs,
            param_graph=kwargs.get("param_graph"),
            engine_provider=kwargs.get("engine_provider"),
            graph_lock=kwargs.get("graph_lock"),
            local_jobs=kwargs.get("local_jobs"),
        ))
