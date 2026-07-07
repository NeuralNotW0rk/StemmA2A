from abc import ABC, abstractmethod
from typing import Any
from param_graph.elements.base_elements import GraphElement

from .model_adapters.stable_audio_adapter import StableAudioAdapter
from .model_adapters.stylegan_adapter import StyleGANAdapter

class Engine(ABC):
    def __init__(self):
        self.adapter_registry = {
            'stable_audio_tools': StableAudioAdapter,
            'stylegan2': StyleGANAdapter
        }

    def _get_adapter_class(self, adapter_name: str):
        adapter_class = self.adapter_registry.get(adapter_name)
        if not adapter_class:
            raise Exception(f"Adapter '{adapter_name}' not found in adapter_registry.")
        return adapter_class

    async def get_adapter_config(self, adapter_name: str) -> dict:
        """Get the form configuration for a specific adapter."""
        adapter_class = self._get_adapter_class(adapter_name)
        # Instantiate the adapter just to get the config
        adapter_instance = adapter_class()
        if not hasattr(adapter_instance, 'get_form_config'):
            raise Exception(f"Adapter '{adapter_name}' does not have a form configuration")

        return adapter_instance.get_form_config()

    async def get_supported_operations(self) -> list[dict]:
        """
        Returns the top-level async operations explicitly supported by the engine.
        The full dynamic forms will be resolved via the adapter once a model is linked.
        """
        import inspect
        supported_ops = {}

        for adapter_name, adapter_class in self.adapter_registry.items():
            for attr_name in dir(adapter_class):
                member = getattr(adapter_class, attr_name)
                if getattr(member, "_is_operation", False):
                    op_name = member._op_name
                    is_standard = member._op_is_standard
                    description = member._op_description
                    initiator_types = member._op_initiator_types
                    context_overrides = member._op_context_overrides

                    if is_standard:
                        if op_name not in supported_ops:
                            supported_ops[op_name] = {
                                "name": op_name,
                                "description": description,
                                "initiator_types": set(initiator_types),
                                "context_overrides": dict(context_overrides),
                                "form_config": [
                                    {"name": "model", "type": "node", "label": "Model", "filter": {"type": "model"}, "required": True}
                                ]
                            }
                            if op_name == "invert":
                                supported_ops[op_name]["form_config"].append(
                                    {"name": "source_audio", "type": "node", "label": "Source Audio", "filter": {"type": "audio"}, "required": True}
                                )
                        else:
                            supported_ops[op_name]["initiator_types"].update(initiator_types)
                            supported_ops[op_name]["context_overrides"].update(context_overrides)
                    else:
                        namespaced_name = f"{op_name}:{adapter_name}"
                        supported_ops[namespaced_name] = {
                            "name": namespaced_name,
                            "description": description,
                            "initiator_types": list(initiator_types),
                            "context_overrides": dict(context_overrides),
                            "form_config": [
                                {"name": "model", "type": "node", "label": "Model", "filter": {"type": "model"}, "required": True}
                            ]
                        }

        for op in supported_ops.values():
            if isinstance(op["initiator_types"], set):
                op["initiator_types"] = list(op["initiator_types"])

        return list(supported_ops.values())

    @abstractmethod
    async def execute(self, operation_id: str, **kwargs) -> str:
        """Queues an operation and returns a job ID."""
        pass

    @abstractmethod
    async def get_job_status(self, job_id: str) -> dict[str, Any]:
        """Gets the status of a job by its ID."""
        pass

    @abstractmethod
    async def cancel_job(self, job_id: str) -> None:
        """Requests cancellation of a job."""
        pass

    @abstractmethod
    async def register_model(self, adapter_name: str, **kwargs) -> GraphElement:
        pass

    @abstractmethod
    async def get_shared_models(self) -> list[dict]:
        pass

    @abstractmethod
    async def get_model_layers(self, model_element: GraphElement) -> list[dict]:
        pass

    @abstractmethod
    async def create_grating(self, model_element: GraphElement, name: str, elements: list) -> GraphElement:
        pass

