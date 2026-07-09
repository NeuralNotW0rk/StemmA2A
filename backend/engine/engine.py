import os
import tempfile
import uuid
import inspect
import torch
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from param_graph.elements.base_elements import GraphElement, Asset
from param_graph.elements.artifacts.grating_element import Grating as GratingArtifact
from diffracture.topology.grating import Grating as DiffractureGrating
from diffracture.registry import get_element as get_bending_element_cls

from .model_adapters.stable_audio_adapter import StableAudioAdapter
from .model_adapters.stylegan_adapter import StyleGANAdapter
from .model_cache import ModelCache

class Engine(ABC):
    def __init__(self, data_root: str = None):
        self.adapter_registry = {
            'stable_audio_tools': StableAudioAdapter,
            'stylegan2': StyleGANAdapter
        }

        if data_root:
            self.data_root = Path(data_root)
        else:
            self.data_root = Path(tempfile.mkdtemp())

        cache_capacity = int(os.environ.get("MODEL_CACHE_CAPACITY", 1))
        self.model_cache = ModelCache(capacity=cache_capacity)

    def _get_adapter_class(self, adapter_name: str):
        adapter_class = self.adapter_registry.get(adapter_name)
        if not adapter_class:
            raise Exception(f"Adapter '{adapter_name}' not found in adapter_registry.")
        return adapter_class

    def _extract_model_layers(self, model) -> list[dict]:
        """Helper to extract model layers locally from a loaded PyTorch model."""
        import torch
        layers = []
        for name, module in model.named_modules():
            if not name:
                continue
            cls_name = module.__class__.__name__
            if cls_name in ["Linear", "Conv1d", "Conv2d", "Conv3d", "ModulatedConv2d", "StyledConv"] or "Conv" in cls_name or "Linear" in cls_name:
                shape = None
                if hasattr(module, 'weight') and isinstance(module.weight, torch.Tensor):
                    shape = list(module.weight.shape)
                elif hasattr(module, 'conv') and hasattr(module.conv, 'weight') and isinstance(module.conv.weight, torch.Tensor):
                    shape = list(module.conv.weight.shape)
                    
                in_features = getattr(module, 'in_features', getattr(module, 'in_channels', getattr(module, 'in_channel', None)))
                out_features = getattr(module, 'out_features', getattr(module, 'out_channels', getattr(module, 'out_channel', None)))
                
                layers.append({
                    "address": name,
                    "type": cls_name,
                    "shape": shape,
                    "in_features": in_features,
                    "out_features": out_features
                })
        return layers

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
    async def cluster_features(self, model_element: GraphElement, address: str, num_clusters: int) -> list[int]:
        pass

    async def _cluster_features_logic(self, **kwargs) -> list[int]:
        """Logic wrapper used by the worker queue to execute feature clustering."""
        model_element = kwargs["model_element"]
        address = kwargs["address"]
        num_clusters = kwargs["num_clusters"]
        return await self.cluster_features(model_element, address, num_clusters)

    async def create_grating(self, model_element: GraphElement, name: str, elements: list) -> GraphElement:
        grating = DiffractureGrating()
        
        # Build the elements
        for el_data in elements:
            addr = el_data.get("address")
            ktype = el_data.get("kernel_type")
            params = el_data.get("params", {})
            indices = el_data.get("indices") or []
            
            cluster = el_data.get("cluster")
            cluster_map = None
            if el_data.get("perform_clustering"):
                num_clusters = el_data.get("num_clusters", 5)
                # Request atomic feature clustering from the executor (self)
                cluster_map = await self.cluster_features(model_element, addr, num_clusters)
                
            # Lookup element class and instantiate
            el_cls = get_bending_element_cls(ktype)
            metadata = {
                "indices": indices,
                "cluster": cluster,
                "cluster_map": cluster_map,
                **params
            }
            element = el_cls.from_metadata(addr, metadata)
            grating.add_element(element)
            
        # Create output directory for the grating file
        output_dir = self.data_root / "generate"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save Grating safetensors file
        grating_uid = "grating_" + str(uuid.uuid4())[:8]
        file_name = f"{grating_uid}.safetensors"
        grating_path = output_dir / file_name
        grating.save(str(grating_path))
        
        # Add Grating Artifact to project graph
        elements_config = [{
            "address": element.address,
            "kernel_type": element.kernel_type,
            "metadata": element.metadata
        } for element in grating._nodes]
        
        asset = Asset(path=str(grating_path.resolve()), uid=grating_uid, extension=".safetensors")
        grating_artifact = GratingArtifact(
            id=grating_uid,
            name=name,
            context={},
            file=asset,
            base_model_id=model_element.id,
            elements=elements_config
        )
        return grating_artifact

