import inspect
import json
import os
from abc import ABC, abstractmethod
from typing import Any, List, Tuple
from param_graph.elements.base_elements import GraphElement

# Base class for all Synchronous Operations
class SyncOperation(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        pass
        
    @property
    def description(self) -> str:
        return ""

    @abstractmethod
    def execute(self, **kwargs) -> List[Tuple[GraphElement, Any]]:
        """
        Executes the operation.
        Returns a list of tuples containing the Artifact blueprint and the raw data.
        """
        pass

    def get_form_config(self) -> list:
        """Returns UI form configuration for dynamic argument generation.

        By default, attempts to load <name>.json or <module_name>.json colocated
        with the operation subclass file.
        """
        try:
            cls_file = inspect.getfile(self.__class__)
            dir_name = os.path.dirname(cls_file)
            name_json_path = os.path.join(dir_name, f"{self.name}.json")
            if os.path.exists(name_json_path):
                with open(name_json_path, "r", encoding="utf-8") as f:
                    return json.load(f)

            module_base = os.path.splitext(os.path.basename(cls_file))[0]
            module_json_path = os.path.join(dir_name, f"{module_base}.json")
            if os.path.exists(module_json_path):
                with open(module_json_path, "r", encoding="utf-8") as f:
                    return json.load(f)
        except Exception as e:
            print(f"Error loading form config for operation '{self.name}': {e}")
        return []

    @property
    def initiator_types(self) -> list:
        """Supported initiator node types. If empty, it's not restricted."""
        return []

    @property
    def category(self) -> str:
        return "dsp"

    @property
    def execution(self) -> str:
        return "immediate"

    @property
    def context_overrides(self) -> dict:
        return {}

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "category": self.category,
            "execution": self.execution,
            "execution_mode": "sync",  # backward compatibility alias
            "form_config": self.get_form_config(),
            "initiator_types": self.initiator_types,
            "context_overrides": self.context_overrides
        }