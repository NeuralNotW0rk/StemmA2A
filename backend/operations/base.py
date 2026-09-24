"""Unified base Operation abstraction for host-level immediate and queued operations."""

import inspect
import json
import os
from abc import ABC, abstractmethod
from typing import Any, Optional
from param_graph.elements.base_elements import GraphElement


class Operation(ABC):
    """
    Unified Base Operation.
    Defines metadata, dynamic form configs, and execution interfaces
    for all host operations across DSP, Evolution, IO, and Utility tasks.
    """
    @property
    @abstractmethod
    def name(self) -> str:
        """Unique operation identifier (e.g. 'gain', 'slice', 'mutate', 'recombine')."""
        pass

    @property
    def description(self) -> str:
        """Human-readable description of the operation."""
        return ""

    @property
    def category(self) -> str:
        """Operation category: 'dsp', 'evolution', 'io', 'utility', 'generative'."""
        return "dsp"

    @property
    def execution(self) -> str:
        """Execution mode: 'immediate' for synchronous in-band execution, 'queued' for background tasks."""
        return "immediate"

    @property
    def execution_mode(self) -> str:
        """Backward-compatibility alias for execution ('sync' for immediate, 'async' for queued)."""
        return "sync" if self.execution == "immediate" else "async"

    @property
    def initiator_types(self) -> list[str]:
        """Supported initiator node types. If empty, the operation is unrestricted."""
        return []

    @property
    def context_overrides(self) -> dict[str, Any]:
        """Metadata context overrides injected into downstream artifacts."""
        return {}

    def get_form_config(self) -> list[dict[str, Any]]:
        """
        Returns UI dynamic form configuration schema (list of field definitions).
        By default, attempts to load colocated <name>.json or <module_name>.json.
        """
        try:
            cls_file = inspect.getfile(self.__class__)
            dir_name = os.path.dirname(cls_file)
            
            candidates = [
                os.path.join(dir_name, f"{self.name}.json"),
                os.path.join(dir_name, f"{os.path.splitext(os.path.basename(cls_file))[0]}.json"),
            ]
            
            for candidate in candidates:
                if os.path.exists(candidate):
                    with open(candidate, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        if isinstance(data, list):
                            return data
                        elif isinstance(data, dict):
                            return data.get("form_config", data.get("fields", []))
        except Exception as e:
            print(f"Error loading form config for operation '{self.name}': {e}")
        return []

    def execute(self, **kwargs: Any) -> list[tuple[GraphElement, Any]] | dict[str, Any]:
        """
        Executes an immediate (synchronous) operation in-band.
        For DSP operations, returns a list of tuples containing (Artifact, raw_data).
        For graph utility operations, returns a dictionary response.
        """
        raise NotImplementedError(f"Operation '{self.name}' does not implement synchronous execute().")

    def execute_task(self, job_id: str, **kwargs: Any) -> Any:
        """
        Executes a queued background task, reporting incremental progress via TaskManager.
        """
        raise NotImplementedError(f"Operation '{self.name}' does not implement background execute_task().")

    def to_dict(self) -> dict[str, Any]:
        """Serializes the operation metadata into a dictionary for API/UI discovery."""
        return {
            "name": self.name,
            "description": self.description,
            "category": self.category,
            "execution": self.execution,
            "execution_mode": self.execution_mode,
            "form_config": self.get_form_config(),
            "initiator_types": self.initiator_types,
            "context_overrides": self.context_overrides,
        }


# Backward-compatibility alias
SyncOperation = Operation