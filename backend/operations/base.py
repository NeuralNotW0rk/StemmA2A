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
        """Returns UI form configuration for dynamic argument generation."""
        return []

    @property
    def initiator_types(self) -> list:
        """Supported initiator node types. If empty, it's not restricted."""
        return []

    @property
    def context_overrides(self) -> dict:
        """Contextual naming/description overrides based on initiator type."""
        return {}
        
    def to_dict(self):
        return {
            "name": self.name,
            "description": self.description,
            "form_config": self.get_form_config(),
            "initiator_types": self.initiator_types,
            "context_overrides": self.context_overrides
        }