"""Unified Operation Registry for immediate and queued host operations."""

import inspect
from typing import Optional, Union, Type
from .base import Operation, SyncOperation


class OperationRegistry:
    """Singleton registry tracking all available host operations."""
    _instance: Optional["OperationRegistry"] = None

    def __new__(cls) -> "OperationRegistry":
        if cls._instance is None:
            cls._instance = super(OperationRegistry, cls).__new__(cls)
            cls._instance._operations = {}
        return cls._instance

    def register(self, operation: Union[Type[Operation], Operation]) -> Union[Type[Operation], Operation]:
        """Registers an operation class or instance and returns the original class/instance."""
        if inspect.isclass(operation):
            op_instance = operation()
            self._operations[op_instance.name] = op_instance
            return operation
        else:
            self._operations[operation.name] = operation
            return operation

    def get(self, name: str) -> Optional[Operation]:
        """Retrieves an operation instance by its unique name."""
        return self._operations.get(name)

    def has(self, name: str) -> bool:
        """Returns True if an operation with the given name is registered."""
        return name in self._operations

    def get_all(self) -> list[Operation]:
        """Returns all registered operations."""
        return list(self._operations.values())

    def get_by_category(self, category: str) -> list[Operation]:
        """Returns all registered operations belonging to a specific category."""
        return [op for op in self._operations.values() if op.category == category]


# Global singleton and decorator
operation_registry = OperationRegistry()
register = operation_registry.register

# Backward-compatibility alias
SyncRegistry = OperationRegistry
sync_registry = operation_registry