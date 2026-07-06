from typing import Dict, List, Type
from .base import SyncOperation

# Global registry mapping names to operation instances
_SYNC_OPERATIONS_REGISTRY: Dict[str, SyncOperation] = {}

def register(op_class: Type[SyncOperation]):
    """Decorator to automatically instantiate and register a SyncOperation."""
    instance = op_class()
    name = instance.name
    print(f"DEBUG: Registering sync operation '{name}' ({op_class.__name__})")
    if name in _SYNC_OPERATIONS_REGISTRY:
        raise ValueError(f"Operation name '{name}' is already registered.")
    _SYNC_OPERATIONS_REGISTRY[name] = instance
    return op_class

class SyncRegistry:
    def __init__(self):
        self._operations: Dict[str, SyncOperation] = _SYNC_OPERATIONS_REGISTRY

    def register(self, operation: SyncOperation):
        self._operations[operation.name] = operation

    def get(self, name: str) -> SyncOperation:
        if name not in self._operations:
            raise ValueError(f"Sync operation '{name}' not found in registry. Available operations: {list(self._operations.keys())}")
        return self._operations[name]
        
    def get_all(self) -> List[SyncOperation]:
        return list(self._operations.values())