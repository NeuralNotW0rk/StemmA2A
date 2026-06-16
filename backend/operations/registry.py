from typing import Dict, List
from .base import SyncOperation
from .core import GainOperation, NormalizeOperation, SliceOperation, LibrosaOnsetSliceOperation

class SyncRegistry:
    def __init__(self):
        self._operations: Dict[str, SyncOperation] = {}
        self._register_defaults()

    def _register_defaults(self):
        self.register(GainOperation())
        self.register(NormalizeOperation())
        self.register(SliceOperation())
        self.register(LibrosaOnsetSliceOperation())

    def register(self, operation: SyncOperation):
        self._operations[operation.name] = operation

    def get(self, name: str) -> SyncOperation:
        if name not in self._operations:
            raise ValueError(f"Sync operation '{name}' not found in registry.")
        return self._operations[name]
        
    def get_all(self) -> List[SyncOperation]:
        return list(self._operations.values())