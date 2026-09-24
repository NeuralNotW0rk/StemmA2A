"""Operations package coordinating immediate DSP, evolutionary, and host background tasks."""

import pkgutil
import importlib

# Recursively walk all packages and modules from this directory
# and import them. This ensures all @register decorators are run.
for loader, module_name, is_pkg in pkgutil.walk_packages(__path__, prefix=f'{__package__}.'):
    if '__init__' not in module_name:
        try:
            importlib.import_module(module_name)
        except Exception as e:
            pass

from .base import Operation, SyncOperation
from .registry import OperationRegistry, SyncRegistry, operation_registry, sync_registry, register
from .task_manager import TaskManager, task_manager
from .dispatcher import dispatch_operation

__all__ = [
    "Operation",
    "SyncOperation",
    "OperationRegistry",
    "SyncRegistry",
    "operation_registry",
    "sync_registry",
    "register",
    "TaskManager",
    "task_manager",
    "dispatch_operation",
]