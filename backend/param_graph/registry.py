# backend/param_graph/registry.py
from typing import Dict, Type, Any
import inspect


# The global registry mapping keys to dataclass types
_DATACLASS_REGISTRY: Dict[str, Type[Any]] = {}

def register(key: str):
    """A decorator to register a dataclass with a given key."""
    def decorator(cls: Type[Any]):
        print(f"DEBUG: Registering '{cls.__name__}' with key '{key}'")
        if key in _DATACLASS_REGISTRY:
            # This can help catch copy/paste errors during development
            raise ValueError(f"Key '{key}' is already registered.")
        _DATACLASS_REGISTRY[key] = cls
        return cls
    return decorator

def get_class(key: str) -> Type[Any]:
    """Looks up a dataclass in the registry by its key."""
    cls = _DATACLASS_REGISTRY.get(key)
    if cls is None:
        raise ValueError(f"No dataclass registered for key '{key}'. Available keys: {list(_DATACLASS_REGISTRY.keys())}")
    return cls

def get_key_from_attrs(attrs: Dict[str, Any]) -> str:
    """Constructs the registry key from a node's attributes."""
    node_type = attrs.get('type')
    if not node_type:
        raise ValueError("Node attributes must include a 'type' field to be resolved.")

    if node_type == 'model':
        # For models, the key is a composite: "type:adapter"
        adapter = attrs.get('adapter')
        if not adapter:
            raise ValueError("Model attributes must include an 'adapter' field.")
        return f"{node_type}:{adapter}"
    else:
        # For all other elements, the 'type' field is the key
        return node_type
    
def resolve_element(attrs: Dict[str, Any]) -> Type[Any]:
    # Determine the correct class and instantiate it
    key = get_key_from_attrs(attrs)
    TargetClass = get_class(key)
    
    # Filter attrs to only include keys that are in the TargetClass's __init__
    # This is necessary because the graph node attributes may contain extra
    # data that is not part of the class's definition.
    sig = inspect.signature(TargetClass)
    allowed_keys = set(sig.parameters.keys())
    filtered_attrs = {k: v for k, v in attrs.items() if k in allowed_keys}

    return TargetClass(**filtered_attrs)

