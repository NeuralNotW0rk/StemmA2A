from typing import Dict, Type, Any, get_type_hints
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

    # Get constructor signature and type hints for the target class
    sig = inspect.signature(TargetClass)
    try:
        type_hints = get_type_hints(TargetClass)
    except (NameError, TypeError):
        type_hints = {} # Fallback if hints can't be resolved

    constructor_attrs = {}
    for param_name in sig.parameters:
        if param_name in attrs:
            value = attrs[param_name]
            param_type = type_hints.get(param_name)

            # If the parameter is typed as a class and the value is a dictionary,
            # we attempt to instantiate the class with the dictionary's values.
            # This handles nested objects like the 'Asset' dataclass.
            if inspect.isclass(param_type) and isinstance(value, dict):
                try:
                    # This assumes the nested dict keys match the nested class's __init__ args
                    constructor_attrs[param_name] = param_type(**value)
                except TypeError:
                    # If instantiation fails, we pass the dictionary as is.
                    constructor_attrs[param_name] = value
            else:
                constructor_attrs[param_name] = value

    return TargetClass(**constructor_attrs)
