# backend/pydantic_util.py
from pydantic import create_model, BaseModel
from typing import List, Dict, Any, Type

def create_dynamic_model(config: List[Dict[str, Any]]) -> Type[BaseModel]:
    """Dynamically creates a Pydantic model from a form configuration."""
    field_definitions = {}
    type_mapping = {
        "textarea": str,
        "text": str,
        "integer": int,
        "float": float,
        "boolean": bool,
        "select": str,
        "file": str, # Assuming file path is passed as string
        "node": (Dict, None),
    }

    for item in config:
        name = item.get("name")
        if not name:
            continue
            
        field_type = type_mapping.get(item.get("type"), str)
        
        # Pydantic uses '...' (Ellipsis) to mark a field as required
        if item.get("required"):
            field_definitions[name] = (field_type, ...)
        else:
            default_value = item.get("defaultValue")
            field_definitions[name] = (field_type, default_value)

    DynamicModel = create_model(f'DynamicGenerateModel', **field_definitions)
    return DynamicModel
