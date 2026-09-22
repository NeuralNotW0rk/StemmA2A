"""Host-level evolutionary operations registry and definitions."""

import json
import os

def get_evolution_operations() -> list[dict]:
    """Returns the list of host-level evolutionary operations loaded from operations.json."""
    json_path = os.path.join(os.path.dirname(__file__), "operations.json")
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Evolution operations config not found at: {json_path}")
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)
