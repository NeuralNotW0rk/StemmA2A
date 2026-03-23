# backend/param_graph/utils.py
import shutil
from pathlib import Path
from dataclasses import replace
from typing import Dict, Any, Tuple, List, TYPE_CHECKING

from .elements.base_elements import GraphElement
from .registry import resolve_element

if TYPE_CHECKING:
    from .graph import ParameterGraph


def extract_graph_elements(
    form_config: List[Dict[str, Any]],
    params: Dict[str, Any],
    param_graph: "ParameterGraph",
) -> Tuple[Dict[str, GraphElement], List[GraphElement]]:
    """
    Extracts graph elements from a dictionary of parameters based on a form config.

    It finds parameters that correspond to "node" types in the form config,
    retrieves the full GraphElement object from the ParameterGraph, and returns
    them. It also removes the processed node ID from the input `params` dict.

    Args:
        form_config: The form configuration list that defines field types.
        params: A dictionary of parameters, where some values are node IDs.
        param_graph: The ParameterGraph instance to resolve node IDs from.

    Returns:
        A tuple containing:
        - A dictionary of engine arguments for the resolved elements (e.g., {"foo_element": <GraphElement>}).
        - A list of the resolved GraphElement objects.
    """
    engine_args: Dict[str, GraphElement] = {}
    linked_elements: List[GraphElement] = []

    for field_config in form_config:
        if field_config.get("type") == "node":
            field_name = field_config.get("name")
            node_id = params.pop(field_name, None)

            if node_id:
                # Convention: form field 'foo' maps to engine arg 'foo_element'
                arg_name = f"{field_name}_element"
                element = param_graph.get_element(node_id)

                engine_args[arg_name] = element
                linked_elements.append(element)

    return engine_args, linked_elements


def resolve_elements_from_dicts(
    params: Dict[str, Any]
) -> Tuple[Dict[str, Any], Dict[str, GraphElement]]:
    """
    Finds dictionaries that look like serialized GraphElements and resolves them
    into actual GraphElement objects.

    Args:
        params: The dictionary of parameters to process.

    Returns:
        A tuple containing:
        - A new dictionary with the resolved GraphElement objects.
        - A dictionary of the resolved GraphElement objects, keyed by their
          original key in the params dict.
    """
    resolved_params = params.copy()
    all_elements: Dict[str, GraphElement] = {}
    for key, value in params.items():
        # This is a simple check. We might need a more robust way to identify
        # dicts that are meant to be graph elements.
        if isinstance(value, dict) and "id" in value and "type" in value:
            element = resolve_element(value)
            resolved_params[key] = element
            all_elements[key] = element
    return resolved_params, all_elements


def find_elements(d: dict) -> dict[str, GraphElement]:
    """
    Finds existing GraphElement objects within a dictionary.
    """
    elements = {}
    for k, v in d.items():
        if isinstance(v, GraphElement):
            elements[k] = v
        elif isinstance(v, dict):
            # For now, we won't recurse into dicts.
            # This can be expanded if we have nested elements.
            pass
    return elements


def save_artifact_asset(
    artifact: GraphElement, destination_dir: Path, asset_name: str = "file"
) -> GraphElement:
    """
    Moves a specific asset within an artifact from its current temporary
    location to a permanent one.

    Args:
        artifact: The artifact containing the asset to save.
        destination_dir: The directory to save the file in.
        asset_name: The name of the attribute on the artifact that holds the asset.

    Returns:
        A new artifact instance with the path of the specified asset updated.
    """
    asset_to_save = getattr(artifact, asset_name, None)
    if not asset_to_save or not asset_to_save.path:
        raise ValueError(
            f"Artifact does not have a valid asset at '{asset_name}' with a path to save from."
        )

    temp_path = Path(asset_to_save.path)

    # Use artifact's name for a human-readable filename, handling collisions
    base_name = artifact.name
    suffix = temp_path.suffix
    permanent_path = destination_dir / f"{base_name}{suffix}"

    counter = 1
    while permanent_path.exists():
        permanent_path = destination_dir / f"{base_name}_{counter}{suffix}"
        counter += 1

    # Ensure the destination directory exists
    destination_dir.mkdir(parents=True, exist_ok=True)

    # Move the file
    shutil.move(str(temp_path), permanent_path)

    # Make the path relative to the project root for portability
    # The project root is the parent of the destination_dir (e.g., 'generated/')
    project_root = destination_dir.parent
    relative_path = permanent_path.relative_to(project_root)

    # Create a new asset object with the updated path
    updated_asset = replace(asset_to_save, path=str(relative_path))

    # Create a new artifact with the updated asset
    updated_artifact = replace(artifact, **{asset_name: updated_asset})

    # The temporary directory reference is now stale and can be removed.
    # It's attached to the artifact, so we operate on the new instance.
    if hasattr(updated_artifact, "_temp_dir_ref"):
        # This is not perfectly clean, as a new temp dir object will be created
        # for each asset. However, for now, we assume one temp dir per artifact.
        del updated_artifact._temp_dir_ref

    return updated_artifact
