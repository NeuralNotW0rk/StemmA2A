# backend/param_graph/utils.py
import shutil
from pathlib import Path
from dataclasses import replace
from typing import Dict, Any, Tuple, List, TYPE_CHECKING

from .elements.base_elements import GraphElement
from .registry import resolve_element

if TYPE_CHECKING:
    from .graph import ParameterGraph


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
    suffix = asset_to_save.extension
    permanent_path = destination_dir / f"{base_name}{suffix}"

    counter = 1
    while permanent_path.exists():
        permanent_path = destination_dir / f"{base_name}_{counter}{suffix}"
        counter += 1

    # Ensure the destination directory exists
    destination_dir.mkdir(parents=True, exist_ok=True)

    # Move the file
    shutil.move(str(temp_path), permanent_path)

    # Create a new asset object with the updated path
    updated_asset = replace(asset_to_save, path=str(permanent_path))

    # Create a new artifact with the updated asset
    updated_artifact = replace(artifact, **{asset_name: updated_asset})

    # The temporary directory reference is now stale and can be removed.
    # It's attached to the artifact, so we operate on the new instance.
    if hasattr(updated_artifact, "_temp_dir_ref"):
        # This is not perfectly clean, as a new temp dir object will be created
        # for each asset. However, for now, we assume one temp dir per artifact.
        del updated_artifact._temp_dir_ref

    return updated_artifact
