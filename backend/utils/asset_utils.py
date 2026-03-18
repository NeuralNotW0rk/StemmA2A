# backend/utils/asset_utils.py
import shutil
from pathlib import Path
from dataclasses import replace
from param_graph.elements.base_elements import GraphElement

def save_artifact_asset(artifact: GraphElement, destination_dir: Path, asset_name: str = "file") -> GraphElement:
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
        raise ValueError(f"Artifact does not have a valid asset at '{asset_name}' with a path to save from.")

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

    # Create a new asset object with the updated path
    updated_asset = replace(asset_to_save, path=str(permanent_path))
    
    # Create a new artifact with the updated asset
    updated_artifact = replace(artifact, **{asset_name: updated_asset})

    # The temporary directory reference is now stale and can be removed.
    # It's attached to the artifact, so we operate on the new instance.
    if hasattr(updated_artifact, '_temp_dir_ref'):
        # This is not perfectly clean, as a new temp dir object will be created
        # for each asset. However, for now, we assume one temp dir per artifact.
        del updated_artifact._temp_dir_ref
        
    return updated_artifact
