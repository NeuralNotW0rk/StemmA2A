import shutil
from pathlib import Path

def export_project(project_path: str, export_dir: str):
    """
    Copies a project directory to a specified export directory.
    """
    project_path = Path(project_path)
    export_dir = Path(export_dir)

    if not project_path.is_dir():
        raise FileNotFoundError(f"Project path '{project_path}' does not exist or is not a directory.")

    # Create a destination directory inside the export_dir
    destination = export_dir / project_path.name
    
    if destination.exists():
        # Clear out the destination directory if it already exists
        shutil.rmtree(destination)
    
    # Copy the entire directory tree
    shutil.copytree(project_path, destination)
    
    return str(destination)
