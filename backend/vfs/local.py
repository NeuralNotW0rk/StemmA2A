import os
from pathlib import Path
from typing import List, Dict, Any, Optional

from .base import FileProvider

class LocalFileProvider(FileProvider):
    """
    File provider for accessing the local filesystem, sandboxed to a root directory.
    """

    def __init__(self, root: str = "stemma2a_data"):
        """
        Initializes the provider with a specific root directory.
        Defaults to 'stemma2a_data' in the current working directory.
        """
        self.root = Path(root).resolve()
        self.projects_dir = self.root / "projects"
        self.models_dir = self.root / "models"

        self.projects_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)

    def list_projects(self) -> List[str]:
        """
        Returns a list of project IDs from the local filesystem.
        """
        if not self.projects_dir.is_dir():
            return []
        return [d.name for d in self.projects_dir.iterdir() if d.is_dir()]

    def list_models(self, project_id: Optional[str] = None) -> List[str]:
        """
        Returns a list of model IDs. If project_id is provided, it includes
        project-specific models.
        """
        all_models = set()
        # Global models
        if self.models_dir.is_dir():
            all_models.update(d.name for d in self.models_dir.iterdir() if d.is_dir())

        # Project-specific models
        if project_id:
            project_models_dir = self.projects_dir / project_id / "models"
            if project_models_dir.is_dir():
                all_models.update(d.name for d in project_models_dir.iterdir() if d.is_dir())
        
        return sorted(list(all_models))

    def get_file_metadata(self, path: str) -> Optional[Dict[str, Any]]:
        """
        Returns metadata for a file or directory on the local filesystem.
        """
        p = Path(path).resolve() # Resolve to handle relative paths
        
        # Security check: ensure path is within the root, projects, or models directory
        if not (str(p).startswith(str(self.root))):
             raise PermissionError("Access to path is denied.")

        if not p.exists():
            return None
        
        stat = p.stat()
        return {
            "name": p.name,
            "path": str(p),
            "is_dir": p.is_dir(),
            "size": stat.st_size,
            "created_time": stat.st_ctime,
            "modified_time": stat.st_mtime,
        }

    def exists(self, path: str) -> bool:
        """
        Returns True if a file or directory exists at the given path on the local filesystem.
        """
        try:
            p = Path(path).resolve()
            if not (str(p).startswith(str(self.root))):
                return False
            return p.exists()
        except Exception:
            return False
    
    def get_project_path(self, project_name: str) -> str:
        """
        Returns the full path for a project.
        """
        return str(self.projects_dir / project_name)

    def get_model_path(self, model_name: str, project_id: Optional[str] = None) -> str:
        """
        Returns the full path for a model.
        It prefers project-specific models if a project_id is provided.
        """
        if project_id:
            project_model_path = self.projects_dir / project_id / "models" / model_name
            if project_model_path.exists():
                return str(project_model_path)
        return str(self.models_dir / model_name)

    def resolve_to_physical_path(self, logical_path: str) -> str:
        """
        For the local provider, the logical path IS the physical path.
        This method validates that the path exists and is readable.
        """
        p = Path(logical_path).resolve()

        if not p.exists():
            raise FileNotFoundError(f"The path '{logical_path}' does not exist.")

        # Check for read access at the OS level.
        if not os.access(str(p), os.R_OK):
            raise PermissionError(f"Read access to path '{logical_path}' is denied by the filesystem.")

        return str(p)

    def export_model_package(self, model_data: Dict[str, Any]) -> None:
        """
        Copies model files to a local '_exports' directory for staging.
        """
        import shutil

        model_name = model_data.get("name")
        if not model_name:
            raise ValueError("Model name is required for export.")

        # Create export directory in the project root
        export_dir = Path.cwd() / "_exports" / model_name
        export_dir.mkdir(parents=True, exist_ok=True)

        # Copy and rename files
        config_src = model_data.get("config_path")
        checkpoint_src = model_data.get("checkpoint_path")

        if config_src and Path(config_src).exists():
            shutil.copy(config_src, export_dir / "model_config.json")
            print(f"Copied config to {export_dir / 'model_config.json'}")

        if checkpoint_src and Path(checkpoint_src).exists():
            shutil.copy(checkpoint_src, export_dir / "model.ckpt")
            print(f"Copied checkpoint to {export_dir / 'model.ckpt'}")

