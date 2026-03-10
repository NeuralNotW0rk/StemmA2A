import os
import requests
from pathlib import Path
from typing import List, Dict, Any, Optional

from .base import FileProvider

# --- Remote Client: For Laptop UI to call PC API ---

class RemoteClientVFS(FileProvider):
    """
    File provider for accessing a remote filesystem via an API.
    This is intended to run on the 'client' (e.g., a laptop), making
    requests to the 'server' (e.g., a powerful PC with a GPU).
    """

    def __init__(self, base_url: str, api_key: Optional[str] = None):
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        if api_key:
            self.session.headers.update({"Authorization": f"Bearer {api_key}"})

    def list_projects(self) -> List[str]:
        response = self.session.get(f"{self.base_url}/vfs/projects")
        response.raise_for_status()
        return response.json()

    def list_models(self, project_id: str) -> List[str]:
        response = self.session.get(f"{self.base_url}/vfs/models?project_id={project_id}")
        response.raise_for_status()
        return response.json()

    def get_file_metadata(self, path: str) -> Optional[Dict[str, Any]]:
        response = self.session.get(f"{self.base_url}/vfs/metadata?path={path}")
        if response.status_code == 404:
            return None
        response.raise_for_status()
        return response.json()

    def exists(self, path: str) -> bool:
        response = self.session.get(f"{self.base_url}/vfs/exists?path={path}")
        response.raise_for_status()
        return response.json()

    def get_project_path(self, project_name: str) -> str:
        response = self.session.get(f"{self.base_url}/vfs/project-path?project_name={project_name}")
        response.raise_for_status()
        return response.json()["path"]

    def get_model_path(self, model_name: str) -> str:
        response = self.session.get(f"{self.base_url}/vfs/model-path?model_name={model_name}")
        response.raise_for_status()
        return response.json()["path"]

    def resolve_to_physical_path(self, logical_path: str) -> str:
        response = self.session.get(f"{self.base_url}/api/vfs/resolve-path?path={logical_path}")
        response.raise_for_status()
        return response.json()["path"]


# --- Remote Server: For code running in Docker on the PC ---

class RemoteServerVFS(FileProvider):
    """
    File provider for the server-side component that runs inside a
    Docker container and interacts with the actual filesystem.
    It ensures that all file access is sandboxed to a specific root directory.
    """
    def __init__(self, logical_root: str = "/app/data"):
        self.logical_root = Path(logical_root).resolve()
        self.projects_dir = self.logical_root / "projects"
        self.models_dir = self.logical_root / "models"

        self.projects_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)


    def _sanitize_and_resolve(self, user_path: str) -> Path:
        """
        Prevents directory traversal and resolves a user-provided path
        against the logical root.
        """
        # Create a Path object. This also helps sanitize input slightly (e.g. null bytes)
        path = Path(user_path)

        # Disallow absolute paths. `is_absolute()` is the right check.
        # On Windows, C:foo is not absolute, but we don't want it. `anchor` checks for `C:`.
        if path.is_absolute() or path.anchor:
            raise PermissionError("Absolute paths and anchors are not allowed.")

        # Join the user path to the root and resolve it to get the canonical path
        # (e.g., removing '.', '..', and following symlinks).
        full_path = (self.logical_root / path).resolve()

        # Ensure the resolved path is still within the logical root directory.
        # `self.logical_root` is already resolved in __init__.
        if not str(full_path).startswith(str(self.logical_root)):
            raise PermissionError("Directory traversal detected.")

        return full_path


    def list_projects(self) -> List[str]:
        if not self.projects_dir.is_dir():
            return []
        # Return clean, display-friendly names
        return [d.name for d in self.projects_dir.iterdir() if d.is_dir()]

    def list_models(self, project_id: Optional[str] = None) -> List[str]:
        """
        Lists available models. If a project_id is provided, it includes
        models from that project's 'models' directory, in addition to
        the global models.
        """
        model_names = set()

        # Add global models
        if self.models_dir.is_dir():
            for item in self.models_dir.iterdir():
                if item.is_dir():
                    model_names.add(item.name)

        # Add project-specific models
        if project_id:
            try:
                project_path = self._sanitize_and_resolve(f"projects/{project_id}")
                project_models_dir = project_path / "models"
                if project_models_dir.is_dir():
                    for item in project_models_dir.iterdir():
                        if item.is_dir():
                            model_names.add(item.name)
            except PermissionError:
                pass  # Ignore invalid project_id

        return sorted(list(model_names))

    def get_file_metadata(self, path: str) -> Optional[Dict[str, Any]]:
        try:
            resolved_path = self._sanitize_and_resolve(path)
            if not resolved_path.exists():
                return None
            
            stat = resolved_path.stat()
            return {
                "name": resolved_path.name,
                "path": str(resolved_path.relative_to(self.logical_root).as_posix()),
                "is_dir": resolved_path.is_dir(),
                "size": stat.st_size,
                "created_time": stat.st_ctime,
                "modified_time": stat.st_mtime,
            }
        except PermissionError:
            return None


    def exists(self, path: str) -> bool:
        try:
            return self._sanitize_and_resolve(path).exists()
        except PermissionError:
            return False

    def get_project_path(self, project_name: str) -> str:
        # Return a logical, virtual path, not a physical one
        return (Path("projects") / project_name).as_posix()

    def get_model_path(self, model_name: str, project_id: Optional[str] = None) -> str:
        """
        Returns the logical path for a model.
        It prefers project-specific models if a project_id is provided.
        """
        if project_id:
            # Check if the project-specific model exists before returning its path
            try:
                project_model_path = self._sanitize_and_resolve(
                    f"projects/{project_id}/models/{model_name}"
                )
                if project_model_path.exists():
                    return (Path("projects") / project_id / "models" / model_name).as_posix()
            except PermissionError:
                pass # Fallback to global
        # Return the logical path to the global model
        return (Path("models") / model_name).as_posix()

    def resolve_to_physical_path(self, logical_path: str) -> str:
        """
        Resolves a logical path to a physical path on the server's filesystem.
        """
        return str(self._sanitize_and_resolve(logical_path))
