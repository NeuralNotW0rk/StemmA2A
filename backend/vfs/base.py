from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

class FileProvider(ABC):
    """
    Abstract base class for a file provider, defining an interface
    for accessing files and directories.
    """

    @abstractmethod
    def list_projects(self) -> List[str]:
        """
        Returns a list of project IDs.
        """
        pass

    @abstractmethod
    def list_models(self, project_id: str) -> List[str]:
        """
        Returns a list of model IDs for a given project.
        """
        pass

    @abstractmethod
    def get_file_metadata(self, path: str) -> Optional[Dict[str, Any]]:
        """
        Returns metadata for a file or directory.
        """
        pass

    @abstractmethod
    def exists(self, path: str) -> bool:
        """
        Returns True if a file or directory exists at the given path.
        """
        pass

    @abstractmethod
    def get_project_path(self, project_name: str) -> str:
        """
        Returns the full path for a project.
        """
        pass

    @abstractmethod
    def get_model_path(self, model_name: str) -> str:
        """
        Returns the full path for a model.
        """
        pass

    @abstractmethod
    def resolve_to_physical_path(self, logical_path: str) -> str:
        """
        Resolves a logical path or ID to a physical path on the filesystem.
        """
        pass

    @abstractmethod
    def export_model_package(self, model_data: Dict[str, Any]) -> None:
        """
        Exports a model and its dependencies to a standardized package structure.
        """
        pass

