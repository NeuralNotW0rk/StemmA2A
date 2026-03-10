import os
from typing import Optional

from .base import FileProvider
from .local import LocalFileProvider
from .remote import RemoteClientVFS, RemoteServerVFS

def get_vfs_provider() -> FileProvider:
    """
    Factory function to get the appropriate VFS provider based on environment variables.
    """
    vfs_mode = os.getenv("VFS_MODE", "LOCAL").upper()

    if vfs_mode == "LOCAL":
        # Returns a file provider that interacts directly with the local filesystem.
        # The root is determined by an environment variable, defaulting to 'data'.
        return LocalFileProvider(root=os.getenv("PROJECT_ROOT", "data"))

    elif vfs_mode == "REMOTE_CLIENT":
        # Returns a client that communicates with a remote VFS server.
        # Requires the URL of the remote server.
        remote_url = os.getenv("REMOTE_URL")
        if not remote_url:
            raise ValueError("REMOTE_URL environment variable must be set for VFS_MODE=REMOTE_CLIENT")
        return RemoteClientVFS(base_url=remote_url)

    elif vfs_mode == "REMOTE_SERVER":
        # Returns the server-side VFS provider that runs in the remote environment.
        # It uses a logical root path within its container.
        logical_root = os.getenv("PROJECT_ROOT", "/app/data")
        return RemoteServerVFS(logical_root=logical_root)

    else:
        raise ValueError(f"Invalid VFS_MODE: {vfs_mode}")
