# backend/utils/filesystem.py
import os
from pathlib import Path

def check_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir, exist_ok=True)
    return dir

def remove_path(path: Path) -> None:
    """Deletes a file or a directory tree, if there is one."""
    import shutil
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink(missing_ok=True)

def get_path_size(path: Path) -> int:
    """
    Returns the total size in bytes for a file or a directory (sum of all file sizes recursively).
    """
    if path.is_file():
        return path.stat().st_size
    elif path.is_dir():
        total_size = 0
        for p in path.glob("**/*"):
            if p.is_file():
                total_size += p.stat().st_size
        return total_size
    else:
        raise FileNotFoundError(f"Path not found: {path}")
