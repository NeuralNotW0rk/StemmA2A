# backend/asset_cache.py
from pathlib import Path

class AssetCache:
    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def get_path(self, hsh: str, create_parents: bool = False) -> Path:
        """
        Gets the sharded path for a given hash.
        Handles suffixed hashes like 'abcdef.xxh3_64'.
        Example: 'abcdef123.xxh3_64' -> <root>/ab/abcdef123.xxh3_64
        """
        if '.' in hsh:
            hash_val = hsh.rsplit('.', 1)[0]
        else:
            hash_val = hsh

        if len(hash_val) < 2:
            # Keep a flat structure for short hashes
            path = self.root / hsh
        else:
            l1 = hash_val[:2]
            path = self.root / l1 / hsh
        
        if create_parents:
            path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def exists(self, hsh: str) -> bool:
        return self.get_path(hsh).exists()

    def save_file(self, file_storage, hsh: str):
        """Saves a file from a file storage object (like Flask's request.files)."""
        path = self.get_path(hsh, create_parents=True)
        file_storage.save(str(path))

    def get_anchor(self, hsh: str) -> Path:
        """
        This is a bit of a hack. The anchor for a file is its parent directory.
        This is used by the model element to resolve its assets.
        """
        return self.get_path(hsh).parent
