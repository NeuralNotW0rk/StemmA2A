import json
import xxhash
from torch import Tensor
from torch.nn import Module
from abc import ABC, abstractmethod
from pathlib import Path


def path_from_uid(uid: str, depth: int = 1) -> Path:
    """Shards a UID into N levels of directories."""
    parts = [uid[i:i+2] for i in range(0, depth * 2, 2)]
    return Path("cache", *parts, uid)




class UIDMismatchError(Exception):
    """Raised when a model's stored UID does not match its calculated UID."""
    pass

class UIDGenerator(ABC):
    DELIMITER = "."

    @abstractmethod
    def get_method_name(self) -> str:
        pass
    
    @abstractmethod
    def from_string(self, data: str) -> str:
        pass

    @abstractmethod
    def from_tensor(self, tensor: Tensor) -> str:
        pass

    @abstractmethod
    def from_module(self, model: Module) -> str:
        pass

    @abstractmethod
    def from_state_dict(self, state_dict: dict) -> str:
        pass

    @abstractmethod
    def from_dict(self, data: dict) -> str:
        pass

    @abstractmethod
    def from_uids(self, uids: list) -> str:
        pass

    @abstractmethod
    def from_directory(self, directory_path: Path) -> str:
        pass


class XXH3_64(UIDGenerator):

    def get_method_name(self):
        return "xxh3_64"
    
    def from_string(self, data: str) -> str:
        """Generates a hex string UID using XXH3."""
        h = xxhash.xxh3_64
        return f"{xxhash.xxh3_64_hexdigest(data)}{self.DELIMITER}{self.get_method_name()}"
    
    def from_tensor(self, tensor: Tensor) -> str:
        """
        Returns a deterministic xxh3_64 hex digest of a single PyTorch tensor.
        """
        # 1. Standardize to CPU and ensure memory is contiguous
        # 2. View as a NumPy array (zero-copy)
        # 3. Pass directly to xxhash (handles the buffer protocol)
        data = tensor.detach().cpu().contiguous().numpy()
        
        return f"{xxhash.xxh3_64(data).hexdigest()}{self.DELIMITER}{self.get_method_name()}"
    
    def from_module(self, module: Module) -> str:
        """
        Generates a deterministic xxh3_64 UID for a PyTorch model's weights.
        """
        return self.from_state_dict(module.state_dict())

    def from_state_dict(self, state_dict: dict) -> str:
        """
        Generates a deterministic xxh3_64 UID for a model's state dict weights.
        """
        h = xxhash.xxh3_64()
        
        # Sorting keys ensures the UID is identical regardless of internal dict order
        for key in sorted(state_dict.keys()):
            # .contiguous() is the 'secret sauce'—it ensures the memory layout 
            # matches the logical data, regardless of previous slices or transposes.
            tensor = state_dict[key].detach().cpu()
            if not tensor.is_contiguous():
                tensor = tensor.contiguous()
            h.update(memoryview(tensor.numpy()))
            
        return f"{h.hexdigest()}{self.DELIMITER}{self.get_method_name()}"

    def from_dict(self, data: dict) -> str:
        """
        Generates a deterministic xxh3_64 UID for a dictionary.
        """
        # Serialize the dictionary to a JSON string with sorted keys
        serialized_data = json.dumps(data, sort_keys=True)
        return self.from_string(serialized_data)
    
    def from_uids(self, uids: list) -> str:
        """
        Generates a deterministic xxh3_64 UID from a list of UIDs.
        """
        # Sort the UIDs to ensure determinism
        uids.sort()
        # Concatenate the sorted UIDs
        concatenated_uids = "".join(uids)
        # Hash the concatenated string
        return self.from_string(concatenated_uids)

    def from_directory(self, directory_path: Path) -> str:
        """
        Generates a deterministic xxh3_64 UID for a directory asset.
        Normalizes POSIX relative paths, filters OS metadata, and normalizes text/JSON files.
        """
        h = xxhash.xxh3_64()
        ignored_names = {".DS_Store", "Thumbs.db", "desktop.ini", "__pycache__", ".git", ".gitignore"}

        # Find all files recursively and sort by POSIX relative path to ensure determinism across OSs
        file_paths: list[Path] = []
        for p in directory_path.glob("**/*"):
            if p.is_file() and not any(part in ignored_names for part in p.parts):
                file_paths.append(p)
        file_paths.sort(key=lambda p: p.relative_to(directory_path).as_posix())

        for file_path in file_paths:
            # 1. Update hash with normalized POSIX relative path
            rel_posix_path = file_path.relative_to(directory_path).as_posix()
            h.update(rel_posix_path.encode("utf-8"))

            # 2. Update hash with content
            suffix = file_path.suffix.lower()
            if suffix == ".json":
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    canonical_json = json.dumps(data, sort_keys=True, separators=(",", ":"))
                    h.update(canonical_json.encode("utf-8"))
                except Exception:
                    with open(file_path, "rb") as f:
                        while chunk := f.read(8192):
                            h.update(chunk)
            elif suffix in {".txt", ".yaml", ".yml", ".md", ".csv"}:
                try:
                    with open(file_path, "r", encoding="utf-8", newline="") as f:
                        text = f.read()
                    normalized_text = text.replace("\r\n", "\n")
                    h.update(normalized_text.encode("utf-8"))
                except Exception:
                    with open(file_path, "rb") as f:
                        while chunk := f.read(8192):
                            h.update(chunk)
            else:
                with open(file_path, "rb") as f:
                    while chunk := f.read(8192):
                        h.update(chunk)

        return f"{h.hexdigest()}{self.DELIMITER}{self.get_method_name()}"