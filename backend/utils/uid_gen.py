import json
import xxhash
from torch import Tensor
from torch.nn import Module
from abc import ABC, abstractmethod


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
    def from_dict(self, data: dict) -> str:
        pass

    @abstractmethod
    def from_hashes(self, hashes: list) -> str:
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
        Generates a deterministic xxh3_64 hash for a PyTorch model's weights.
        """
        h = xxhash.xxh3_64()
        state_dict = module.state_dict()
        
        # Sorting keys ensures the hash is identical regardless of internal dict order
        for key in sorted(state_dict.keys()):
            # .contiguous() is the 'secret sauce'—it ensures the memory layout 
            # matches the logical data, regardless of previous slices or transposes.
            data = state_dict[key].detach().cpu().contiguous().numpy().tobytes()
            h.update(data)
            
        return f"{h.hexdigest()}{self.DELIMITER}{self.get_method_name()}"

    def from_dict(self, data: dict) -> str:
        """
        Generates a deterministic xxh3_64 hash for a dictionary.
        """
        # Serialize the dictionary to a JSON string with sorted keys
        serialized_data = json.dumps(data, sort_keys=True)
        return self.from_string(serialized_data)
    
    def from_hashes(self, hashes: list) -> str:
        """
        Generates a deterministic xxh3_64 hash from a list of hashes.
        """
        # Sort the hashes to ensure determinism
        hashes.sort()
        # Concatenate the sorted hashes
        concatenated_hashes = "".join(hashes)
        # Hash the concatenated string
        return self.from_string(concatenated_hashes)