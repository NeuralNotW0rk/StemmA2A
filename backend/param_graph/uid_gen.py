import xxhash
from torch import Tensor
from torch.nn import Module


class UIDMismatchError(Exception):
    """Raised when a model's stored UID does not match its calculated UID."""
    pass

class UIDGenerator:
    
    def __init__(self):
        self.type = None
        self.version = None

    def from_string(self, data: str) -> str:
        pass

    def from_tensor(self, tensor: Tensor) -> str:
        pass

    def from_module(self, model: Module) -> str:
        pass


class XXH3_64(UIDGenerator):

    def __init__(self):
        self.type = 'XXH3_64'
        self.version = xxhash.XXHASH_VERSION

    def from_string(self, data: str) -> str:
        """Generates a hex string UID using XXH3."""
        h = xxhash.xxh3_64
        return xxhash.xxh3_64_hexdigest(data)
    
    def from_tensor(self, tensor: Tensor) -> str:
        """
        Returns a deterministic xxh3_64 hex digest of a single PyTorch tensor.
        """
        # 1. Standardize to CPU and ensure memory is contiguous
        # 2. View as a NumPy array (zero-copy)
        # 3. Pass directly to xxhash (handles the buffer protocol)
        data = tensor.detach().cpu().contiguous().numpy()
        
        return xxhash.xxh3_64(data).hexdigest()
    
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
            
        return h.hexdigest()