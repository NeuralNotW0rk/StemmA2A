from abc import ABC, abstractmethod
import torch

class Encoder(ABC):
    @abstractmethod
    def get_embedding(self, file_path: str) -> torch.Tensor:
        """
        Get the embedding for a file (audio, image, etc.).
        """
        pass
