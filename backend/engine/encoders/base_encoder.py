from abc import ABC, abstractmethod
import torch

class Encoder(ABC):
    @abstractmethod
    def get_embedding(self, audio_path: str) -> torch.Tensor:
        """
        Get the embedding for an audio file.
        """
        pass
