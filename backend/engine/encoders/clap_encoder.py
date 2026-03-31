import torch
import laion_clap
import laion_clap.hook # For monkey patching
from .base_encoder import Encoder

def patched_load_state_dict(checkpoint_path: str, map_location="cpu", skip_params=True):
    """
    A patched version of load_state_dict that sets weights_only=False to address
    an issue with loading checkpoints in newer PyTorch versions.
    """
    checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint
    if skip_params:
        if next(iter(state_dict.items()))[0].startswith("module"):
            state_dict = {k[7:]: v for k, v in state_dict.items()}
    return state_dict

class CLAPEncoder(Encoder):
    """
    A CLAP encoder for generating audio embeddings using the laion-clap model.

    This class provides a simple interface to load a pre-trained CLAP model
    and use it to encode audio tensors into semantic embeddings. It handles
    audio preprocessing steps such as resampling and channel conversion.
    """
    def __init__(self):
        """
        Initializes the CLAPEncoder.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = None
        self.target_sr = 48000

    def load_model(self):
        """
        Loads the pre-trained CLAP model. This method temporarily patches
        the model loading functionality to allow older checkpoints to be loaded.
        """
        if self._model is None:
            print("Loading CLAP model...")

            original_load_state_dict = laion_clap.hook.load_state_dict
            laion_clap.hook.load_state_dict = patched_load_state_dict
            
            try:
                # Initialize the CLAP model. 'HTSAT-base' is a good general-purpose model.
                self._model = laion_clap.CLAP_Module(enable_fusion=False, amodel='HTSAT-tiny')
                
                # Load the pre-trained checkpoint. This will download the model if not cached.
                self._model.load_ckpt()
                
                self._model.to(self.device)
                self._model.eval()
            finally:
                # Restore the original function
                laion_clap.hook.load_state_dict = original_load_state_dict

            print("CLAP model loaded.")

    @property
    def name(self) -> str:
        return "CLAPEncoder"

    @property
    def embedding_type(self) -> str:
        return "clap"

    def get_embedding(self, audio_path: str) -> torch.Tensor:
        if self._model is None:
            self.load_model()
            
        # This method internally handles: Loading, Resampling to 48k, 
        # Mono-conversion, and Batching.
        with torch.no_grad():
            embedding = self._model.get_audio_embedding_from_filelist(x=[audio_path])
            
        return torch.from_numpy(embedding).squeeze()
