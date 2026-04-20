import torch
import torchaudio
import torchaudio.transforms as T
from transformers import ClapModel, ClapProcessor
from .base_encoder import Encoder


class CLAPEncoder(Encoder):
    """
    A CLAP encoder for generating audio embeddings using the Hugging Face transformers model.

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
        self._processor = None
        self.target_sr = 48000

    def load_model(self):
        """
        Loads the pre-trained CLAP model and processor from Hugging Face.
        """
        if self._model is None:
            model_id = "laion/clap-htsat-unfused"
            print(f"Loading CLAP model ({model_id})...")
            try:
                # Fast path: attempt to load from local cache to bypass network overhead
                self._processor = ClapProcessor.from_pretrained(model_id, local_files_only=True)
                self._model = ClapModel.from_pretrained(model_id, local_files_only=True).to(self.device)
                print("CLAP model loaded directly from local cache.")
            except Exception:
                # Fallback: go online to download and cache if not fully present
                print("CLAP model not fully cached. Downloading from Hugging Face Hub...")
                self._processor = ClapProcessor.from_pretrained(model_id)
                self._model = ClapModel.from_pretrained(model_id).to(self.device)
                print("CLAP model downloaded and loaded.")
                
            self._model.eval()

    @property
    def name(self) -> str:
        return "CLAPEncoder"

    @property
    def embedding_type(self) -> str:
        return "clap"

    def get_embedding(self, audio_path: str) -> torch.Tensor:
        if self._model is None:
            self.load_model()
            
        waveform, sr = torchaudio.load(audio_path)
        
        # Convert to mono if necessary
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
            
        # Resample if necessary
        if sr != self.target_sr:
            resampler = T.Resample(orig_freq=sr, new_freq=self.target_sr)
            waveform = resampler(waveform)
            
        # HF Processor expects a 1D numpy array
        audio_input = waveform.squeeze().numpy()
        
        inputs = self._processor(audios=audio_input, return_tensors="pt", sampling_rate=self.target_sr)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            embedding = self._model.get_audio_features(**inputs)
            
        return embedding.squeeze()

    def get_text_embedding(self, texts: list[str]) -> torch.Tensor:
        if self._model is None:
            self.load_model()
            
        inputs = self._processor(text=texts, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            embedding = self._model.get_text_features(**inputs)
            
        return embedding
