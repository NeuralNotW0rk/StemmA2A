import torch
from PIL import Image as PILImage
from transformers import CLIPModel, CLIPProcessor
from .base_encoder import Encoder


class CLIPEncoder(Encoder):
    """
    A CLIP encoder for generating image embeddings using the Hugging Face transformers model.

    This class provides a simple interface to load a pre-trained CLIP model
    and use it to encode images into semantic embeddings.
    """
    def __init__(self):
        """
        Initializes the CLIPEncoder.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = None
        self._processor = None

    def load_model(self):
        """
        Loads the pre-trained CLIP model and processor from Hugging Face.
        """
        if self._model is None:
            model_id = "openai/clip-vit-base-patch32"
            print(f"Loading CLIP model ({model_id})...")
            try:
                # Fast path: attempt to load from local cache to bypass network overhead
                self._processor = CLIPProcessor.from_pretrained(model_id, local_files_only=True)
                self._model = CLIPModel.from_pretrained(model_id, local_files_only=True).to(self.device)
                print("CLIP model loaded directly from local cache.")
            except Exception:
                # Fallback: go online to download and cache if not fully present
                print("CLIP model not fully cached. Downloading from Hugging Face Hub...")
                self._processor = CLIPProcessor.from_pretrained(model_id)
                self._model = CLIPModel.from_pretrained(model_id).to(self.device)
                print("CLIP model downloaded and loaded.")
                
            self._model.eval()

    @property
    def name(self) -> str:
        return "CLIPEncoder"

    @property
    def embedding_type(self) -> str:
        return "clip"

    def get_embedding(self, file_path: str) -> torch.Tensor:
        if self._model is None:
            self.load_model()
            
        # Load image
        image = PILImage.open(file_path).convert("RGB")
        
        inputs = self._processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model.get_image_features(**inputs)
            
        # Address the BaseModelOutputWithPooling directly
        if not isinstance(outputs, torch.Tensor):
            embedding = outputs.pooler_output
        else:
            embedding = outputs
            
        return embedding.squeeze()

    def get_text_embedding(self, texts: list[str]) -> torch.Tensor:
        if self._model is None:
            self.load_model()
            
        inputs = self._processor(text=texts, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self._model.get_text_features(**inputs)
            
        # Address the BaseModelOutputWithPooling directly
        if not isinstance(outputs, torch.Tensor):
            embedding = outputs.pooler_output
        else:
            embedding = outputs
            
        return embedding
