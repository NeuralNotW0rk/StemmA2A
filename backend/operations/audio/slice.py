import torch

from ..base import SyncOperation
from ..registry import register
from param_graph.elements.artifacts.audio_element import Audio
from param_graph.elements.base_elements import Asset
from utils.audio import load_audio
from utils.uid import XXH3_64

@register
class SliceOperation(SyncOperation):
    @property
    def name(self) -> str:
        return "slice"

    @property
    def description(self) -> str:
        return "Slices audio into multiple chunks of a given duration."

    @property
    def initiator_types(self) -> list:
        return ["audio"]

    def get_form_config(self) -> list:
        return [
            {"name": "source_audio", "type": "node", "label": "Source Audio", "filter": {"type": "audio"}, "required": True},
            {"name": "chunk_duration", "type": "float", "label": "Chunk Duration (s)", "defaultValue": 1.0, "required": True},
            {"name": "overlap", "type": "float", "label": "Overlap (s)", "defaultValue": 0.0, "required": False}
        ]

    def execute(self, **kwargs) -> list[tuple[Audio, torch.Tensor]]:
        source_element = kwargs.get("source_audio_element")
        chunk_duration = float(kwargs.get("chunk_duration", 1.0))
        overlap = float(kwargs.get("overlap", 0.0))
        device = kwargs.get("device", "cpu")
        sample_rate = kwargs.get("sample_rate", 48000)
        
        if chunk_duration <= 0:
            raise ValueError("Chunk duration must be greater than 0.")
        if overlap < 0:
            raise ValueError("Overlap must be greater than or equal to 0.")
            
        audio_tensor = load_audio(device, source_element.file.path, sample_rate)
        chunk_samples = int(chunk_duration * sample_rate)
        overlap_samples = int(overlap * sample_rate)
        step_samples = chunk_samples - overlap_samples
        
        if step_samples <= 0:
            raise ValueError("Overlap must be strictly less than the chunk duration.")
            
        total_samples = audio_tensor.shape[-1]
        
        results = []
        uid_gen = XXH3_64()
        
        for i in range(0, total_samples, step_samples):
            chunk = audio_tensor[..., i:i + chunk_samples]
            if chunk.shape[-1] > 0:
                content_uid = uid_gen.from_tensor(chunk)
                artifact = Audio(
                    id=content_uid,
                    name=f"{source_element.name}_slice_{len(results)}",
                    file=Asset(path="", uid=content_uid, extension=".wav"),
                    sample_rate=sample_rate,
                    duration=float(chunk.shape[-1]) / sample_rate,
                    context={
                        "operation": self.name,
                        "params": {"chunk_duration": chunk_duration, "overlap": overlap},
                        "source_id": source_element.id,
                        "index": len(results),
                        "seconds_start": float(i) / sample_rate,
                        "seconds_total": float(total_samples) / sample_rate,
                    }
                )
                results.append((artifact, chunk))
                
        return results
