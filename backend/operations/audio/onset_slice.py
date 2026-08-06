import torch

from ..base import SyncOperation
from ..registry import register
from param_graph.elements.artifacts.audio_element import Audio
from param_graph.elements.base_elements import Asset
from utils.audio import load_audio
from utils.uid import XXH3_64

@register
class LibrosaOnsetSliceOperation(SyncOperation):
    @property
    def name(self) -> str:
        return "librosa_onset_slice"

    @property
    def description(self) -> str:
        return "Slices audio into dynamic chunks based on transient onset detection."

    @property
    def initiator_types(self) -> list:
        return ["audio"]

    def get_form_config(self) -> list:
        return [
            {"name": "source_audio", "type": "node", "label": "Source Audio", "filter": {"type": "audio"}, "required": True},
            {"name": "backtrack", "type": "boolean", "label": "Backtrack to Local Minima", "defaultValue": True, "required": False}
        ]

    def execute(self, **kwargs) -> list[tuple[Audio, torch.Tensor]]:
        import librosa
        
        source_element = kwargs.get("source_audio_element")
        backtrack = bool(kwargs.get("backtrack", True))
        device = kwargs.get("device", "cpu")
        sample_rate = kwargs.get("sample_rate", 48000)
        
        audio_tensor = load_audio(device, source_element.file.path, sample_rate)
        mono_audio = audio_tensor.mean(dim=0).cpu().numpy()
        
        onset_samples = librosa.onset.onset_detect(
            y=mono_audio, 
            sr=sample_rate, 
            units='samples',
            backtrack=backtrack
        )
        
        total_samples = audio_tensor.shape[-1]
        boundaries = [0] + onset_samples.tolist() + [total_samples]
        
        results = []
        uid_gen = XXH3_64()
        
        for i in range(len(boundaries) - 1):
            start = boundaries[i]
            end = boundaries[i+1]
            if end > start:
                chunk = audio_tensor[..., start:end]
                if chunk.shape[-1] > 0:
                    content_uid = uid_gen.from_tensor(chunk)
                    artifact = Audio(
                        id=content_uid,
                        name=f"{source_element.name}_onset_{len(results)}",
                        file=Asset(path="", uid=content_uid, extension=".wav"),
                        sample_rate=sample_rate,
                        duration=float(chunk.shape[-1]) / sample_rate,
                        context={
                            "operation": self.name,
                            "params": {"backtrack": backtrack},
                            "source_id": source_element.id,
                            "index": len(results),
                            "seconds_start": float(start) / sample_rate,
                            "seconds_total": float(total_samples) / sample_rate,
                        }
                    )
                    results.append((artifact, chunk))
                    
        return results
