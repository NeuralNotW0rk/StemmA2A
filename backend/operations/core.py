import torch
import torchaudio.functional as F
import numpy as np

from .base import SyncOperation
from .registry import register
from param_graph.elements.artifacts.audio_element import Audio
from param_graph.elements.base_elements import Asset
from utils.audio import load_audio
from utils.uid import XXH3_64

@register
class GainOperation(SyncOperation):
    @property
    def name(self) -> str:
        return "gain"
        
    @property
    def description(self) -> str:
        return "Applies a linear gain to the audio."

    @property
    def initiator_types(self) -> list:
        return ["audio"]

    def get_form_config(self) -> list:
        return [
            {"name": "source_audio", "type": "node", "label": "Source Audio", "filter": {"type": "audio"}, "required": True},
            {
                "name": "gain_db",
                "type": "float",
                "label": "Gain (dB)",
                "defaultValue": 3.0,
                "required": True
            }
        ]

    def execute(self, **kwargs) -> list[tuple[Audio, torch.Tensor]]:
        source_element = kwargs.get("source_audio_element")
        gain_db = float(kwargs.get("gain_db", 3.0))
        device = kwargs.get("device", "cpu")
        sample_rate = kwargs.get("sample_rate", 48000)
        
        audio_tensor = load_audio(device, source_element.file.path, sample_rate)
        processed_tensor = F.gain(audio_tensor, gain_db)
        
        uid_gen = XXH3_64()
        content_uid = uid_gen.from_tensor(processed_tensor)
        
        artifact = Audio(
            id=content_uid,
            name=f"{source_element.name}_gain",
            file=Asset(path="", uid=content_uid, extension=".wav"),
            sample_rate=sample_rate,
            duration=float(processed_tensor.shape[-1]) / sample_rate,
            context={"operation": self.name, "params": {"gain_db": gain_db}, "source_id": source_element.id}
        )
        return [(artifact, processed_tensor)]

@register
class NormalizeOperation(SyncOperation):
    @property
    def name(self) -> str:
        return "normalize"

    @property
    def description(self) -> str:
        return "Normalizes audio to a target peak level."

    @property
    def initiator_types(self) -> list:
        return ["audio"]

    def get_form_config(self) -> list:
        return [
            {"name": "source_audio", "type": "node", "label": "Source Audio", "filter": {"type": "audio"}, "required": True},
            {"name": "target_peak", "type": "float", "label": "Target Peak", "defaultValue": 1.0, "required": True}
        ]

    def execute(self, **kwargs) -> list[tuple[Audio, torch.Tensor]]:
        source_element = kwargs.get("source_audio_element")
        target_peak = float(kwargs.get("target_peak", 1.0))
        device = kwargs.get("device", "cpu")
        sample_rate = kwargs.get("sample_rate", 48000)
        
        audio_tensor = load_audio(device, source_element.file.path, sample_rate)
        max_val = torch.max(torch.abs(audio_tensor))
        processed_tensor = (audio_tensor / max_val) * target_peak if max_val > 0 else audio_tensor
        
        uid_gen = XXH3_64()
        content_uid = uid_gen.from_tensor(processed_tensor)
        
        artifact = Audio(
            id=content_uid,
            name=f"{source_element.name}_norm",
            file=Asset(path="", uid=content_uid, extension=".wav"),
            sample_rate=sample_rate,
            duration=float(processed_tensor.shape[-1]) / sample_rate,
            context={"operation": self.name, "params": {"target_peak": target_peak}, "source_id": source_element.id}
        )
        return [(artifact, processed_tensor)]

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