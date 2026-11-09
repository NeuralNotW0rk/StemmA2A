import torch

from ..base import SyncOperation
from ..registry import register
from param_graph.elements.artifacts.audio_element import Audio
from param_graph.elements.base_elements import Asset
from utils.audio import load_audio
from utils.uid import XXH3_64

# librosa's peak-picking threshold on its normalized onset envelope; sensitivity 0.5 maps to it exactly
DEFAULT_ONSET_DELTA = 0.07


def onset_delta_from_sensitivity(sensitivity: float) -> float:
    """
    Maps a 0-1 sensitivity onto librosa's onset threshold (delta) on a log scale: 0 gives 0.7 (only the
    strongest attacks), 0.5 gives librosa's default 0.07 and 1 gives 0.007 (nearly every peak).
    """
    if not 0.0 <= sensitivity <= 1.0:
        raise ValueError(f"Sensitivity must be between 0 and 1, got {sensitivity}.")
    return DEFAULT_ONSET_DELTA * 10 ** (1.0 - 2.0 * sensitivity)


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

    def execute(self, **kwargs) -> list[tuple[Audio, torch.Tensor]]:
        import librosa
        
        source_element = kwargs.get("source_audio_element")
        backtrack = bool(kwargs.get("backtrack", True))
        sensitivity = float(kwargs.get("sensitivity", 0.5))
        delta = onset_delta_from_sensitivity(sensitivity)
        device = kwargs.get("device", "cpu")
        sample_rate = kwargs.get("sample_rate", 48000)
        
        audio_tensor = load_audio(device, source_element.file.path, sample_rate)
        mono_audio = audio_tensor.mean(dim=0).cpu().numpy()
        
        onset_samples = librosa.onset.onset_detect(
            y=mono_audio, 
            sr=sample_rate, 
            units='samples',
            backtrack=backtrack,
            delta=delta
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
                            "params": {"backtrack": backtrack, "sensitivity": sensitivity},
                            "source_id": source_element.id,
                            "index": len(results),
                            "seconds_start": float(start) / sample_rate,
                            "seconds_total": float(total_samples) / sample_rate,
                        }
                    )
                    results.append((artifact, chunk))
                    
        return results
