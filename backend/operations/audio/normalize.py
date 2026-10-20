import torch

from ..base import SyncOperation
from ..registry import register
from param_graph.elements.artifacts.audio_element import Audio
from param_graph.elements.base_elements import Asset
from utils.audio import load_audio
from utils.uid import XXH3_64

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
