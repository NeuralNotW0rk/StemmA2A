import torch
import torchaudio.functional as F

from ..base import SyncOperation
from ..registry import register
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
