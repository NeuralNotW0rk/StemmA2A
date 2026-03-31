from dataclasses import dataclass
from param_graph.registry import register
from .base_model_element import Model
from param_graph.elements.base_elements import Asset

@register('model:stable_audio_tools')
@dataclass(kw_only=True)
class StableAudioModel(Model):
    checkpoint: Asset
    config: dict
    model_type: str
    adapter: str = 'stable_audio_tools'