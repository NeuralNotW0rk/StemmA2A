from dataclasses import dataclass
from param_graph.registry import register
from .base import Model

@register('model:stable_audio_tools')
@dataclass(kw_only=True)
class StableAudioModel(Model):
    id: str
    checkpoint_path: str
    config: dict
    model_type: str
    engine: str = 'stable_audio_tools'
