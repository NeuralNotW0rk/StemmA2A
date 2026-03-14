from dataclasses import dataclass
from param_graph.registry import register
from .base import Model

@register('model:stable_audio_tools')
@dataclass(kw_only=True)
class StableAudioModel(Model):
    id: str
    checkpoint_path: str
    checkpoint_hash: str
    config_path: str
    config_hash: str
    config: dict
    model_type: str
    adapter: str = 'stable_audio_tools'

    def get_asset_map(self) -> dict[str, str]:
        assets = super().get_asset_map()
        assets.update({
            "checkpoint_path": self.checkpoint_hash,
            "config_path": self.config_hash,
        })
        return assets
    
