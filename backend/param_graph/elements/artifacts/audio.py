from dataclasses import dataclass

from param_graph.elements.base_elements import Artifact
from param_graph.registry import register

@register('audio')
@dataclass(kw_only=True)
class Audio(Artifact):
    path: str
    uid: str
    type: str = 'audio'

    def get_asset_map(self) -> dict[str, str]:
        assets = super().get_asset_map()
        assets.update({
            "path": self.uid,
        })
        return assets
