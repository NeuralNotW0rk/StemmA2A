from dataclasses import dataclass
from param_graph.registry import register
from .base_model_element import Model
from param_graph.elements.base_elements import Asset

@register('model:stylegan2')
@dataclass(kw_only=True)
class StyleGANModel(Model):
    checkpoint: Asset
    config: dict | None = None
    adapter: str = 'stylegan2'
    output_type: str = 'image'
