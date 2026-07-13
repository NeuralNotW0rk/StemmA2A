from dataclasses import dataclass

from param_graph.elements.base_elements import Artifact

@dataclass(kw_only=True)
class Model(Artifact):
    adapter: str
    output_type: str
    type: str = 'model'
    layers: list[dict] | None = None
