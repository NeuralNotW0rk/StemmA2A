from dataclasses import dataclass

from param_graph.elements.base_elements import Artifact

@dataclass(kw_only=True)
class Model(Artifact):
    engine: str
    type: str = 'model'
