from __future__ import annotations
from dataclasses import dataclass, asdict, field, replace
from pathlib import Path
from time import time
from ..registry import register

@dataclass(kw_only=True)
class GraphElement:
    id: str
    type: str
    created: int = field(default_factory=lambda: int(time()))

    def to_dict(self):
        return asdict(self)
    
    def get_asset_map(self) -> dict[str, str]:
        """Returns a mapping of {field_name: hash_value}."""
        return {} # Base model has no hashed assets

    def get_hashes(self) -> list[str]:
        """Automatically derived from the asset map."""
        return list(self.get_asset_map().values())
    
    def de_anchor(self) -> GraphElement:
        """
        Removes local path information before sending the model 
        to a remote server. Keeps hashes intact.
        """
        # Use your existing asset map to find which fields to clear
        updates = {field: "" for field in self.get_asset_map()}
        return replace(self, **updates)
    
    def anchor(self, root_dir: str | Path) -> GraphElement:
        root = Path(root_dir)
        # Use the asset map to build the update dictionary
        updates = {
            field: str(root / hsh)
            for field, hsh in self.get_asset_map().items()
        }
        return replace(self, **updates)
    
@dataclass(kw_only=True)
class Artifact(GraphElement):
    name: str
    type: str = 'artifact'

@register('external')
@dataclass(kw_only=True)
class ExternalSource(GraphElement):
    path: str
    type: str = 'external'

@register('set')
@dataclass(kw_only=True)
class Set(GraphElement):
    alias: str
    type: str = 'set'

@register('batch')
@dataclass(kw_only=True)
class Batch(GraphElement):
    alias: str
    type: str = 'batch'
