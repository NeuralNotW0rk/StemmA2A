from __future__ import annotations
from dataclasses import dataclass, asdict, field, fields, replace
from pathlib import Path
from time import time
from typing import Iterator, Tuple

from utils.uid import path_from_uid
from ..registry import register

@dataclass(frozen=True)
class Asset:
    """A data container for a path and a UID."""
    path: str
    uid: str

@dataclass(kw_only=True)
class GraphElement:
    id: str
    type: str
    created: int = field(default_factory=lambda: int(time()))

    def to_dict(self):
        return asdict(self)
    
    def _iter_assets(self) -> Iterator[Tuple[str, Asset]]:
        """
        Automatically discovers fields of type Asset and yields their
        name and value.
        """
        for f in fields(self):
            field_value = getattr(self, f.name)
            if isinstance(field_value, Asset):
                yield f.name, field_value

    def get_local_assets(self) -> dict[str, str]:
        """Returns a mapping of {uid: local_path} for all discovered assets."""
        assets = {}
        for _, asset_obj in self._iter_assets():
            if asset_obj.path:
                assets[asset_obj.uid] = asset_obj.path
        return assets

    def get_uids(self) -> list[str]:
        """Automatically derived from all discovered assets."""
        return [asset_obj.uid for _, asset_obj in self._iter_assets()]
    
    def de_anchor(self) -> GraphElement:
        """
        Removes local path information from all discovered assets.
        Keeps UIDs intact.
        """
        updates = {}
        for field_name, asset_obj in self._iter_assets():
            new_asset_obj = replace(asset_obj, path="")
            updates[field_name] = new_asset_obj
        return replace(self, **updates)
    
    def anchor(self, root: Path) -> GraphElement:
        """
        Constructs local paths for all discovered assets based on their UID
        and a given root directory.
        """
        updates = {}
        for field_name, asset_obj in self._iter_assets():
            new_path = str(root / path_from_uid(asset_obj.uid))
            new_asset_obj = replace(asset_obj, path=new_path)
            updates[field_name] = new_asset_obj
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
