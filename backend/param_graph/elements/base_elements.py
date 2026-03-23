from __future__ import annotations
from dataclasses import dataclass, asdict, field, fields, replace
from pathlib import Path
from time import time
from typing import Iterator, Tuple

from utils.uid import path_from_uid

@dataclass(frozen=True)
class Asset:
    """A data container for a path, a UID, and an extension."""
    path: str
    uid: str
    extension: str = ""

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
    
    def anchor(self, root: Path, with_extension: bool = True) -> GraphElement:
        """
        Constructs local paths for all discovered assets based on their UID
        and a given root directory.
        """
        updates = {}
        for field_name, asset_obj in self._iter_assets():
            base_path = path_from_uid(asset_obj.uid)
            
            if with_extension and hasattr(asset_obj, 'extension') and asset_obj.extension:
                ext = asset_obj.extension if asset_obj.extension.startswith('.') else '.' + asset_obj.extension
                final_path = base_path.with_suffix(ext)
            else:
                final_path = base_path

            new_path = str(Path(root) / final_path)
            new_asset_obj = replace(asset_obj, path=new_path)
            updates[field_name] = new_asset_obj
        return replace(self, **updates)
    
@dataclass(kw_only=True)
class Artifact(GraphElement):
    name: str
    context: dict
    type: str = 'artifact'

@dataclass(kw_only=True)
class Edge(GraphElement):
    source: str
    target: str
    action: str
    type: str = 'edge'
