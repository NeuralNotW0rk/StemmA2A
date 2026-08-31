import os
import json
import shutil
from pathlib import Path
from time import time

import networkx as nx

from utils.filesystem import check_dir
from .const import *
from .elements.base_elements import GraphElement
from .registry import resolve_element

DEFAULT_SR = 48000


def _safe_json_default(obj):
    """
    Fallback serializer for custom or non-standard objects in graph nodes.
    Converts NumPy types, Path objects, Sets, and PyTorch tensors safely.
    """
    if isinstance(obj, Path):
        return str(obj)
    
    if isinstance(obj, (set, frozenset)):
        return list(obj)
    
    try:
        import numpy as np
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
    except ImportError:
        pass
    
    try:
        import torch
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().tolist()
        if isinstance(obj, torch.dtype):
            return str(obj)
    except ImportError:
        pass
    
    if hasattr(obj, "to_dict") and callable(obj.to_dict):
        return obj.to_dict()
    
    if hasattr(obj, "__dict__"):
        return obj.__dict__
        
    return str(obj)


class ParameterGraph:
    def __init__(self, data_path, backend=None) -> None:
        self.root = Path(data_path)
        self.backend = backend
        self.G = nx.DiGraph()
        self.project_name = None

    # IO functions
    def load(self) -> bool:
        check_dir(self.root)

        data_path = self.root / DICT_FILE
        bak_path = self.root / f"{DICT_FILE}.bak"

        target_path = None

        if os.path.exists(data_path) and os.path.getsize(data_path) > 0:
            target_path = data_path
        elif os.path.exists(bak_path) and os.path.getsize(bak_path) > 0:
            print(f"[ParameterGraph] Warning: '{data_path}' is missing or empty. Recovering from backup '{bak_path}'.")
            target_path = bak_path

        if not target_path:
            return False

        try:
            with open(target_path, 'r', encoding='utf-8') as df:
                data = json.load(df)
        except json.JSONDecodeError as e:
            # If main file is corrupted, try backup if available
            if target_path != bak_path and os.path.exists(bak_path) and os.path.getsize(bak_path) > 0:
                print(f"[ParameterGraph] Warning: '{data_path}' is corrupted ({e}). Falling back to backup '{bak_path}'.")
                try:
                    with open(bak_path, 'r', encoding='utf-8') as df:
                        data = json.load(df)
                except Exception as bak_err:
                    raise ValueError(f"Corrupted project graph file at '{data_path}' and backup at '{bak_path}': {bak_err}") from e
            else:
                raise ValueError(f"Corrupted project graph file at '{target_path}': {e}") from e

        self.project_name = data.get('project_name', self.root.name)
        self.G = nx.cytoscape.cytoscape_graph(data.get('graph', {}))
        
        # Clean up legacy edge nodes that were added as nodes
        legacy_edge_nodes = [
            n for n, d in self.G.nodes(data=True) if d.get('type') == 'edge'
        ]
        if legacy_edge_nodes:
            self.G.remove_nodes_from(legacy_edge_nodes)

        # Clean up legacy edges between compound parent and direct child
        legacy_parent_child_edges = [
            (u, v) for u, v in self.G.edges()
            if self.G.has_node(u) and self.G.has_node(v) and self.G.nodes[v].get('parent') == u
        ]
        if legacy_parent_child_edges:
            self.G.remove_edges_from(legacy_parent_child_edges)

        # Ensure all model nodes have output_type populated
        for node, d in self.G.nodes(data=True):
            if d.get('type') == 'model':
                if 'output_type' not in d:
                    adapter = d.get('adapter')
                    if adapter == 'stable_audio_tools':
                        d['output_type'] = 'audio'
                    elif adapter == 'stylegan2':
                        d['output_type'] = 'image'
        return True

    def save(self):
        check_dir(self.root)
        data_path = self.root / DICT_FILE
        temp_path = self.root / f"{DICT_FILE}.tmp"
        bak_path = self.root / f"{DICT_FILE}.bak"

        data = {
            'project_name': self.project_name or self.root.name,
            'graph': nx.cytoscape.cytoscape_data(self.G, ident='id'),
        }

        # 1. Serialize in-memory FIRST. If this fails, the file on disk is untouched.
        json_str = json.dumps(data, indent=4, default=_safe_json_default)

        # 2. Write to temporary file with explicit flush and fsync
        with open(temp_path, 'w', encoding='utf-8') as df:
            df.write(json_str)
            df.flush()
            os.fsync(df.fileno())

        # 3. Create rolling backup if target file currently exists and is non-empty
        if data_path.exists() and data_path.stat().st_size > 0:
            try:
                shutil.copy2(str(data_path), str(bak_path))
            except Exception as e:
                print(f"[ParameterGraph] Warning: Failed to create backup {bak_path}: {e}")

        # 4. Atomically swap temp_path to data_path
        try:
            os.replace(str(temp_path), str(data_path))
        except Exception:
            if os.path.exists(str(data_path)):
                os.remove(str(data_path))
            os.rename(str(temp_path), str(data_path))

    def to_json(self, mode='batch'):
        if mode == 'batch':
            return nx.cytoscape.cytoscape_data(self.G, ident='id')
        elif mode == 'cluster':
            C = nx.DiGraph()
            for node, data in self.G.nodes(data=True):
                if data['type'] == 'audio':
                    C.add_node(node, **data)
                    C.nodes[node].pop('parent', None)
            return nx.cytoscape.cytoscape_data(C, ident='id')
        
    def add_element(self, ele: GraphElement):
        ele_attrs = ele.to_dict()
        if ele_attrs.get('type') == 'model' and not ele_attrs.get('output_type'):
            adapter = ele_attrs.get('adapter')
            if adapter == 'stable_audio_tools':
                ele_attrs['output_type'] = 'audio'
            elif adapter == 'stylegan2':
                ele_attrs['output_type'] = 'image'
        ele_id = ele_attrs.get('id', None)
        self.G.add_node(ele_id, **ele_attrs)

    def link(self, source: GraphElement, target: GraphElement, **kwargs):
        edge_id = f"{source.id}->{target.id}"
        edge_attrs = {"type": source.type, "id": edge_id}
        edge_attrs.update(kwargs)
        self.G.add_edge(source.id, target.id, **edge_attrs)
        return {
            "source": source.id,
            "target": target.id,
            **edge_attrs
        }

    def get_element(self, id: str) -> GraphElement:
        """
        Retrieves a node's attributes from the graph and reconstructs
        its corresponding dataclass object using the central registry.
        """
        if not self.G.has_node(id):
            raise ValueError(f"Node '{id}' not found in the graph.")

        attrs = self.G.nodes[id].copy()
        return resolve_element(attrs)

    def get_path_from_id(self, id: str, relative=False):
        if self.G.has_node(id):
            node_data = self.G.nodes[id]
            file_info = node_data.get('file')
            if not file_info:
                return None
            
            path_str = file_info.get('path')
            if path_str:
                path = Path(path_str)
                if not path.is_absolute():
                    path = self.root / path
                
                # Check if the path exists, if not, try adding a .wav extension
                # for backwards compatibility with old projects.
                if not path.exists():
                    path_with_ext = path.with_suffix(".wav")
                    if path_with_ext.exists():
                        path = path_with_ext

                if relative:
                    # This is tricky because the "relative" path needs to be
                    # relative to the VFS root, not the graph's root.
                    # For now, we assume the graph root IS the VFS root.
                    return str(path)
                
                return str(path.resolve())
        return None

    # Simple element attribute update
    def update_element(self, id: str, attrs: dict):
        if self.G.has_node(id):
            node_attrs = self.G.nodes[id]
            node_attrs.update(attrs)

    # Remove element (and children recursively)
    def remove_element(self, id: str):
        to_remove = {id}
        
        while True:
            added = False
            for node, data in self.G.nodes(data=True):
                if node not in to_remove and data.get('parent') in to_remove:
                    to_remove.add(node)
                    added = True
            if not added:
                break

        self.G.remove_nodes_from(to_remove)
