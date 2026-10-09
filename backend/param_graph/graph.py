import os
import json
import uuid
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
        # Embeddings as of the last sidecar read/write, used to skip unchanged rewrites
        self._saved_embeddings = None

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

        self._load_embeddings()
        return True

    def _load_embeddings(self) -> None:
        """
        Merges node embeddings from the sidecar file back into the in-memory graph. Embeddings still
        stored inline in graph.json (legacy projects) are kept, and move to the sidecar on next save.
        """
        self._saved_embeddings = None
        sidecar_path = self.root / EMBEDDINGS_FILE
        if not sidecar_path.exists():
            return

        from safetensors.numpy import load_file
        try:
            tensors = load_file(str(sidecar_path))
        except Exception as e:
            print(f"[ParameterGraph] Warning: Could not read embeddings file '{sidecar_path}': {e}")
            return

        for key, arr in tensors.items():
            node_id, _, embedding_type = key.rpartition(EMBEDDING_KEY_SEPARATOR)
            if self.G.has_node(node_id):
                node_embeddings = self.G.nodes[node_id].get('embeddings')
                if not isinstance(node_embeddings, dict):
                    node_embeddings = self.G.nodes[node_id]['embeddings'] = {}
                node_embeddings[embedding_type] = arr.tolist()
        self._saved_embeddings = self._collect_embeddings()

    def _collect_embeddings(self) -> dict[str, dict]:
        """Returns a {node_id: {embedding_type: vector}} snapshot of all numeric node embeddings."""
        collected = {}
        for node, d in self.G.nodes(data=True):
            embeddings = d.get('embeddings')
            if isinstance(embeddings, dict) and embeddings:
                collected[node] = dict(embeddings)
        return collected

    def _save_embeddings(self, embeddings: dict[str, dict]) -> None:
        """Writes node embeddings to the sidecar file as float32 vectors, skipping the write if unchanged."""
        sidecar_path = self.root / EMBEDDINGS_FILE
        if embeddings == self._saved_embeddings or (not embeddings and not sidecar_path.exists()):
            self._saved_embeddings = embeddings
            return

        import numpy as np
        from safetensors.numpy import save_file

        tensors = {
            f"{node}{EMBEDDING_KEY_SEPARATOR}{embedding_type}": np.ascontiguousarray(vector, dtype=np.float32)
            for node, node_embeddings in embeddings.items()
            for embedding_type, vector in node_embeddings.items()
        }
        temp_path = self.root / f"{EMBEDDINGS_FILE}.{uuid.uuid4().hex[:8]}.tmp"
        save_file(tensors, str(temp_path))
        os.replace(str(temp_path), str(sidecar_path))
        self._saved_embeddings = embeddings

    def save(self):
        check_dir(self.root)
        data_path = self.root / DICT_FILE
        temp_path = self.root / f"{DICT_FILE}.{uuid.uuid4().hex[:8]}.tmp"
        bak_path = self.root / f"{DICT_FILE}.bak"

        graph_data = nx.cytoscape.cytoscape_data(self.G, ident='id')
        # Embeddings live in a binary sidecar file; cytoscape_data copies each node's dict, so this
        # leaves the in-memory graph untouched
        for node in graph_data['elements']['nodes']:
            if 'embeddings' in node['data']:
                node['data']['embeddings'] = {}
        data = {
            'project_name': self.project_name or self.root.name,
            'graph': graph_data,
        }

        # 1. Serialize in-memory FIRST. If this fails, the file on disk is untouched.
        json_str = json.dumps(data, separators=(',', ':'), default=_safe_json_default)

        # 2. Write the embeddings sidecar, then the graph to a temporary file with explicit flush and fsync
        self._save_embeddings(self._collect_embeddings())
        with open(temp_path, 'w', encoding='utf-8') as df:
            df.write(json_str)
            df.flush()
            os.fsync(df.fileno())

        # 3. Rotate the current file into the rolling backup (a rename, not a copy). load() falls back
        # to the backup if the process stops before the swap below
        if data_path.exists() and data_path.stat().st_size > 0:
            try:
                os.replace(str(data_path), str(bak_path))
            except Exception as e:
                print(f"[ParameterGraph] Warning: Failed to create backup {bak_path}: {e}")

        # 4. Atomically swap temp_path to data_path (with retry on Windows)
        last_err = None
        for attempt in range(5):
            try:
                os.replace(str(temp_path), str(data_path))
                last_err = None
                break
            except Exception as e:
                last_err = e
                import time
                time.sleep(0.05 * (attempt + 1))
        
        if last_err is not None:
            try:
                if os.path.exists(str(data_path)):
                    os.remove(str(data_path))
                os.rename(str(temp_path), str(data_path))
            except Exception as final_err:
                if os.path.exists(str(temp_path)):
                    try:
                        os.remove(str(temp_path))
                    except Exception:
                        pass
                raise final_err

    def to_json(self, mode='batch'):
        """Returns the graph in Cytoscape format for the frontend, without node embedding vectors."""
        if mode == 'batch':
            graph_data = nx.cytoscape.cytoscape_data(self.G, ident='id')
        elif mode == 'cluster':
            C = nx.DiGraph()
            for node, data in self.G.nodes(data=True):
                if data['type'] == 'audio':
                    C.add_node(node, **data)
                    C.nodes[node].pop('parent', None)
            graph_data = nx.cytoscape.cytoscape_data(C, ident='id')
        else:
            return None

        # cytoscape_data copies each node's dict, so this leaves the graph untouched
        for node in graph_data['elements']['nodes']:
            if 'embeddings' in node['data']:
                node['data']['embeddings'] = {}
        return graph_data
        
    def add_element(self, ele: GraphElement, allow_duplicates: bool = True) -> bool:
        """
        Adds an element node to the parameter graph.

        If allow_duplicates is False and a node with the same ID already exists in the graph,
        the node is NOT added or overwritten, and False is returned.

        Returns:
            bool: True if the element was added/updated in the graph, False if blocked as duplicate.
        """
        ele_attrs = ele.to_dict()
        ele_id = ele_attrs.get('id', None)
        if not allow_duplicates and ele_id and self.G.has_node(ele_id):
            return False

        if ele_attrs.get('type') == 'model' and not ele_attrs.get('output_type'):
            adapter = ele_attrs.get('adapter')
            if adapter == 'stable_audio_tools':
                ele_attrs['output_type'] = 'audio'
            elif adapter == 'stylegan2':
                ele_attrs['output_type'] = 'image'
        self.G.add_node(ele_id, **ele_attrs)
        return True

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
