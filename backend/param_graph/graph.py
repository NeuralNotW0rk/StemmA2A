import os
import json
from pathlib import Path
from time import time

import networkx as nx

from utils.filesystem import check_dir
from .const import *
from .elements.base_elements import GraphElement
from .registry import resolve_element

DEFAULT_SR = 48000

class ParameterGraph:
    def __init__(self, data_path, backend=None) -> None:
        self.root = Path(data_path)
        self.export_target = EXPORT_DIR
        self.backend = backend
        self.G = nx.DiGraph()
        self.project_name = None

    # IO functions
    def load(self) -> bool:
        check_dir(self.root)

        data_path = self.root / DICT_FILE
        if os.path.exists(data_path):
            with open(data_path, 'r') as df:
                data = json.load(df)
                self.project_name = data['project_name']
                self.export_target = Path(data['export_target'])
                self.G = nx.cytoscape.cytoscape_graph(data['graph'])
                
                # Clean up legacy edge nodes that were added as nodes
                legacy_edge_nodes = [
                    n for n, d in self.G.nodes(data=True) if d.get('type') == 'edge'
                ]
                if legacy_edge_nodes:
                    self.G.remove_nodes_from(legacy_edge_nodes)
            return True
        return False

    def save(self):
        check_dir(self.root)
        data_path = self.root / DICT_FILE

        # Write new json
        with open(data_path, 'w') as df:
            data = {
                'project_name': self.project_name,
                'export_target': str(self.export_target),
                'graph': nx.cytoscape.cytoscape_data(self.G, ident='id'),
            }
            df.write(json.dumps(data, indent=4))

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
