import os
import json
from pathlib import Path
from time import time
import networkx as nx

from .util import *
from .const import *
from .components.importer import Importer
from .components.exporter import Exporter
from .components.inference_logger import InferenceLogger
from .components.clusterer import Clusterer

DEFAULT_SR = 48000

class ParameterGraph:
    def __init__(self, data_path, backend=None, relative=True) -> None:
        self.root = Path(data_path)
        self.export_target = EXPORT_DIR
        self.backend = backend
        self.G = nx.DiGraph()
        self.project_name = None
        
        # Components
        self.importer = Importer(self.G, self.root)
        self.exporter = Exporter(self.G, self.root)
        self.inference_logger = InferenceLogger(self.G, self.root)
        self.clusterer = Clusterer(self.G, self.root)
        
        self.load()

    # Component-based methods
    def import_model(self, *args, **kwargs):
        return self.importer.import_model(*args, **kwargs)

    def add_external_source(self, *args, **kwargs):
        return self.importer.add_external_source(*args, **kwargs)

    def scan_dir(self, *args, **kwargs):
        return self.importer.scan_dir(*args, **kwargs)

    def scan_external_source(self, *args, **kwargs):
        return self.importer.scan_external_source(*args, **kwargs)

    def import_audio_set(self, *args, **kwargs):
        return self.importer.import_audio_set(*args, **kwargs)

    def export_single(self, *args, **kwargs):
        return self.exporter.export_single(*args, **kwargs)

    def export_batch(self, *args, **kwargs):
        return self.exporter.export_batch(*args, **kwargs)

    def log_inference(self, *args, **kwargs):
        return self.inference_logger.log_inference(*args, **kwargs)

    def update_tsne(self, *args, **kwargs):
        return self.clusterer.update_tsne(*args, **kwargs)

    # IO functions
    def load(self):
        check_dir(self.root)

        with open(self.root / DICT_FILE, 'r') as df:
            data = json.load(df)
            self.project_name = data['project_name']
            self.export_target = Path(data['export_target'])
            self.G = nx.cytoscape.cytoscape_graph(data['graph'])
        

    def save(self):
        os.system(
            f'cp "{self.root / DICT_FILE}" "{check_dir(self.root / BACKIP_DIR) / DICT_FILE}_{int(time())}"'
        )
        with open(self.root / DICT_FILE, 'w') as df:
            data = {
                'project_name': self.project_name,
                'export_target': str(self.export_target),
                'graph': nx.cytoscape.cytoscape_data(self.G),
            }
            df.write(json.dumps(data, indent=4))

    def to_json(self, mode='batch'):
        if mode == 'batch':
            return nx.cytoscape.cytoscape_data(self.G)
        elif mode == 'cluster':
            C = nx.DiGraph()
            for node, data in self.G.nodes(data=True):
                if data['type'] == 'audio':
                    C.add_node(node, **data)
                    C.nodes[node].pop('parent', None)
            return nx.cytoscape.cytoscape_data(C)

    def get_path_from_name(self, name: str, relative=False):
        if self.G.has_node(name):
            path_str = self.G.nodes[name].get('path')
            if path_str:
                path = Path(path_str)
                if path.is_absolute():
                    return str(path)
                return str(self.root / path)
        return None

    # Simple element attribute update
    def update_element(self, name: str, attrs: dict):
        nx.function.set_node_attributes(self.G, {name: attrs})

    # Slightly less simple batch attribute update
    def update_batch(self, name: str, attrs: dict):
        if 'alias' in attrs:
            # Update batch alias
            nx.function.set_node_attributes(self.G, {name: {'alias': attrs['alias']}})
            if attrs['apply_child_alias']:
                # Update all children aliases
                for node, data in self.G.nodes(data=True):
                    if data.get('parent') == name:
                        new_alias = f'{attrs["alias"]}_{data["batch_index"]}'
                        self.G.nodes[node]['alias'] = new_alias

        if 'tags' in attrs and attrs['tags']:
            # Add tags to child tag lists
            for node, data in self.G.nodes(data=True):
                if data.get('parent') == name:
                    existing_tags = data.get('tags', [])
                    if isinstance(existing_tags, str):
                        existing_tags = [tag.strip() for tag in existing_tags.split(',') if tag.strip()]
                    
                    new_tags = attrs['tags']
                    if isinstance(new_tags, str):
                        new_tags = [tag.strip() for tag in new_tags.split(',') if tag.strip()]

                    updated_tags = sorted(list(set(existing_tags) | set(new_tags)))
                    self.G.nodes[node]['tags'] = updated_tags

    # Remove element (and children in the case of batches)
    def remove_element(self, name: str):
        to_remove = [name]
        for node, data in self.G.nodes(data=True):
            if data.get('parent') == name:
                to_remove.append(node)

        self.G.remove_nodes_from(to_remove)