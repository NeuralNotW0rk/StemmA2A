import os
import json
from pathlib import Path
from time import time
import networkx as nx

import torch
import numpy as np
import librosa as lr

from sklearn.manifold import TSNE

from .util import *
from .const import *

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
            return True
        return False

    def save(self):
        data_path = self.root / DICT_FILE

        # Backup json if it exists
        if os.path.exists(data_path):
            os.system(
                f'cp "{data_path}" "{check_dir(self.root / BACKIP_DIR) / DICT_FILE}_{int(time())}"'
            )

        # Write new json
        with open(data_path, 'w') as df:
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

    def update_tsne(
        self, n_components=2, perplexity=40, n_iter=300, sample_rate=48000, sample_size=None
    ):
        if sample_size is None:
            sample_size = sample_rate

        # Gather audio samples
        print('Gathering audio samples for t-SNE calculation...')
        names = []
        samples = []
        for node, data in self.G.nodes(data=True):
            if data['type'] == 'audio':
                try:
                    path_str = data.get('path')
                    if not path_str:
                        print(f"Node {node} has no path, skipping.")
                        continue
                    
                    audio_path = Path(path_str)
                    if not audio_path.is_absolute():
                        audio_path = self.root / audio_path

                    sample_raw = load_audio(
                        'cpu', str(audio_path), sample_rate=sample_rate
                    )
                    sample = torch.zeros(sample_size)
                    cropped_size = min(sample_size, sample_raw.size(1))
                    sample[:cropped_size] += sample_raw[0, :cropped_size]
                    samples.append(sample.numpy())
                    names.append(node)
                except Exception as e:
                    print(f"Could not load audio for node {node}, skipping. Error: {e}")

        if not samples:
            print("No audio samples found or loaded. Skipping t-SNE calculation.")
            return
        
        samples = np.asarray(samples)

        # Handle if number of samples is smaller than perplexity
        perplexity = min(perplexity, len(samples) - 1)

        # Convert to spectrograms
        # TODO: Save spectrograms for reuse
        print('Extracting spectrograms...')
        specs = lr.stft(samples, n_fft=512)
        specs = np.abs(specs)
        specs = np.reshape(specs, newshape=(specs.shape[0], specs.shape[1] * specs.shape[2]))

        # Compute t-SNE
        tsne = TSNE(
            n_components=n_components, verbose=1, perplexity=perplexity, n_iter=n_iter, random_state=0
        )
        tsne_results = tsne.fit_transform(specs)

        # Update nodes
        attrs = {}
        for name, result in zip(names, tsne_results):
            attrs[name] = {'tsne': result.tolist()}
        
        nx.set_node_attributes(self.G, attrs)
