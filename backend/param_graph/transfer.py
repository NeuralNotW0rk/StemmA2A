"""
Moves a subset of one project's graph, with the files it owns, into another graph.

collect_subgraph() snapshots a selection plus everything it depends on from a source graph;
write_subgraph() copies that snapshot's project-internal files into a target graph's root and
adds its nodes and edges there. Breakout (fork a selection into a new project) is the first
user; importing or pasting between projects only needs a different target.
"""
import copy
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import networkx as nx

from utils.uid import path_from_uid

# Context keys referencing the individuals a generated exemplar was expressed from. They only hold
# while the exemplar sits inside its individual, so they move to the ancestry record when forked without it
EXEMPLAR_INDIVIDUAL_KEYS = ('individuals', 'individual_ids', 'baseline_grating', 'baseline_grating_id')
# Context keys that are bookkeeping, not references to other elements
NON_REFERENCE_CONTEXT_KEYS = ('job_id', 'group_id', 'batch_id', 'parent_id', 'execution_mode')


@dataclass
class Subgraph:
    """A detached copy of graph nodes and edges, with asset paths still pointing into source_root."""
    source_root: Path
    source_name: str
    selected_ids: list[str]
    nodes: dict[str, dict] = field(default_factory=dict)
    edges: list[tuple[str, str, dict]] = field(default_factory=list)
    # Generated exemplars whose individual was left behind, now plain artifacts
    converted_exemplars: list[str] = field(default_factory=list)


def _is_generated_exemplar(G: nx.DiGraph, node_id: str) -> bool:
    """Mirrors ParameterGraph.is_generated_exemplar for a bare networkx graph."""
    attrs = G.nodes[node_id]
    parent_id = attrs.get('parent')
    if not parent_id or not G.has_node(parent_id) or G.nodes[parent_id].get('type') != 'individual':
        return False
    context = attrs.get('context')
    individuals = context.get('individuals') if isinstance(context, dict) else None
    if not isinstance(individuals, list):
        return False
    return any((ind.get('id') if isinstance(ind, dict) else ind) == parent_id for ind in individuals)


def _ref_id(item) -> str | None:
    value = item.get('id') if isinstance(item, dict) else item
    return value if isinstance(value, str) and value else None


def _context_references(context: dict, skip_keys: tuple[str, ...] = ()) -> list[str]:
    """Element IDs an artifact's context points at: the inputs it was made from and its lineage."""
    refs: list[str] = []
    for key, value in context.items():
        if key in NON_REFERENCE_CONTEXT_KEYS or key in skip_keys:
            continue
        if isinstance(value, str) and (
            key.endswith('_id') or key in ('init_audio', 'init_latent', 'init_image', 'source_audio', 'precursor_audio')
        ):
            refs.append(value)
        elif key in ('gratings', 'grating_ids', 'individuals', 'individual_ids') and isinstance(value, list):
            refs.extend(ref for ref in map(_ref_id, value) if ref)
        elif key == 'lineage' and isinstance(value, dict):
            refs.extend(ref for ref in map(_ref_id, value.get('parent_ids') or []) if ref)
    return refs


def _dependencies(G: nx.DiGraph, node_id: str, include_lineage: bool) -> list[str]:
    """
    Upstream elements a node needs. Required ones (the model an individual or grating binds to,
    an individual's baseline grating) always come along; inputs and ancestors only with lineage.
    A generated exemplar's inputs are implied through its individual, so they are not followed here.
    """
    attrs = G.nodes[node_id]
    deps = [attrs[key] for key in ('base_model_id', 'baseline_grating_id') if isinstance(attrs.get(key), str)]
    if not include_lineage or _is_generated_exemplar(G, node_id):
        return deps

    deps.extend(
        source for source, _, edge in G.in_edges(node_id, data=True)
        # A shared exemplar is reached from its individual, it does not depend on it
        if edge.get('type') != 'spring' and edge.get('relation') != 'shares_phenotype'
    )
    context = attrs.get('context')
    if isinstance(context, dict):
        deps.extend(_context_references(context))
    return deps


def _contents(G: nx.DiGraph, node_id: str) -> list[str]:
    """Nodes nested in a compound node: its children and, for groups and bundles, its members."""
    contents = [n for n, d in G.nodes(data=True) if d.get('parent') == node_id]
    member_ids = G.nodes[node_id].get('member_ids')
    if isinstance(member_ids, list):
        contents.extend(m for m in member_ids if isinstance(m, str))
    return contents


def _ancestry(G: nx.DiGraph, individual_ids: list[str]) -> list[dict]:
    """Walks individual lineage breadth-first, summarizing each ancestor for a detached exemplar."""
    chain: list[dict] = []
    seen: set[str] = set()
    frontier = list(individual_ids)
    while frontier:
        ind_id = frontier.pop(0)
        if ind_id in seen or not G.has_node(ind_id):
            continue
        seen.add(ind_id)
        attrs = G.nodes[ind_id]
        context = attrs.get('context') if isinstance(attrs.get('context'), dict) else {}
        parent_ids = [ref for ref in map(_ref_id, (context.get('lineage') or {}).get('parent_ids') or []) if ref]
        entry = {
            'id': ind_id,
            'name': attrs.get('name'),
            'generation': attrs.get('generation'),
            'parent_ids': parent_ids,
        }
        for key in ('mutation_operation', 'recombine_operation', 'precursor_artifact_id'):
            if key in context:
                entry[key] = copy.deepcopy(context[key])
        chain.append(entry)
        frontier.extend(parent_ids)
    return chain


def _detach_exemplar(G: nx.DiGraph, node_id: str, attrs: dict) -> None:
    """
    Turns a generated exemplar into a plain artifact. Its references to individuals move into
    context['ancestry'], together with the lineage recovered from those individuals, so its
    provenance survives without them.
    """
    context = attrs['context']
    individual_ids = [ref for ref in map(_ref_id, context.get('individuals') or []) if ref]
    ancestry = {key: context.pop(key) for key in EXEMPLAR_INDIVIDUAL_KEYS if key in context}
    # Legacy exemplars inlined the whole baseline grating; its ID is all the record needs
    baseline = ancestry.pop('baseline_grating', None)
    if isinstance(baseline, dict) and baseline.get('id') and 'baseline_grating_id' not in ancestry:
        ancestry['baseline_grating_id'] = baseline['id']
    ancestry['lineage'] = _ancestry(G, individual_ids)
    context['ancestry'] = ancestry


def collect_subgraph(graph, element_ids: list[str], include_lineage: bool = True) -> Subgraph:
    """
    Snapshots the given elements, the contents of any compound among them, and their dependencies.
    Generated exemplars whose individual is not part of the result become plain artifacts, and
    their own inputs (model, init latents, ...) are followed as dependencies instead.
    Node attributes are deep copies, so the snapshot can be written without holding the graph lock.
    """
    G: nx.DiGraph = graph.G
    missing = [eid for eid in element_ids if not G.has_node(eid)]
    if missing:
        raise ValueError(f"Elements not found in the graph: {missing}")

    selected = set(element_ids)
    included: set[str] = set()

    def expand(frontier: list[str]) -> None:
        queue = list(frontier)
        while queue:
            node_id = queue.pop()
            if node_id in included or not G.has_node(node_id):
                continue
            included.add(node_id)
            # Individuals always carry their exemplars; other compounds only when selected directly
            if node_id in selected or G.nodes[node_id].get('type') == 'individual':
                queue.extend(_contents(G, node_id))
            queue.extend(_dependencies(G, node_id, include_lineage))

    def orphaned_exemplars() -> list[str]:
        return [
            n for n in included
            if _is_generated_exemplar(G, n) and G.nodes[n]['parent'] not in included
        ]

    expand(list(element_ids))
    # Detached exemplars lose the individual their inputs were implied through, so follow them directly
    while include_lineage:
        extra = [
            ref
            for n in orphaned_exemplars()
            for ref in _context_references(G.nodes[n]['context'], skip_keys=EXEMPLAR_INDIVIDUAL_KEYS)
            if G.has_node(ref) and ref not in included
        ]
        if not extra:
            break
        expand(extra)

    subgraph = Subgraph(
        source_root=Path(graph.root),
        source_name=graph.project_name or Path(graph.root).name,
        selected_ids=list(element_ids),
    )
    orphans = set(orphaned_exemplars())
    for node_id in included:
        attrs = copy.deepcopy(G.nodes[node_id])
        if node_id in orphans:
            _detach_exemplar(G, node_id, attrs)
            subgraph.converted_exemplars.append(node_id)
        if attrs.get('parent') and attrs['parent'] not in included:
            attrs.pop('parent')
            attrs.pop('alias', None)
            attrs.pop('full_alias', None)
        if isinstance(attrs.get('member_ids'), list):
            attrs['member_ids'] = [m for m in attrs['member_ids'] if m in included]
        subgraph.nodes[node_id] = attrs

    # Spring edges are similarity data over the whole project and are rebuilt by the embedding pass
    subgraph.edges = [
        (u, v, copy.deepcopy(d)) for u, v, d in G.edges(data=True)
        if u in included and v in included and d.get('type') != 'spring'
    ]
    return subgraph


def _iter_asset_dicts(attrs: dict):
    """Yields every serialized Asset ({path, uid, ...}) among a node's attributes."""
    for value in attrs.values():
        if isinstance(value, dict) and 'path' in value and 'uid' in value:
            yield value


def _relative_to(path: Path, root: Path) -> Path | None:
    """path relative to root if it lies inside it (case-insensitive on Windows), else None."""
    path_norm = os.path.normcase(os.path.abspath(path))
    root_norm = os.path.normcase(os.path.abspath(root))
    try:
        if os.path.commonpath([path_norm, root_norm]) != root_norm:
            return None
    except ValueError:
        # Different drives
        return None
    return Path(os.path.relpath(os.path.abspath(path), os.path.abspath(root)))


def _locate_internal_asset(asset: dict, source_root: Path) -> tuple[Path, Path] | None:
    """
    For an asset stored inside the source project, returns (existing source file, path relative to
    the project root), falling back to the project's CAS copy. Returns None for external assets,
    which are referenced in place. Raises if an internal asset has no file.
    """
    path = Path(asset['path'])
    if not path.is_absolute():
        path = source_root / path
    relative = _relative_to(path, source_root)
    if relative is None:
        return None
    if path.exists():
        return path, relative

    cas_path = source_root / path_from_uid(asset['uid'])
    extension = asset.get('extension') or ''
    for candidate in (cas_path.with_suffix(extension) if extension else cas_path, cas_path):
        if candidate.exists():
            return candidate, candidate.relative_to(source_root)
    raise FileNotFoundError(f"Missing project file for asset '{asset['uid']}': {path}")


def write_subgraph(subgraph: Subgraph, target) -> dict:
    """
    Adds a snapshot's nodes and edges to a target graph. Files the source project owns are copied
    to the same relative location under the target root and the assets repointed there; external
    files (shared models, sample folders) stay referenced in place. All internal files are located
    before anything is copied, so a missing file fails the write without side effects.
    The caller saves the target graph.
    """
    target_root = Path(target.root)
    copies: dict[Path, Path] = {}
    rewrites: list[tuple[dict, Path]] = []
    for attrs in subgraph.nodes.values():
        for asset in _iter_asset_dicts(attrs):
            located = _locate_internal_asset(asset, subgraph.source_root)
            if located is None:
                continue
            source, relative = located
            destination = target_root / relative
            copies[source] = destination
            rewrites.append((asset, destination))

    for source, destination in copies.items():
        destination.parent.mkdir(parents=True, exist_ok=True)
        if source.is_dir():
            shutil.copytree(source, destination, dirs_exist_ok=True)
        else:
            shutil.copy2(source, destination)
    for asset, destination in rewrites:
        asset['path'] = str(destination)

    for node_id, attrs in subgraph.nodes.items():
        target.G.add_node(node_id, **attrs)
    for u, v, attrs in subgraph.edges:
        target.G.add_edge(u, v, **attrs)

    return {
        'element_count': len(subgraph.nodes),
        'edge_count': len(subgraph.edges),
        'copied_files': len(copies),
        'converted_exemplars': list(subgraph.converted_exemplars),
    }
