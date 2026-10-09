"""Evolutionary graph resolution and lineage extraction helpers."""

from collections import deque
from typing import Any, Iterator, Optional
import copy
import hashlib
import json
import threading
from param_graph.elements.artifacts.individual_element import Individual
from param_graph.elements.artifacts.audio_element import Audio
from param_graph.elements.artifacts.bundle_element import Bundle
from param_graph.elements.collections.group_element import Group


def extract_individual_parent_ids(
    parent_input: Any,
    param_graph: Optional[Any] = None,
    graph_lock: Optional[threading.Lock] = None,
) -> list[str]:
    """
    Extracts valid Individual node IDs from parent inputs, automatically
    unpacking groups/bundles and discarding edge IDs (e.g. containing '->') or invalid elements.

    Args:
        parent_input: Parent identifier, dictionary payload, or list of identifiers/objects.
        param_graph: Optional ParameterGraph instance to validate and unpack nodes from.
        graph_lock: Optional threading.Lock for thread-safe graph inspection.

    Returns:
        A list of resolved Individual node IDs without duplicates.
    """
    if not parent_input:
        return []
    if isinstance(parent_input, (str, dict)):
        items = [parent_input]
    elif isinstance(parent_input, list):
        items = parent_input
    else:
        return []

    resolved_ids: list[str] = []
    for item in items:
        if isinstance(item, dict):
            node_obj = item.get("node") if "node" in item else item
            if isinstance(node_obj, dict):
                if node_obj.get("source") and node_obj.get("target"):
                    continue
                pid = node_obj.get("id")
            elif isinstance(node_obj, str):
                pid = node_obj
            else:
                continue
        elif isinstance(item, str):
            pid = item
        else:
            continue

        if not pid or not isinstance(pid, str) or "->" in pid:
            continue

        if param_graph is not None:
            if graph_lock is not None:
                with graph_lock:
                    if param_graph.G.has_node(pid):
                        elem = param_graph.get_element(pid)
                        if isinstance(elem, Individual) and pid not in resolved_ids:
                            resolved_ids.append(pid)
                        elif isinstance(elem, (Group, Bundle)) and hasattr(elem, "member_ids") and elem.member_ids:
                            for mid in elem.member_ids:
                                if param_graph.G.has_node(mid):
                                    m_elem = param_graph.get_element(mid)
                                    if isinstance(m_elem, Individual) and mid not in resolved_ids:
                                        resolved_ids.append(mid)
            else:
                if param_graph.G.has_node(pid):
                    elem = param_graph.get_element(pid)
                    if isinstance(elem, Individual) and pid not in resolved_ids:
                        resolved_ids.append(pid)
                    elif isinstance(elem, (Group, Bundle)) and hasattr(elem, "member_ids") and elem.member_ids:
                        for mid in elem.member_ids:
                            if param_graph.G.has_node(mid):
                                m_elem = param_graph.get_element(mid)
                                if isinstance(m_elem, Individual) and mid not in resolved_ids:
                                    resolved_ids.append(mid)
        else:
            if pid not in resolved_ids:
                resolved_ids.append(pid)

    return resolved_ids


def find_exemplar_audios(
    individual_or_id: Any,
    param_graph: Any,
    graph_lock: Optional[threading.Lock] = None,
) -> list[Audio]:
    """
    Finds every exemplar Audio artifact associated with a given Individual node, in priority order:

    1. Child Audio nodes where node attribute `parent == individual_id` (in graph order).
    2. Shared exemplar Audio node if individual has `shared_exemplar_id` (phenotypic duplicates).
    3. Outgoing edges with `relation == "shares_phenotype"`.
    4. Incoming precursor Audio nodes with `relation == "precursor"`.
    5. Precursor audio ID recorded in `individual.context["precursor_artifact_id"]` (generation 0 only).

    Args:
        individual_or_id: Individual instance or individual node ID string.
        param_graph: ParameterGraph instance.
        graph_lock: Optional threading.Lock for graph concurrency.

    Returns:
        The resolved Audio elements without duplicates (empty if none).
    """
    if not individual_or_id or param_graph is None:
        return []

    ind_id = individual_or_id if isinstance(individual_or_id, str) else getattr(individual_or_id, "id", None)
    if not ind_id:
        return []

    def _lookup() -> list[Audio]:
        if not param_graph.G.has_node(ind_id):
            return []

        ind_elem = param_graph.get_element(ind_id)
        candidate_ids: list[str] = []

        # 1. Directly parented Audio nodes (most common)
        candidate_ids.extend(
            node_id for node_id, attrs in param_graph.G.nodes(data=True) if attrs.get("parent") == ind_id
        )

        # 2. Shared exemplar ID in context or element attributes (for phenotypic duplicates)
        if isinstance(ind_elem, Individual):
            shared_id = (
                getattr(ind_elem, "shared_exemplar_id", None)
                or (ind_elem.context or {}).get("shared_exemplar_id")
            )
            if shared_id:
                candidate_ids.append(shared_id)

        # 3. Shares phenotype edge: (individual) -> (audio) with relation='shares_phenotype'
        candidate_ids.extend(
            target_id for _, target_id, edge_data in param_graph.G.out_edges(ind_id, data=True)
            if edge_data.get("relation") == "shares_phenotype"
        )

        # 4. Precursor edge: (audio) -> (individual) with relation='precursor'
        candidate_ids.extend(
            source_id for source_id, _, edge_data in param_graph.G.in_edges(ind_id, data=True)
            if edge_data.get("relation") == "precursor"
        )

        # 5. Precursor artifact ID stored in context. Only wrapped (generation 0) individuals have a
        # precursor of their own; descendants inherit the ID through their copied context
        if isinstance(ind_elem, Individual) and ind_elem.context and not ind_elem.generation:
            precursor_id = ind_elem.context.get("precursor_artifact_id")
            if precursor_id:
                candidate_ids.append(precursor_id)

        audios: list[Audio] = []
        seen: set[str] = set()
        for node_id in candidate_ids:
            if node_id in seen or not param_graph.G.has_node(node_id):
                continue
            seen.add(node_id)
            elem = param_graph.get_element(node_id)
            if isinstance(elem, Audio):
                audios.append(elem)
        return audios

    if graph_lock is not None:
        with graph_lock:
            return _lookup()
    return _lookup()


def find_exemplar_audio(
    individual_or_id: Any,
    param_graph: Any,
    graph_lock: Optional[threading.Lock] = None,
) -> Optional[Audio]:
    """
    Finds the primary exemplar Audio artifact of an Individual: the first result of find_exemplar_audios.

    Args:
        individual_or_id: Individual instance or individual node ID string.
        param_graph: ParameterGraph instance.
        graph_lock: Optional threading.Lock for graph concurrency.

    Returns:
        The resolved Audio element if found, or None.
    """
    audios = find_exemplar_audios(individual_or_id, param_graph, graph_lock=graph_lock)
    return audios[0] if audios else None


def _lineage_parent_ids(node_id: str, element: Any, param_graph: Any) -> list[str]:
    """Returns a node's lineage parents: 'parent' graph edges, then mutation and lineage context records."""
    parent_ids = [
        u for u, _, edge_data in param_graph.G.in_edges(node_id, data=True)
        if edge_data.get("relation") == "parent"
    ]
    context = getattr(element, "context", None) or {}
    mutation_parent = context.get("mutation_operation", {}).get("parent_id")
    if mutation_parent:
        parent_ids.append(mutation_parent)
    parent_ids.extend(context.get("lineage", {}).get("parent_ids", []))
    return parent_ids


def iter_lineage(
    individual_id: str,
    param_graph: Any,
    max_depth: Optional[int] = None,
) -> Iterator[tuple[str, int]]:
    """
    Yields (node_id, depth) for an Individual (depth 0) and then its ancestors, breadth-first and
    nearest first. Ancestors are found through 'parent' edges and the mutation/lineage records in
    context; only Individual ancestors are expanded further. Callers must hold the graph lock.

    Args:
        individual_id: ID of the Individual whose lineage to walk.
        param_graph: ParameterGraph instance.
        max_depth: Maximum ancestor depth to visit (None for the whole lineage).
    """
    if not param_graph.G.has_node(individual_id):
        return
    element = param_graph.get_element(individual_id)
    if not isinstance(element, Individual):
        return

    yield individual_id, 0
    visited: set[str] = {individual_id}
    queue: deque[tuple[str, int]] = deque()
    for parent_id in _lineage_parent_ids(individual_id, element, param_graph):
        if parent_id not in visited:
            visited.add(parent_id)
            queue.append((parent_id, 1))

    while queue:
        curr_id, depth = queue.popleft()
        if max_depth is not None and depth > max_depth:
            continue
        if not param_graph.G.has_node(curr_id):
            continue

        yield curr_id, depth
        ancestor = param_graph.get_element(curr_id)
        if isinstance(ancestor, Individual):
            for parent_id in _lineage_parent_ids(curr_id, ancestor, param_graph):
                if parent_id not in visited:
                    visited.add(parent_id)
                    queue.append((parent_id, depth + 1))


def resolve_exemplar_context(
    individual_or_id: Any,
    param_graph: Any,
    graph_lock: Optional[threading.Lock] = None,
    max_ancestor_depth: int = 10,
) -> dict[str, Any]:
    """
    Resolves generative synthesis parameters for an Individual by querying the
    closest parent/ancestor exemplar audio node's context in the evolutionary lineage.

    Resolution strategy:
    1. If target individual already has an exemplar audio, returns its context.
    2. Otherwise, traverses up parent lineage (via 'parent' edges and lineage metadata)
       to find the nearest ancestor with an exemplar audio and returns that context.
    3. Falls back to precursor audio artifact context or individual context dictionary.

    Args:
        individual_or_id: Target Individual instance or individual node ID string.
        param_graph: ParameterGraph instance.
        graph_lock: Optional threading.Lock for graph concurrency.
        max_ancestor_depth: Maximum search depth when walking up the ancestor tree.

    Returns:
        A dictionary containing generative context parameters (e.g. prompt, seconds_total, truncation, etc.).
    """
    if not individual_or_id or param_graph is None:
        return {}

    ind_id = individual_or_id if isinstance(individual_or_id, str) else getattr(individual_or_id, "id", None)
    if not ind_id:
        return {}

    def _resolve() -> dict[str, Any]:
        if not param_graph.G.has_node(ind_id):
            return {}

        target_elem = param_graph.get_element(ind_id)
        if not isinstance(target_elem, Individual):
            return {}

        # 1-2. Nearest exemplar on the individual itself or its ancestors
        for node_id, _ in iter_lineage(ind_id, param_graph, max_depth=max_ancestor_depth):
            exemplar = find_exemplar_audio(node_id, param_graph)
            if exemplar and getattr(exemplar, "context", None):
                return copy.deepcopy(exemplar.context)

        # 3. Fallback to precursor artifact if reachable
        precursor_id = (target_elem.context or {}).get("precursor_artifact_id")
        if precursor_id and param_graph.G.has_node(precursor_id):
            prec_elem = param_graph.get_element(precursor_id)
            if prec_elem and getattr(prec_elem, "context", None):
                return copy.deepcopy(prec_elem.context)

        # 4. Fallback to individual's own context dict
        return copy.deepcopy(target_elem.context or {})

    if graph_lock is not None:
        with graph_lock:
            return _resolve()
    return _resolve()


def exemplar_generation_params(context: dict[str, Any], form_config: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Extracts the generation parameters an exemplar was made with, keyed like a generate request.
    Every form field gets a value (its default when the context predates the field). Node fields are
    given as '<name>_id', set to None when unused so a request does not inherit another exemplar's
    input node. Extra gratings applied on top of the genome are kept.

    Args:
        context: The exemplar Audio artifact's context.
        form_config: The model adapter's generate form fields.
    """
    params: dict[str, Any] = {}
    for field in form_config:
        name = field.get("name")
        if not name:
            continue
        if field.get("type") == "node":
            value = context.get(f"{name}_id", context.get(name))
            if isinstance(value, dict):
                value = value.get("id")
            params[f"{name}_id"] = value or None
        else:
            params[name] = context.get(name, field.get("defaultValue"))
    if context.get("gratings"):
        params["gratings"] = context["gratings"]
    return params


def collect_exemplar_presets(
    individual_ids: list[str],
    param_graph: Any,
    form_config: list[dict[str, Any]],
    graph_lock: Optional[threading.Lock] = None,
) -> list[dict[str, Any]]:
    """
    Collects the distinct generation settings of all exemplars across the given Individuals and their
    ancestors, so the same generations can be repeated on those Individuals.

    Exemplars with identical generation parameters are merged into one preset. Presets are ordered
    nearest lineage first, then most recent first.

    Args:
        individual_ids: IDs of the target Individuals.
        param_graph: ParameterGraph instance.
        form_config: The model adapter's generate form fields, defining which context keys are parameters.
        graph_lock: Optional threading.Lock for graph concurrency.

    Returns:
        A list of presets, each a dict with:
            key: Stable identifier of the parameter set.
            params: Generation parameters (see exemplar_generation_params).
            sources: Exemplars with these parameters ({audio_id, audio_name, individual_id,
                individual_name, distance}), where distance is the number of generations between the
                exemplar's individual and the nearest target.
            covered_ids: Target Individuals that already have an exemplar with these parameters.
    """
    def _collect() -> list[dict[str, Any]]:
        presets: dict[str, dict[str, Any]] = {}
        # Per preset: (nearest distance, latest exemplar creation time), for ordering
        order: dict[str, tuple[int, float]] = {}

        for target_id in individual_ids:
            for node_id, depth in iter_lineage(target_id, param_graph):
                node = param_graph.get_element(node_id)
                for audio in find_exemplar_audios(node_id, param_graph):
                    if not audio.context:
                        continue
                    params = exemplar_generation_params(audio.context, form_config)
                    signature = json.dumps(params, sort_keys=True, default=str)
                    key = hashlib.sha1(signature.encode("utf-8")).hexdigest()[:16]

                    preset = presets.setdefault(key, {"key": key, "params": params, "sources": [], "covered_ids": []})
                    if depth == 0 and target_id not in preset["covered_ids"]:
                        preset["covered_ids"].append(target_id)

                    source = next((s for s in preset["sources"] if s["audio_id"] == audio.id), None)
                    if source is None:
                        preset["sources"].append({
                            "audio_id": audio.id,
                            "audio_name": getattr(audio, "alias", None) or audio.name,
                            "individual_id": node_id,
                            "individual_name": getattr(node, "alias", None) or getattr(node, "name", node_id),
                            "distance": depth,
                        })
                    elif depth < source["distance"]:
                        source["distance"] = depth

                    created = float(getattr(audio, "created", 0) or 0)
                    nearest, latest = order.get(key, (depth, created))
                    order[key] = (min(nearest, depth), max(latest, created))

        for preset in presets.values():
            preset["sources"].sort(key=lambda s: s["distance"])
        return sorted(presets.values(), key=lambda p: (order[p["key"]][0], -order[p["key"]][1]))

    if graph_lock is not None:
        with graph_lock:
            return _collect()
    return _collect()
