"""Evolutionary graph resolution and lineage extraction helpers."""

from typing import Any, Optional
import copy
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


def find_exemplar_audio(
    individual_or_id: Any,
    param_graph: Any,
    graph_lock: Optional[threading.Lock] = None,
) -> Optional[Audio]:
    """
    Finds the exemplar Audio artifact associated with a given Individual node.

    Resolution strategy:
    1. Child Audio node where node attribute `parent == individual_id`.
    2. Shared exemplar Audio node if individual has `shared_exemplar_id` or `phenotype_duplicate_of`.
    3. Outgoing edge with `relation == "shares_phenotype"`.
    4. Incoming precursor Audio node with `relation == "precursor"`.
    5. Precursor audio ID recorded in `individual.context["precursor_artifact_id"]`.

    Args:
        individual_or_id: Individual instance or individual node ID string.
        param_graph: ParameterGraph instance.
        graph_lock: Optional threading.Lock for graph concurrency.

    Returns:
        The resolved Audio element if found, or None.
    """
    if not individual_or_id or param_graph is None:
        return None

    ind_id = individual_or_id if isinstance(individual_or_id, str) else getattr(individual_or_id, "id", None)
    if not ind_id:
        return None

    def _lookup() -> Optional[Audio]:
        if not param_graph.G.has_node(ind_id):
            return None

        ind_elem = param_graph.get_element(ind_id)

        # 1. Direct parented Audio node (most common)
        for node_id, attrs in param_graph.G.nodes(data=True):
            if attrs.get("parent") == ind_id:
                elem = param_graph.get_element(node_id)
                if isinstance(elem, Audio):
                    return elem

        # 2. Shared exemplar ID in context or element attributes (for phenotypic duplicates)
        shared_id = None
        if isinstance(ind_elem, Individual):
            shared_id = (
                getattr(ind_elem, "shared_exemplar_id", None)
                or (ind_elem.context or {}).get("shared_exemplar_id")
            )
        if shared_id and param_graph.G.has_node(shared_id):
            elem = param_graph.get_element(shared_id)
            if isinstance(elem, Audio):
                return elem

        # 3. Shares phenotype edge: (individual) -> (audio) with relation='shares_phenotype'
        for _, target_id, edge_data in param_graph.G.out_edges(ind_id, data=True):
            if edge_data.get("relation") == "shares_phenotype" and param_graph.G.has_node(target_id):
                elem = param_graph.get_element(target_id)
                if isinstance(elem, Audio):
                    return elem

        # 4. Precursor edge: (audio) -> (individual) with relation='precursor'
        for source_id, _, edge_data in param_graph.G.in_edges(ind_id, data=True):
            if edge_data.get("relation") == "precursor" and param_graph.G.has_node(source_id):
                elem = param_graph.get_element(source_id)
                if isinstance(elem, Audio):
                    return elem

        # 5. Precursor artifact ID stored in context
        if isinstance(ind_elem, Individual) and ind_elem.context:
            precursor_id = ind_elem.context.get("precursor_artifact_id")
            if precursor_id and param_graph.G.has_node(precursor_id):
                elem = param_graph.get_element(precursor_id)
                if isinstance(elem, Audio):
                    return elem

        return None

    if graph_lock is not None:
        with graph_lock:
            return _lookup()
    return _lookup()


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

        # 1. Check if the individual itself already has an exemplar audio
        own_exemplar = find_exemplar_audio(ind_id, param_graph)
        if own_exemplar and getattr(own_exemplar, "context", None):
            return copy.deepcopy(own_exemplar.context)

        # 2. Lineage queue for BFS traversal up the parent graph
        visited: set[str] = {ind_id}
        queue: list[tuple[str, int]] = []

        # Find direct parent IDs
        direct_parents: list[str] = []
        for u, _, edge_data in param_graph.G.in_edges(ind_id, data=True):
            if edge_data.get("relation") == "parent" and u not in visited:
                direct_parents.append(u)

        # Fallback to context lineage if no graph edges
        if target_elem.context:
            lineage_parents = target_elem.context.get("lineage", {}).get("parent_ids", [])
            mutation_parent = target_elem.context.get("mutation_operation", {}).get("parent_id")
            if mutation_parent and mutation_parent not in direct_parents:
                direct_parents.append(mutation_parent)
            for p in lineage_parents:
                if p not in direct_parents:
                    direct_parents.append(p)

        for p in direct_parents:
            queue.append((p, 1))
            visited.add(p)

        while queue:
            curr_id, depth = queue.pop(0)
            if depth > max_ancestor_depth:
                continue

            if not param_graph.G.has_node(curr_id):
                continue

            # Look for exemplar on this ancestor
            ancestor_audio = find_exemplar_audio(curr_id, param_graph)
            if ancestor_audio and getattr(ancestor_audio, "context", None):
                return copy.deepcopy(ancestor_audio.context)

            # If ancestor is an Individual without an exemplar, queue its parents
            ancestor_elem = param_graph.get_element(curr_id)
            if isinstance(ancestor_elem, Individual):
                for u, _, edge_data in param_graph.G.in_edges(curr_id, data=True):
                    if edge_data.get("relation") == "parent" and u not in visited:
                        visited.add(u)
                        queue.append((u, depth + 1))
                if ancestor_elem.context:
                    lineage_parents = ancestor_elem.context.get("lineage", {}).get("parent_ids", [])
                    mut_p = ancestor_elem.context.get("mutation_operation", {}).get("parent_id")
                    if mut_p and mut_p not in visited:
                        visited.add(mut_p)
                        queue.append((mut_p, depth + 1))
                    for p in lineage_parents:
                        if p not in visited:
                            visited.add(p)
                            queue.append((p, depth + 1))

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

