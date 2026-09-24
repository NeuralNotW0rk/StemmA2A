"""Evolutionary graph resolution and lineage extraction helpers."""

from typing import Any, Optional
import threading
from param_graph.elements.artifacts.individual_element import Individual
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
