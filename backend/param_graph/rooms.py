"""
Rooms: separate views of parts of one project graph.

A node belongs to the room named by its `room` attribute, unless it is nested in a compound
(an exemplar in its individual, a member in its group), in which case it is wherever its
top-level ancestor is. Nodes without a room are in the main graph. Rooms are themselves nodes
and can nest. Breaking out moves nodes into a new room and dissolving moves them back; nothing
is copied, so edges, lineage and operations work across room boundaries unchanged.
"""
import uuid
from contextvars import ContextVar

import networkx as nx

from .elements.collections.room_element import Room

# The room new top-level nodes are placed in. Set per request (and carried into worker threads
# by copying the context), so a job finishing after the user changed views still lands where
# it was started
placement_room: ContextVar[str | None] = ContextVar('placement_room', default=None)


def is_room(G: nx.DiGraph, node_id: str | None) -> bool:
    return bool(node_id) and G.has_node(node_id) and G.nodes[node_id].get('type') == 'room'


def top_level(G: nx.DiGraph, node_id: str) -> str:
    """The outermost compound containing a node (the node itself if it has no parent)."""
    seen = {node_id}
    parent = G.nodes[node_id].get('parent')
    while parent and G.has_node(parent) and parent not in seen:
        node_id = parent
        seen.add(node_id)
        parent = G.nodes[node_id].get('parent')
    return node_id


def room_of(G: nx.DiGraph, node_id: str) -> str | None:
    """The room a node is shown in, or None for the main graph. Dangling room IDs mean main."""
    room = G.nodes[top_level(G, node_id)].get('room')
    return room if is_room(G, room) else None


def enclosing_rooms(G: nx.DiGraph, node_id: str) -> list[str]:
    """Every room containing a node, innermost first."""
    rooms: list[str] = []
    room = room_of(G, node_id)
    while room and room not in rooms:
        rooms.append(room)
        room = room_of(G, room)
    return rooms


def _movable_root(G: nx.DiGraph, node_id: str) -> str:
    """
    The node that has to move for this one to change rooms. Exemplars, expressed gratings and
    directory entries cannot leave their compound, so the compound moves; group membership is
    looser and is dropped instead (see break_out).
    """
    parent = G.nodes[node_id].get('parent')
    while parent and G.has_node(parent) and G.nodes[parent].get('type') != 'group':
        node_id = parent
        parent = G.nodes[node_id].get('parent')
    return node_id


def break_out(graph, element_ids: list[str], name: str) -> str:
    """
    Moves elements out of their current view into a new room placed in that view. Selected
    groups bring their members; a group member selected without its group leaves the group.
    Returns the new room's ID.
    """
    G: nx.DiGraph = graph.G
    missing = [eid for eid in element_ids if not G.has_node(eid)]
    if missing:
        raise ValueError(f"Elements not found in the graph: {missing}")

    roots = list(dict.fromkeys(_movable_root(G, eid) for eid in element_ids))
    views = {room_of(G, root) for root in roots}
    if len(views) > 1:
        raise ValueError("Elements to break out must all be in the same view.")
    view = views.pop()

    room = Room(id=f"room_{uuid.uuid4().hex[:8]}", name=name, room=view)
    graph.add_element(room)
    # The room sits where its members were, so the main view keeps its shape
    positions = [G.nodes[r]['position'] for r in roots if isinstance(G.nodes[r].get('position'), dict)]
    if positions:
        graph.update_element(room.id, {'position': {
            'x': sum(p['x'] for p in positions) / len(positions),
            'y': sum(p['y'] for p in positions) / len(positions),
        }})

    for root in roots:
        parent = G.nodes[root].get('parent')
        if parent and G.has_node(parent) and parent not in roots:
            # Leaving a group that stays behind
            group = G.nodes[parent]
            if isinstance(group.get('member_ids'), list):
                group['member_ids'] = [m for m in group['member_ids'] if m != root]
            G.nodes[root].pop('parent')
            G.nodes[root].pop('alias', None)
        G.nodes[root]['room'] = room.id
    return room.id


def release_members(G: nx.DiGraph, room_id: str) -> list[str]:
    """Moves a room's direct members (including nested rooms) into the room's own view."""
    destination = G.nodes[room_id].get('room')
    members = [n for n, d in G.nodes(data=True) if d.get('room') == room_id and n != room_id]
    for member in members:
        if destination:
            G.nodes[member]['room'] = destination
        else:
            G.nodes[member].pop('room')
    return members


def dissolve(graph, room_id: str) -> list[str]:
    """Rejoins a room's members with the view the room sits in and removes the room."""
    if not is_room(graph.G, room_id):
        raise ValueError(f"'{room_id}' is not a room.")
    members = release_members(graph.G, room_id)
    graph.G.remove_node(room_id)
    return members
