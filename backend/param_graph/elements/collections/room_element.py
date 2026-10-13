from dataclasses import dataclass

from param_graph.elements.base_elements import GraphElement
from param_graph.registry import register


@register('room')
@dataclass(kw_only=True)
class Room(GraphElement):
    """
    A separate view of part of the graph. Elements join a room through their `room` attribute
    (compound children follow their parent); the room node itself sits in the view of the room
    it was broken out from, recorded in its own `room` attribute (None for the main graph).
    """
    name: str
    room: str | None = None
    type: str = 'room'
