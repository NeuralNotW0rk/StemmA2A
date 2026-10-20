import sys
import tempfile
import threading
import unittest
from pathlib import Path

backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

import param_graph.elements  # noqa: F401  (registers element types)
from param_graph.graph import ParameterGraph
from param_graph.rooms import placement_room, room_of, enclosing_rooms, break_out, dissolve
from param_graph.elements.artifacts.audio_element import Audio
from param_graph.elements.base_elements import Asset
from operations.task_manager import TaskManager


class TestRooms(unittest.TestCase):
    def setUp(self):
        self.graph = ParameterGraph(tempfile.mkdtemp())
        G = self.graph.G
        G.add_node("model", id="model", type="model", position={"x": 0, "y": 0})
        G.add_node("precursor", id="precursor", type="audio", position={"x": 100, "y": 0})
        G.add_node("ind", id="ind", type="individual", position={"x": 200, "y": 0})
        G.add_node("exemplar", id="exemplar", type="audio", parent="ind", position={"x": 300, "y": 0})
        G.add_node("group", id="group", type="group", member_ids=["child_a", "child_b"])
        G.add_node("child_a", id="child_a", type="individual", parent="group")
        G.add_node("child_b", id="child_b", type="individual", parent="group")
        G.add_edge("model", "precursor", relation="source")
        G.add_edge("precursor", "ind", relation="precursor")

    def tearDown(self):
        placement_room.set(None)

    def test_break_out_moves_selection_into_room_in_current_view(self):
        room = break_out(self.graph, ["precursor", "ind"], name="session")
        G = self.graph.G

        self.assertEqual(G.nodes[room]["type"], "room")
        self.assertIsNone(G.nodes[room]["room"])
        self.assertEqual(G.nodes[room]["position"], {"x": 150, "y": 0})
        self.assertEqual(room_of(G, "precursor"), room)
        self.assertEqual(room_of(G, "ind"), room)
        # Exemplars follow their individual; everything else stays in main
        self.assertEqual(room_of(G, "exemplar"), room)
        self.assertIsNone(room_of(G, "model"))
        # Nothing structural changes
        self.assertTrue(G.has_edge("model", "precursor"))
        self.assertEqual(G.nodes["exemplar"]["parent"], "ind")

    def test_selected_exemplar_brings_its_individual(self):
        room = break_out(self.graph, ["exemplar"], name="session")
        self.assertEqual(self.graph.G.nodes["ind"]["room"], room)
        self.assertNotIn("room", self.graph.G.nodes["exemplar"])

    def test_group_member_leaves_group_that_stays_behind(self):
        room = break_out(self.graph, ["child_a"], name="session")
        G = self.graph.G
        self.assertEqual(room_of(G, "child_a"), room)
        self.assertNotIn("parent", G.nodes["child_a"])
        self.assertEqual(G.nodes["group"]["member_ids"], ["child_b"])
        self.assertIsNone(room_of(G, "child_b"))

    def test_selected_group_moves_with_members(self):
        room = break_out(self.graph, ["group"], name="session")
        G = self.graph.G
        self.assertEqual(room_of(G, "child_a"), room)
        self.assertEqual(G.nodes["child_a"]["parent"], "group")

    def test_nested_rooms_and_dissolve(self):
        outer = break_out(self.graph, ["precursor", "ind"], name="outer")
        inner = break_out(self.graph, ["ind"], name="inner")
        G = self.graph.G
        self.assertEqual(G.nodes[inner]["room"], outer)
        self.assertEqual(enclosing_rooms(G, "exemplar"), [inner, outer])

        with self.assertRaises(ValueError):
            break_out(self.graph, ["precursor", "model"], name="mixed")

        dissolve(self.graph, inner)
        self.assertFalse(G.has_node(inner))
        self.assertEqual(room_of(G, "ind"), outer)
        dissolve(self.graph, outer)
        self.assertIsNone(room_of(G, "ind"))
        self.assertNotIn("room", G.nodes["ind"])

    def test_removing_a_room_keeps_its_members(self):
        room = break_out(self.graph, ["precursor"], name="session")
        self.graph.remove_element(room)
        self.assertTrue(self.graph.G.has_node("precursor"))
        self.assertIsNone(room_of(self.graph.G, "precursor"))

    def test_new_elements_land_in_the_placement_room(self):
        room = break_out(self.graph, ["precursor"], name="session")
        audio = Audio(id="new", name="new", context={}, file=Asset(path="", uid="new"))

        placement_room.set(room)
        self.graph.add_element(audio)
        self.assertEqual(room_of(self.graph.G, "new"), room)

        # Re-adding an existing node never moves it
        placement_room.set(None)
        self.graph.add_element(audio)
        self.assertEqual(room_of(self.graph.G, "new"), room)

    def test_placement_carries_into_task_manager_workers(self):
        room = break_out(self.graph, ["precursor"], name="session")
        placement_room.set(room)
        seen = {}
        done = threading.Event()

        def task():
            seen["room"] = placement_room.get()
            done.set()

        TaskManager().submit_task(task, name="placement test")
        self.assertTrue(done.wait(5))
        self.assertEqual(seen["room"], room)

    def test_rooms_survive_save_and_load(self):
        room = break_out(self.graph, ["precursor"], name="session")
        self.graph.save()
        reloaded = ParameterGraph(str(self.graph.root))
        self.assertTrue(reloaded.load())
        self.assertEqual(room_of(reloaded.G, "precursor"), room)
        self.assertEqual(reloaded.get_element(room).name, "session")


if __name__ == "__main__":
    unittest.main()
