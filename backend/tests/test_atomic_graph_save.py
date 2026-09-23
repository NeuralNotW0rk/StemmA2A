import os
import sys
import json
import shutil
import tempfile
import unittest
import numpy as np
from pathlib import Path

backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from param_graph.graph import ParameterGraph


class TestAtomicGraphSave(unittest.TestCase):
    def test_atomic_save_and_backup(self):
        tmp_dir = tempfile.mkdtemp()
        
        try:
            pg = ParameterGraph(tmp_dir)
            pg.project_name = "AtomicTestProject"
            
            # 1. Initial save
            pg.save()
            
            graph_file = Path(tmp_dir) / "graph.json"
            bak_file = Path(tmp_dir) / "graph.json.bak"
            tmp_file = Path(tmp_dir) / "graph.json.tmp"
            
            self.assertTrue(graph_file.exists(), "graph.json must exist after save")
            self.assertGreater(graph_file.stat().st_size, 0, "graph.json must not be empty")
            self.assertFalse(tmp_file.exists(), ".tmp file must be cleaned up after replace")
            
            # 2. Second save - should generate .bak file of previous version
            pg.G.add_node("node_test_1", type="audio", name="Test Node 1")
            pg.save()
            
            self.assertTrue(bak_file.exists(), "graph.json.bak must be created on update")
            self.assertGreater(bak_file.stat().st_size, 0, "graph.json.bak must not be empty")
            
            # Verify graph.json has node_test_1
            pg2 = ParameterGraph(tmp_dir)
            self.assertTrue(pg2.load())
            self.assertTrue(pg2.G.has_node("node_test_1"))
            
            # 3. Simulate corruption of graph.json - should recover from .bak
            with open(graph_file, "w") as f:
                f.write("{ invalid json")
                
            pg3 = ParameterGraph(tmp_dir)
            self.assertTrue(pg3.load(), "Must recover from backup on corruption")
            
            # 4. Simulate 0-byte graph.json - should recover from .bak
            with open(graph_file, "w") as f:
                pass
            self.assertEqual(graph_file.stat().st_size, 0)
            
            pg4 = ParameterGraph(tmp_dir)
            self.assertTrue(pg4.load(), "Must recover from backup on 0-byte file")
        finally:
            shutil.rmtree(tmp_dir)

    def test_safe_json_serialization(self):
        tmp_dir = tempfile.mkdtemp()
        
        try:
            pg = ParameterGraph(tmp_dir)
            pg.project_name = "SerializationTest"
            
            # Add complex types (NumPy scalars, NumPy arrays, Sets, Paths)
            pg.G.add_node(
                "complex_node",
                type="audio",
                np_float=np.float32(3.14159),
                np_int=np.int64(42),
                np_arr=np.array([1.0, 2.0, 3.0]),
                custom_set={"a", "b", "c"},
                custom_path=Path("some/test/path.wav")
            )
            
            # Should not raise TypeError during save
            pg.save()
            
            graph_file = Path(tmp_dir) / "graph.json"
            self.assertTrue(graph_file.exists())
            self.assertGreater(graph_file.stat().st_size, 0)
            
            pg_loaded = ParameterGraph(tmp_dir)
            self.assertTrue(pg_loaded.load())
            node_data = pg_loaded.G.nodes["complex_node"]
            
            self.assertAlmostEqual(node_data["np_float"], 3.14159, places=4)
            self.assertEqual(node_data["np_int"], 42)
            self.assertEqual(node_data["np_arr"], [1.0, 2.0, 3.0])
            self.assertEqual(sorted(node_data["custom_set"]), ["a", "b", "c"])
            self.assertIn("path.wav", node_data["custom_path"])
        finally:
            shutil.rmtree(tmp_dir)


if __name__ == "__main__":
    unittest.main()
