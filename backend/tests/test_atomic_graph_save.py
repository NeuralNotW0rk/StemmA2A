import os
import json
import shutil
import tempfile
import numpy as np
from pathlib import Path
from param_graph.graph import ParameterGraph

def test_atomic_save_and_backup():
    print("=== Testing Atomic Save and Rolling Backup ===")
    tmp_dir = tempfile.mkdtemp()
    
    try:
        pg = ParameterGraph(tmp_dir)
        pg.project_name = "AtomicTestProject"
        
        # 1. Initial save
        pg.save()
        
        graph_file = Path(tmp_dir) / "graph.json"
        bak_file = Path(tmp_dir) / "graph.json.bak"
        tmp_file = Path(tmp_dir) / "graph.json.tmp"
        
        assert graph_file.exists(), "graph.json must exist after save"
        assert graph_file.stat().st_size > 0, "graph.json must not be empty"
        assert not tmp_file.exists(), ".tmp file must be cleaned up after replace"
        
        # 2. Second save - should generate .bak file of previous version
        pg.G.add_node("node_test_1", type="audio", name="Test Node 1")
        pg.save()
        
        assert bak_file.exists(), "graph.json.bak must be created on update"
        assert bak_file.stat().st_size > 0, "graph.json.bak must not be empty"
        
        # Verify graph.json has node_test_1
        pg2 = ParameterGraph(tmp_dir)
        assert pg2.load() is True
        assert pg2.G.has_node("node_test_1")
        
        # 3. Simulate corruption of graph.json - should recover from .bak
        with open(graph_file, "w") as f:
            f.write("{ invalid json")
            
        pg3 = ParameterGraph(tmp_dir)
        assert pg3.load() is True, "Must recover from backup on corruption"
        
        # 4. Simulate 0-byte graph.json - should recover from .bak
        with open(graph_file, "w") as f:
            pass
        assert graph_file.stat().st_size == 0
        
        pg4 = ParameterGraph(tmp_dir)
        assert pg4.load() is True, "Must recover from backup on 0-byte file"
        
        print("Atomic save and rolling backup verified successfully!")
    finally:
        shutil.rmtree(tmp_dir)

def test_safe_json_serialization():
    print("=== Testing Safe JSON Serialization ===")
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
        assert graph_file.exists()
        assert graph_file.stat().st_size > 0
        
        pg_loaded = ParameterGraph(tmp_dir)
        assert pg_loaded.load() is True
        node_data = pg_loaded.G.nodes["complex_node"]
        
        assert abs(node_data["np_float"] - 3.14159) < 1e-4
        assert node_data["np_int"] == 42
        assert node_data["np_arr"] == [1.0, 2.0, 3.0]
        assert sorted(node_data["custom_set"]) == ["a", "b", "c"]
        assert "path.wav" in node_data["custom_path"]
        
        print("Safe JSON serialization verified successfully!")
    finally:
        shutil.rmtree(tmp_dir)

if __name__ == "__main__":
    test_atomic_save_and_backup()
    test_safe_json_serialization()
