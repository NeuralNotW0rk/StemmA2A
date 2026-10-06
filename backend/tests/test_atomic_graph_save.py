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

    def test_add_element_allow_duplicates_control(self):
        tmp_dir = tempfile.mkdtemp()
        try:
            from param_graph.elements.artifacts.audio_element import Audio
            from param_graph.elements.base_elements import Asset

            pg = ParameterGraph(tmp_dir)
            
            elem1 = Audio(
                id="audio_dup_test",
                name="Audio Original",
                context={},
                file=Asset(path="original.wav", uid="audio_dup_test", extension=".wav")
            )
            # First add should succeed
            res1 = pg.add_element(elem1, allow_duplicates=False)
            self.assertTrue(res1)
            self.assertEqual(pg.G.nodes["audio_dup_test"]["name"], "Audio Original")

            # Second add with allow_duplicates=False should return False and NOT overwrite
            elem2 = Audio(
                id="audio_dup_test",
                name="Audio Overwrite Attempt",
                context={},
                file=Asset(path="overwritten.wav", uid="audio_dup_test", extension=".wav")
            )
            res2 = pg.add_element(elem2, allow_duplicates=False)
            self.assertFalse(res2)
            self.assertEqual(pg.G.nodes["audio_dup_test"]["name"], "Audio Original")

            # Add with allow_duplicates=True (default) should update the node
            res3 = pg.add_element(elem2, allow_duplicates=True)
            self.assertTrue(res3)
            self.assertEqual(pg.G.nodes["audio_dup_test"]["name"], "Audio Overwrite Attempt")
        finally:
            shutil.rmtree(tmp_dir)

    def test_concurrent_save_artifact_asset_no_collision(self):
        """Verify that concurrent threads calling save_artifact_asset with identical names produce distinct files without data loss."""
        import concurrent.futures
        from param_graph.utils import save_artifact_asset
        from param_graph.elements.artifacts.image_element import Image
        from param_graph.elements.base_elements import Asset

        tmp_dir = Path(tempfile.mkdtemp())
        src_dir = Path(tempfile.mkdtemp())
        output_dir = tmp_dir / "generate"

        try:
            num_workers = 16
            artifacts_to_save = []

            for i in range(num_workers):
                src_file = src_dir / f"temp_{i}.png"
                content = f"unique_image_content_{i}".encode("utf-8")
                src_file.write_bytes(content)

                img = Image(
                    id=f"img_uid_{i}",
                    name="stylegan_gen_batch_seed",
                    context={"factor": i * 0.25},
                    file=Asset(path=str(src_file), uid=f"img_uid_{i}", extension=".png")
                )
                artifacts_to_save.append((img, content))

            saved_artifacts = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = [
                    executor.submit(save_artifact_asset, item[0], output_dir, "file")
                    for item in artifacts_to_save
                ]
                for f in concurrent.futures.as_completed(futures):
                    saved_artifacts.append(f.result())

            # 1. Verify every artifact returned a unique path
            saved_paths = [a.file.path for a in saved_artifacts]
            self.assertEqual(len(saved_paths), num_workers)
            self.assertEqual(len(set(saved_paths)), num_workers, "All saved file paths must be distinct")

            # 2. Verify all files physically exist on disk
            for path_str in saved_paths:
                p = Path(path_str)
                self.assertTrue(p.exists(), f"Saved file {p} must exist on disk")

            # 3. Verify content was not corrupted or overwritten across instances
            saved_contents = [Path(p).read_bytes() for p in saved_paths]
            expected_contents = [item[1] for item in artifacts_to_save]
            self.assertEqual(sorted(saved_contents), sorted(expected_contents), "All unique file contents must be preserved")

        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            shutil.rmtree(src_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()

