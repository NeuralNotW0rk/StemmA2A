import os
import sys
import json
import tempfile
import shutil
import unittest
from pathlib import Path

backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from utils.migrations import migrate_batch_to_group, migrate_baseline_references


class TestMigrations(unittest.TestCase):
    def test_batch_to_group_migration(self):
        # 1. Initialize temporary project directory
        tmp_dir = tempfile.mkdtemp()
        project_path = Path(tmp_dir)
        
        try:
            # 2. Write a mock legacy graph.json file containing 'batch' nodes and edges
            graph_data = {
                "project_name": "TestMigrationProject",
                "graph": {
                    "elements": {
                        "nodes": [
                            {
                                "data": {
                                    "id": "model_1",
                                    "type": "model"
                                }
                            },
                            {
                                "data": {
                                    "id": "legacy_batch_1",
                                    "type": "batch",
                                    "member_ids": ["audio_1", "audio_2"]
                                }
                            }
                        ],
                        "edges": [
                            {
                                "data": {
                                    "id": "model_1->legacy_batch_1",
                                    "source": "model_1",
                                    "target": "legacy_batch_1",
                                    "type": "batch"
                                }
                            }
                        ]
                    }
                }
            }
            
            graph_file = project_path / "graph.json"
            with open(graph_file, "w") as f:
                json.dump(graph_data, f, indent=4)
                
            # 3. Run migration
            migrate_batch_to_group(project_path)
            
            # 4. Read back and verify types
            with open(graph_file, "r") as f:
                updated_data = json.load(f)
                
            nodes = updated_data["graph"]["elements"]["nodes"]
            edges = updated_data["graph"]["elements"]["edges"]
            
            # Node verification
            model_node = next(n for n in nodes if n["data"]["id"] == "model_1")
            batch_node = next(n for n in nodes if n["data"]["id"] == "legacy_batch_1")
            self.assertEqual(model_node["data"]["type"], "model")
            self.assertEqual(batch_node["data"]["type"], "group")
            
            # Edge verification
            edge = edges[0]
            self.assertEqual(edge["data"]["type"], "group")
        finally:
            shutil.rmtree(tmp_dir)

    def test_baseline_references_migration(self):
        tmp_dir = tempfile.mkdtemp()
        project_path = Path(tmp_dir)

        try:
            baseline_path = str(project_path / "generate" / "grating_old.safetensors")
            elements = [{"address": "layer1", "kernel_type": "lora", "metadata": {"rank": 1}}]
            ind_context = {"model_id": "model_1", "baseline_elements": elements, "baseline_file_path": baseline_path}
            serialized_grating = {
                "id": "grating_phantom", "type": "grating", "base_model_id": "model_1",
                "elements": elements, "file": {"path": baseline_path, "uid": "grating_phantom"},
            }
            graph_data = {"project_name": "BaselineMigration", "graph": {"elements": {
                "nodes": [
                    {"data": {"id": "model_1", "type": "model"}},
                    {"data": {"id": "ind_1", "type": "individual", "baseline_grating_id": None,
                              "base_model_id": "model_1", "context": dict(ind_context)}},
                    {"data": {"id": "ind_2", "type": "individual", "baseline_grating_id": None,
                              "base_model_id": "model_1", "context": dict(ind_context)}},
                    {"data": {"id": "audio_1", "type": "audio",
                              "context": {"prompt": "drums", "baseline_grating": serialized_grating}}},
                    # Serialized on a remote engine: its path only exists there
                    {"data": {"id": "audio_2", "type": "audio", "context": {"individual_ids": ["ind_2"],
                              "baseline_grating": {**serialized_grating, "id": "grating_remote",
                                                   "file": {"path": "/app/data/cache/gr/grating_remote"}}}}},
                ],
                "edges": [],
            }}}
            graph_file = project_path / "graph.json"
            with open(graph_file, "w") as f:
                json.dump(graph_data, f)

            migrate_baseline_references(project_path)

            with open(graph_file, "r") as f:
                migrated = json.load(f)
            nodes = {n["data"]["id"]: n["data"] for n in migrated["graph"]["elements"]["nodes"]}
            baselines = [d for d in nodes.values() if d.get("type") == "grating"]

            # One shared, hidden baseline grating replaces every copy
            self.assertEqual(len(baselines), 1)
            baseline = baselines[0]
            self.assertTrue(baseline["context"]["is_baseline"])
            self.assertEqual(baseline["elements"], elements)
            self.assertEqual(baseline["file"]["path"], baseline_path)
            for ind_id in ("ind_1", "ind_2"):
                self.assertEqual(nodes[ind_id]["baseline_grating_id"], baseline["id"])
                self.assertNotIn("baseline_elements", nodes[ind_id]["context"])
                self.assertNotIn("baseline_file_path", nodes[ind_id]["context"])
            self.assertNotIn("baseline_grating", nodes["audio_1"]["context"])
            self.assertEqual(nodes["audio_1"]["context"]["baseline_grating_id"], baseline["id"])
            self.assertEqual(nodes["audio_1"]["context"]["prompt"], "drums")
            self.assertEqual(nodes["audio_2"]["context"]["baseline_grating_id"], baseline["id"])
            self.assertNotIn("baseline_grating", nodes["audio_2"]["context"])
            edge = migrated["graph"]["elements"]["edges"][0]["data"]
            self.assertEqual((edge["source"], edge["target"], edge["relation"]), ("model_1", baseline["id"], "binds_to"))

            # The original is kept, and a second run changes nothing
            self.assertTrue((project_path / "graph.json.pre-baseline-migration").exists())
            before = graph_file.read_bytes()
            migrate_baseline_references(project_path)
            self.assertEqual(graph_file.read_bytes(), before)
        finally:
            shutil.rmtree(tmp_dir)

    def test_empty_and_missing_graph_migration(self):
        tmp_dir = tempfile.mkdtemp()
        project_path = Path(tmp_dir)
        
        try:
            # 1. Missing graph.json
            migrate_batch_to_group(project_path)
            
            # 2. Empty graph.json (0-bytes)
            graph_file = project_path / "graph.json"
            graph_file.touch()
            migrate_batch_to_group(project_path)
            self.assertEqual(graph_file.stat().st_size, 0)
        finally:
            shutil.rmtree(tmp_dir)


if __name__ == "__main__":
    unittest.main()
