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

from utils.migrations import migrate_batch_to_group


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
