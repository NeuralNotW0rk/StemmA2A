import os
import json
import tempfile
import shutil
from pathlib import Path
from utils.migrations import migrate_batch_to_group

def test_batch_to_group_migration():
    print("=== Testing migrate_batch_to_group Migration ===")
    
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
        assert model_node["data"]["type"] == "model"
        assert batch_node["data"]["type"] == "group", f"Expected type 'group', got '{batch_node['data']['type']}'"
        
        # Edge verification
        edge = edges[0]
        assert edge["data"]["type"] == "group", f"Expected edge type 'group', got '{edge['data']['type']}'"
        
        print("Legacy batch to group migration tested and verified successfully!")
        
    finally:
        shutil.rmtree(tmp_dir)

def test_empty_and_missing_graph_migration():
    print("=== Testing migrate_batch_to_group on Empty / Missing Files ===")
    tmp_dir = tempfile.mkdtemp()
    project_path = Path(tmp_dir)
    
    try:
        # 1. Missing graph.json
        migrate_batch_to_group(project_path)
        
        # 2. Empty graph.json (0-bytes)
        graph_file = project_path / "graph.json"
        graph_file.touch()
        migrate_batch_to_group(project_path)
        assert graph_file.stat().st_size == 0
        
        print("Empty and missing graph files handled cleanly without errors!")
    finally:
        shutil.rmtree(tmp_dir)

if __name__ == "__main__":
    test_batch_to_group_migration()
    test_empty_and_missing_graph_migration()
