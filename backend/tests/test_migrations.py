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

from utils.migrations import migrate_batch_to_group, migrate_baseline_references, migrate_genome_encoding


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

    def test_genome_encoding_migration(self):
        import torch
        from safetensors import safe_open
        from safetensors.torch import save_file
        from diffracture.topology.grating import Grating as DiffractureGrating
        from diffracture.topology.lora import LoRAElement
        from evolution.lora_genome import GENOME_FORMAT, LoRAGenome, express_to_grating
        from utils.uid import XXH3_64

        tmp_dir = tempfile.mkdtemp()
        project_path = Path(tmp_dir)
        generate = project_path / "generate"
        generate.mkdir()

        try:
            baseline = DiffractureGrating()
            baseline.add_element(LoRAElement("l1", rank=2, alpha=2.0, in_features=8, out_features=4))
            baseline_path = generate / "base.safetensors"
            baseline.save(str(baseline_path))

            # Legacy genomes store phenotype weights directly
            down, up = torch.randn(2, 8) * 3.0, torch.randn(4, 2)
            legacy_meta = {"addresses": json.dumps(["l1"]), "gene_metadata": json.dumps({"l1": {"active": True}})}
            save_file({"l1.lora_down": down, "l1.lora_up": up}, str(generate / "ind_a.safetensors"), metadata=legacy_meta)
            save_file({"l1.lora_down": down * 2.0, "l1.lora_up": up}, str(generate / "ind_orphan.safetensors"), metadata=legacy_meta)

            def file_info(uid):
                # Stored paths may be stale (e.g. from another machine), so the uid fallback is exercised
                return {"path": f"/elsewhere/{uid}.safetensors", "uid": uid, "extension": ".safetensors"}

            graph_data = {"graph": {"elements": {"nodes": [
                {"data": {"id": "base", "type": "grating", "file": file_info("base")}},
                {"data": {"id": "ind_a", "type": "individual", "baseline_grating_id": "base", "file": file_info("ind_a"), "value": "ind_a"}},
                {"data": {"id": "ind_orphan", "type": "individual", "baseline_grating_id": None, "file": file_info("ind_orphan")}},
                {"data": {"id": "grp", "type": "group", "member_ids": ["ind_a", "ind_orphan"]}},
                {"data": {"id": "aud", "type": "audio", "context": {"individuals": [{"id": "ind_a", "strength": 1.0}]}}},
            ], "edges": [
                {"data": {"id": "base->ind_a", "source": "base", "target": "ind_a", "relation": "parent"}},
            ]}}}
            graph_file = project_path / "graph.json"
            graph_file.write_text(json.dumps(graph_data), encoding="utf-8")

            migrate_genome_encoding(project_path)

            migrated = json.loads(graph_file.read_text(encoding="utf-8"))["graph"]["elements"]
            nodes = {n["data"]["id"]: n["data"] for n in migrated["nodes"]}
            self.assertNotIn("ind_a", nodes)
            new_id = next(i for i, d in nodes.items() if d["type"] == "individual" and d["baseline_grating_id"] == "base")

            # The converted genome is stored under the hash of its new content, and the old file is gone
            converted = generate / f"{new_id}.safetensors"
            self.assertFalse((generate / "ind_a.safetensors").exists())
            self.assertEqual(nodes[new_id]["file"]["uid"], new_id)
            self.assertEqual(nodes[new_id]["value"], new_id)
            self.assertEqual(Path(nodes[new_id]["file"]["path"]), converted)
            genome = LoRAGenome.load(str(converted))
            self.assertEqual(XXH3_64().from_state_dict(genome.get_state_dict()), new_id)
            with safe_open(str(converted), framework="pt") as f:
                self.assertEqual(f.metadata()["format"], GENOME_FORMAT)

            # It carries the baseline's topology, and expresses back to its original weights without the baseline
            self.assertEqual(genome[0].topology["alpha"], 2.0)
            expressed = express_to_grating(genome).nodes["l1"]
            self.assertEqual(expressed.metadata["alpha"], 2.0)
            self.assertTrue(torch.allclose(expressed.params["lora_down"], down, atol=1e-6))
            self.assertTrue(torch.allclose(expressed.params["lora_up"], up, atol=1e-6))

            # Without a baseline there is no topology to take, so the genome is left as it was
            with safe_open(str(generate / "ind_orphan.safetensors"), framework="pt") as f:
                self.assertIsNone(f.metadata().get("format"))

            # Every reference follows the re-addressed genome
            self.assertEqual(nodes["grp"]["member_ids"], [new_id, "ind_orphan"])
            self.assertEqual(nodes["aud"]["context"]["individuals"][0]["id"], new_id)
            self.assertEqual(migrated["edges"][0]["data"]["id"], f"base->{new_id}")
            self.assertEqual(migrated["edges"][0]["data"]["target"], new_id)
            self.assertTrue((project_path / "graph.json.pre-genome-migration").exists())

            # A second run changes nothing
            before = graph_file.read_bytes(), converted.read_bytes()
            migrate_genome_encoding(project_path)
            self.assertEqual((graph_file.read_bytes(), converted.read_bytes()), before)
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
