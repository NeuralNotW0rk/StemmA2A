import json
import sys
import shutil
import tempfile
import unittest
from pathlib import Path

backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))
sys.path.insert(0, str(backend_dir / "scripts"))

from migrate_directory_uids import BACKUP_NAME, legacy_directory_uid, migrate
from utils.uid import XXH3_64

CKPT = "ckpt.xxh3_64"


class TestMigrateDirectoryUids(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.encoder = self.root / "encoder"
        self.encoder.mkdir()
        # Spaced JSON hashes differently once from_directory canonicalizes it
        (self.encoder / "config.json").write_text('{\n  "b": 1,\n  "a": 2\n}', encoding="utf-8")
        (self.encoder / "model.safetensors").write_bytes(b"weights")
        self.uid_gen = XXH3_64()
        self.legacy = legacy_directory_uid(self.encoder)
        self.current = self.uid_gen.from_directory(self.encoder)
        self.assertNotEqual(self.legacy, self.current)
        self.project = self.root / "project"
        self.project.mkdir()

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def _write_project(self, encoder_uid: str) -> str:
        model_id = self.uid_gen.from_uids([CKPT, encoder_uid])
        nodes = [
            {"data": {
                "id": model_id, "value": model_id, "type": "model", "name": "Stable Audio",
                "checkpoint": {"uid": CKPT, "path": "C:/models/model.ckpt", "size": 7},
                "encoder": {"uid": encoder_uid, "path": str(self.encoder), "size": 31},
            }},
            {"data": {"id": "audio1", "type": "audio", "context": {"model_id": model_id}}},
            {"data": {"id": "grating1", "type": "grating", "base_model_id": model_id}},
        ]
        edges = [{"data": {"id": f"{model_id}->audio1", "source": model_id, "target": "audio1"}}]
        data = {"project_name": "p", "graph": {"data": [], "elements": {"nodes": nodes, "edges": edges}}}
        (self.project / "graph.json").write_text(json.dumps(data), encoding="utf-8")
        return model_id

    def _graph(self) -> dict:
        return json.loads((self.project / "graph.json").read_text(encoding="utf-8"))

    def test_legacy_uid_and_model_id_are_rewritten_everywhere(self):
        old_model = self._write_project(self.legacy)
        new_model = self.uid_gen.from_uids([CKPT, self.current])

        self.assertTrue(migrate(self.project, dry_run=False, hashed={}))
        graph = self._graph()
        raw = json.dumps(graph)
        self.assertNotIn(self.legacy, raw)
        self.assertNotIn(old_model, raw)

        model, audio, grating = (n["data"] for n in graph["graph"]["elements"]["nodes"])
        self.assertEqual((model["id"], model["value"]), (new_model, new_model))
        self.assertEqual(model["encoder"]["uid"], self.current)
        self.assertEqual(audio["context"]["model_id"], new_model)
        self.assertEqual(grating["base_model_id"], new_model)
        edge = graph["graph"]["elements"]["edges"][0]["data"]
        self.assertEqual((edge["id"], edge["source"]), (f"{new_model}->audio1", new_model))
        self.assertTrue((self.project / BACKUP_NAME).exists())

        # Running again finds nothing left to do
        self.assertFalse(migrate(self.project, dry_run=False, hashed={}))

    def test_dry_run_writes_nothing(self):
        self._write_project(self.legacy)
        before = (self.project / "graph.json").read_text(encoding="utf-8")
        self.assertTrue(migrate(self.project, dry_run=True, hashed={}))
        self.assertEqual((self.project / "graph.json").read_text(encoding="utf-8"), before)
        self.assertFalse((self.project / BACKUP_NAME).exists())

    def test_current_and_unrecognized_uids_are_left_alone(self):
        self._write_project(self.current)
        self.assertFalse(migrate(self.project, dry_run=False, hashed={}))
        self._write_project("0000000000000000.xxh3_64")
        self.assertFalse(migrate(self.project, dry_run=False, hashed={}))
        self.assertFalse((self.project / BACKUP_NAME).exists())


if __name__ == "__main__":
    unittest.main()
