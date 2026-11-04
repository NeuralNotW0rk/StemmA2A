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

from backfill_asset_sizes import BACKUP_NAME, backfill

CKPT_UID = "ckpt.xxh3_64"


class TestBackfillAssetSizes(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path(tempfile.mkdtemp())
        self.ckpt = self.root / "model.safetensors"
        self.ckpt.write_bytes(b"weights")
        self.project = self.root / "project"
        self.project.mkdir()
        self.hash_calls = 0
        # Stands in for the adapter's state dict hash: the UID holds while the file is unchanged
        def fake_hash(path: Path) -> str:
            self.hash_calls += 1
            return CKPT_UID if path.read_bytes() == b"weights" else "other.xxh3_64"
        self.hashers = {("stable_audio_tools", "checkpoint"): fake_hash}

    def tearDown(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)

    def _write_project(self, checkpoint: dict) -> None:
        nodes = [
            {"data": {"id": CKPT_UID, "type": "model", "adapter": "stable_audio_tools", "checkpoint": checkpoint}},
            {"data": {"id": "audio1", "type": "audio", "file": {"uid": "a", "path": str(self.ckpt)}}},
        ]
        data = {"project_name": "p", "graph": {"data": [], "elements": {"nodes": nodes, "edges": []}}}
        (self.project / "graph.json").write_text(json.dumps(data), encoding="utf-8")

    def _nodes(self) -> list[dict]:
        graph = json.loads((self.project / "graph.json").read_text(encoding="utf-8"))
        return [n["data"] for n in graph["graph"]["elements"]["nodes"]]

    def test_verified_checkpoint_gets_its_size(self) -> None:
        self._write_project({"uid": CKPT_UID, "path": str(self.ckpt), "extension": ""})
        self.assertTrue(backfill(self.project, dry_run=False, verified={}, hashers=self.hashers))
        model, audio = self._nodes()
        self.assertEqual(model["checkpoint"]["size"], len(b"weights"))
        # Only model assets are touched
        self.assertNotIn("size", audio["file"])
        self.assertTrue((self.project / BACKUP_NAME).exists())
        # Running again finds nothing left to do, without hashing
        self.assertFalse(backfill(self.project, dry_run=False, verified={}, hashers=self.hashers))
        self.assertEqual(self.hash_calls, 1)

    def test_changed_file_is_left_alone(self) -> None:
        self.ckpt.write_bytes(b"retrained")
        self._write_project({"uid": CKPT_UID, "path": str(self.ckpt)})
        self.assertFalse(backfill(self.project, dry_run=False, verified={}, hashers=self.hashers))
        self.assertNotIn("size", self._nodes()[0]["checkpoint"])
        self.assertFalse((self.project / BACKUP_NAME).exists())

    def test_dry_run_writes_nothing(self) -> None:
        self._write_project({"uid": CKPT_UID, "path": str(self.ckpt)})
        before = (self.project / "graph.json").read_text(encoding="utf-8")
        self.assertTrue(backfill(self.project, dry_run=True, verified={}, hashers=self.hashers))
        self.assertEqual((self.project / "graph.json").read_text(encoding="utf-8"), before)
        self.assertFalse((self.project / BACKUP_NAME).exists())

    def test_missing_file_and_sized_assets_are_skipped(self) -> None:
        self._write_project({"uid": CKPT_UID, "path": str(self.root / "gone.safetensors")})
        self.assertFalse(backfill(self.project, dry_run=False, verified={}, hashers=self.hashers))
        self._write_project({"uid": CKPT_UID, "path": str(self.ckpt), "size": 7})
        self.assertFalse(backfill(self.project, dry_run=False, verified={}, hashers=self.hashers))
        self.assertEqual(self.hash_calls, 0)


if __name__ == "__main__":
    unittest.main()
