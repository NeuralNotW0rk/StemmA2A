import json
import sys
import shutil
import tempfile
import unittest
from pathlib import Path

backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from param_graph import cleanup
from param_graph.graph import ParameterGraph

AUDIO = "audio1.xxh3_64"
INDIVIDUAL = "ind1.xxh3_64"
EXTERNAL = "external_audio"


class TestProjectCleanup(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.root = self.tmp / "project"
        self.graph = ParameterGraph(str(self.root))
        self.graph.project_name = "project"

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _write(self, relative: str, content: bytes = b"data") -> Path:
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        return path

    def _orphans(self) -> list[str]:
        refs = cleanup.collect_references(self.graph)
        return [o.path for o in cleanup.find_unreferenced(self.root, refs)]

    def _build(self, stored_root: Path | None = None):
        """An audio node and an individual whose context points at a baseline grating file."""
        stored_root = stored_root or self.root
        self._write("generate/kept.wav")
        self._write("generate/baseline.safetensors")
        self._write(f"generate/{INDIVIDUAL}.safetensors")
        self._write(f"cache/au/{AUDIO}")
        self._write(f"cache/{EXTERNAL}.wav")
        self._write("generate/removed.wav")
        self._write("process/removed_slice.wav")
        self._write("cache/de/dead.xxh3_64")
        G = self.graph.G
        G.add_node(AUDIO, id=AUDIO, type="audio", name="kept", context={},
                   file={"path": str(stored_root / "generate" / "kept.wav"), "uid": AUDIO, "extension": ".wav"})
        G.add_node(INDIVIDUAL, id=INDIVIDUAL, type="individual", name="ind",
                   context={"shared_context": {"baseline_file_path": str(stored_root / "generate" / "baseline.safetensors")}},
                   file={"path": "", "uid": INDIVIDUAL, "extension": ".safetensors"})
        G.add_node(EXTERNAL, id=EXTERNAL, type="audio", name="ext", context={},
                   file={"path": "D:/elsewhere/sample.wav", "uid": "ext.xxh3_64", "extension": ".wav"})

    def test_only_unreferenced_files_are_found(self):
        self._build()
        self.assertEqual(
            self._orphans(),
            ["cache/de/dead.xxh3_64", "generate/removed.wav", "process/removed_slice.wav"],
        )

    def test_moved_project_keeps_its_files(self):
        # Paths stored under the project's old location still match by their generate/... tail
        self._build(stored_root=self.tmp / "old_location" / "project")
        self.assertNotIn("generate/kept.wav", self._orphans())
        self.assertNotIn("generate/baseline.safetensors", self._orphans())

    def test_backup_graph_references_are_protected(self):
        self._build()
        self.graph.save()
        self.graph.G.remove_node(AUDIO)
        self.graph.save()  # graph.json.bak now holds the graph with the audio node
        self.assertNotIn("generate/kept.wav", self._orphans())
        self.graph.save()  # Both files are past the removal now
        self.assertIn("generate/kept.wav", self._orphans())

    def test_corrupt_graph_file_aborts(self):
        self._build()
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "graph.json").write_text("{not json", encoding="utf-8")
        with self.assertRaises(ValueError):
            cleanup.collect_references(self.graph)

    def test_quarantine_restore_round_trip(self):
        self._build()
        refs = cleanup.collect_references(self.graph)
        orphans = cleanup.find_unreferenced(self.root, refs)
        batch = cleanup.quarantine(self.root, orphans)

        self.assertEqual(batch["file_count"], 3)
        self.assertFalse((self.root / "generate" / "removed.wav").exists())
        self.assertFalse((self.root / "cache" / "de").exists())  # emptied shard dir is pruned
        self.assertTrue((self.root / ".trash" / batch["id"] / "generate" / "removed.wav").exists())
        self.assertEqual(self._orphans(), [])
        self.assertEqual([b["id"] for b in cleanup.list_trash(self.root)], [batch["id"]])

        # A file that reappeared at its original location is not overwritten
        self._write("generate/removed.wav", b"new")
        result = cleanup.restore(self.root, batch["id"])
        self.assertEqual(result["restored"], 2)
        self.assertEqual(result["conflicts"], ["generate/removed.wav"])
        self.assertEqual((self.root / "generate" / "removed.wav").read_bytes(), b"new")
        self.assertTrue((self.root / "cache" / "de" / "dead.xxh3_64").exists())
        manifest = json.loads((self.root / ".trash" / batch["id"] / "manifest.json").read_text())
        self.assertEqual([f["path"] for f in manifest["files"]], ["generate/removed.wav"])

    def test_purge_deletes_batch(self):
        self._build()
        refs = cleanup.collect_references(self.graph)
        batch = cleanup.quarantine(self.root, cleanup.find_unreferenced(self.root, refs))
        result = cleanup.purge(self.root, batch["id"])
        self.assertEqual(result["file_count"], 3)
        self.assertFalse((self.root / ".trash").exists())
        self.assertTrue((self.root / "generate" / "kept.wav").exists())

    def test_batch_ids_cannot_escape_trash(self):
        for bad in ("..", "../generate", "20260101-000000/../..", ""):
            with self.assertRaises(ValueError):
                cleanup.purge(self.root, bad)


if __name__ == "__main__":
    unittest.main()
