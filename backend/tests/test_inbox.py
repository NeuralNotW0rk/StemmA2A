import os
import sys
import shutil
import tempfile
import threading
import time
import unittest
from pathlib import Path

backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from engine.inbox import Inbox
from engine.model_store import ModelStore
from utils.uid import XXH3_64, path_from_uid

UID = "f28c7dd48e02c8a6.xxh3_64"


class TestInbox(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.store = ModelStore(self.root)
        self.inbox = Inbox(self.root, threading.RLock(), self.store.model_path)
        self.inbox.inbox_dir.mkdir()

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def _drop(self, name: str, content: bytes = b"weights", age: float = 30 * 24 * 3600) -> Path:
        path = self.inbox.inbox_dir / name
        path.write_bytes(content)
        # Copies often keep the source's modified time
        os.utime(path, (time.time() - age, time.time() - age))
        return path

    def test_file_is_moved_once_it_is_unchanged_between_scans(self):
        dropped = self._drop(UID)
        self.assertEqual(self.inbox.scan(), [])
        self.assertTrue(dropped.exists())

        self.assertEqual(self.inbox.scan(), [UID])
        cached = self.root / path_from_uid(UID)
        self.assertEqual(cached.read_bytes(), b"weights")
        self.assertFalse(dropped.exists())
        # Marked as just used, not as old as the copied file's time
        self.assertGreater(cached.stat().st_mtime, time.time() - 60)

    def test_file_still_growing_waits(self):
        dropped = self._drop(UID, b"part")
        self.inbox.scan()
        with open(dropped, "ab") as f:
            f.write(b"more")
        self.assertEqual(self.inbox.scan(), [])
        self.assertEqual(self.inbox.scan(), [UID])

    def test_part_files_and_other_names_are_left_alone(self):
        part = self._drop(f"{UID}.part")
        other = self._drop("model.ckpt")
        self.inbox.scan()
        self.assertEqual(self.inbox.scan(), [])
        self.assertTrue(part.exists())
        self.assertTrue(other.exists())

    def test_directory_is_checked_against_its_uid(self):
        source = self.root / "encoder"
        (source / "sub").mkdir(parents=True)
        (source / "sub" / "weights.bin").write_bytes(b"weights")
        uid = XXH3_64().from_directory(source)
        shutil.move(str(source), str(self.inbox.inbox_dir / uid))
        wrong = self.inbox.inbox_dir / UID
        shutil.copytree(self.inbox.inbox_dir / uid, wrong)

        self.inbox.scan()
        self.assertEqual(self.inbox.scan(), [uid])
        self.assertTrue((self.root / path_from_uid(uid) / "sub" / "weights.bin").exists())
        # Contents that don't hash to the name stay in the inbox
        self.assertTrue(wrong.exists())
        self.assertFalse((self.root / path_from_uid(UID)).exists())

    def test_assets_in_the_model_store_are_not_copied_into_the_cache(self):
        stored = self.store.model_path(UID)
        stored.parent.mkdir(parents=True)
        stored.write_bytes(b"weights")
        dropped = self._drop(UID)
        self.inbox.scan()
        self.assertEqual(self.inbox.scan(), [])
        self.assertTrue(dropped.exists())
        self.assertFalse((self.root / path_from_uid(UID)).exists())


if __name__ == "__main__":
    unittest.main()
