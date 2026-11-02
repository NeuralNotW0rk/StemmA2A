import sys
import shutil
import tempfile
import unittest
from pathlib import Path

backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from engine import uploads
from utils.uid import XXH3_64


class TestUploads(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.source = self.root / "source"
        (self.source / "sub").mkdir(parents=True)
        (self.source / "config.json").write_bytes(b'{"a": 1}')
        (self.source / "sub" / "weights.bin").write_bytes(b"0123456789")
        self.uid = XXH3_64().from_directory(self.source)
        self.data = self.root / "data"

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def _send_file(self, uid: str, content: bytes, relpath: str | None = None, chunk: int = 4) -> bool:
        pieces = [content[i:i + chunk] for i in range(0, len(content), chunk)] or [b""]
        done = False
        for index, piece in enumerate(pieces):
            done = uploads.write_chunk(self.data, uid, piece, index, len(pieces), len(content), relpath)
        return done

    def _send_directory(self, uid: str) -> None:
        for path in self.source.glob("**/*"):
            if path.is_file():
                self.assertTrue(self._send_file(uid, path.read_bytes(), path.relative_to(self.source).as_posix()))

    def test_file_is_moved_into_place_after_its_last_chunk(self):
        destination = uploads.destination_path(self.data, "f.xxh3_64")
        content = b"0123456789"
        pieces = [content[:4], content[4:8], content[8:]]
        self.assertFalse(uploads.write_chunk(self.data, "f.xxh3_64", pieces[0], 0, 3, 10))
        self.assertFalse(destination.exists())
        uploads.write_chunk(self.data, "f.xxh3_64", pieces[1], 1, 3, 10)
        self.assertTrue(uploads.write_chunk(self.data, "f.xxh3_64", pieces[2], 2, 3, 10))
        self.assertEqual(destination.read_bytes(), content)
        self.assertFalse(uploads.staging_path(self.data, "f.xxh3_64").exists())

    def test_file_size_mismatch_is_rejected(self):
        with self.assertRaises(uploads.UploadError):
            uploads.write_chunk(self.data, "f.xxh3_64", b"short", 0, 1, 99)
        self.assertFalse(uploads.destination_path(self.data, "f.xxh3_64").exists())

    def test_directory_is_stored_as_a_directory(self):
        self._send_directory(self.uid)
        destination = uploads.destination_path(self.data, self.uid)
        # Staged until finalized
        self.assertFalse(destination.exists())

        uploads.finalize_directory(self.data, self.uid, total_size=18)
        self.assertTrue(destination.is_dir())
        self.assertEqual((destination / "sub" / "weights.bin").read_bytes(), b"0123456789")
        self.assertEqual(XXH3_64().from_directory(destination), self.uid)
        self.assertFalse(uploads.staging_path(self.data, self.uid).exists())

    def test_directory_that_does_not_match_its_uid_is_discarded(self):
        wrong = "0000000000000000.xxh3_64"
        self._send_directory(wrong)
        with self.assertRaises(uploads.UploadError):
            uploads.finalize_directory(self.data, wrong, total_size=18)
        self.assertFalse(uploads.destination_path(self.data, wrong).exists())
        self.assertFalse(uploads.staging_path(self.data, wrong).exists())

    def test_directory_size_mismatch_is_rejected(self):
        self._send_directory(self.uid)
        with self.assertRaises(uploads.UploadError):
            uploads.finalize_directory(self.data, self.uid, total_size=17)
        self.assertFalse(uploads.destination_path(self.data, self.uid).exists())

    def test_member_paths_cannot_escape_the_directory(self):
        for relpath in ("../outside.bin", "/abs.bin", "C:/abs.bin", "sub\\..\\x.bin"):
            with self.assertRaises(uploads.UploadError, msg=relpath):
                uploads.write_chunk(self.data, self.uid, b"x", 0, 1, 1, relpath)
        self.assertFalse((self.data / "cache" / "outside.bin").exists())


if __name__ == "__main__":
    unittest.main()
