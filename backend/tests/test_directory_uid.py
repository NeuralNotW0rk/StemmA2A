import os
import sys
import tempfile
from pathlib import Path

# Add backend directory to sys.path
backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from utils.uid import XXH3_64


def test_directory_uid_determinism():
    uid_gen = XXH3_64()

    # Create temporary directory structure for testing
    with tempfile.TemporaryDirectory() as tmpdir:
        dir_a = Path(tmpdir) / "dir_a"
        dir_b = Path(tmpdir) / "dir_b"

        # Setup dir_a (standard files, LF line endings)
        (dir_a / "subfolder").mkdir(parents=True, exist_ok=True)
        with open(dir_a / "config.json", "wb") as f:
            f.write(b'{\n  "name": "test",\n  "version": 1\n}')
        with open(dir_a / "subfolder" / "notes.txt", "wb") as f:
            f.write(b"Line 1\nLine 2\n")
        with open(dir_a / "weights.bin", "wb") as f:
            f.write(b"\x00\x01\x02\x03\x04")

        # Setup dir_b (CRLF line endings, unformatted JSON, extra .DS_Store file)
        (dir_b / "subfolder").mkdir(parents=True, exist_ok=True)
        with open(dir_b / "config.json", "wb") as f:
            f.write(b'{"version":1, "name":"test"}')  # Different key order and spacing, same JSON semantics
        with open(dir_b / "subfolder" / "notes.txt", "wb") as f:
            f.write(b"Line 1\r\nLine 2\r\n")  # CRLF line endings
        with open(dir_b / "weights.bin", "wb") as f:
            f.write(b"\x00\x01\x02\x03\x04")
        with open(dir_b / ".DS_Store", "wb") as f:
            f.write(b"macOS junk file")

        hash_a = uid_gen.from_directory(dir_a)
        hash_b = uid_gen.from_directory(dir_b)

        assert hash_a == hash_b, f"Hashes mismatch! {hash_a} != {hash_b}"
        print("PASS: Directory UID determinism test passed successfully!")


if __name__ == "__main__":
    test_directory_uid_determinism()
