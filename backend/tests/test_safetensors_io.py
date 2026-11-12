import sys
import tempfile
import unittest
from pathlib import Path

backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

import torch
from safetensors.torch import load_file, save_file

from utils.safetensors_io import read_safetensors


class TestReadSafetensors(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def _path(self, name: str) -> Path:
        return Path(self.tmp.name) / name

    def test_matches_safetensors_load_file(self):
        tensors = {
            "b.weight": torch.randn(64, 32),
            "a.bias": torch.randn(32).to(torch.float16),
            "c.half": torch.randn(3, 4, 5).to(torch.bfloat16),
            "d.ints": torch.arange(10, dtype=torch.int64),
            "e.mask": torch.tensor([True, False, True]),
            "f.scalar": torch.tensor(1.5),
            "g.empty": torch.empty(0, 4),
        }
        path = self._path("model.safetensors")
        save_file(tensors, str(path), metadata={"format": "pt"})

        expected = load_file(str(path))
        actual = read_safetensors(path)
        self.assertEqual(set(actual), set(expected))
        for name, tensor in expected.items():
            self.assertEqual(actual[name].dtype, tensor.dtype, name)
            self.assertEqual(actual[name].shape, tensor.shape, name)
            self.assertTrue(torch.equal(actual[name], tensor), name)

    def test_tensors_own_their_memory(self):
        path = self._path("model.safetensors")
        save_file({"w": torch.ones(8)}, str(path))
        loaded = read_safetensors(path)
        path.unlink()
        # Still readable and writable once the file is gone, so nothing is mapped from it
        loaded["w"].add_(1)
        self.assertTrue(torch.equal(loaded["w"], torch.full((8,), 2.0)))

    def test_truncated_file_raises(self):
        path = self._path("model.safetensors")
        save_file({"w": torch.ones(1024)}, str(path))
        data = path.read_bytes()
        path.write_bytes(data[:-100])
        with self.assertRaises(ValueError):
            read_safetensors(path)

    def test_non_safetensors_header_raises(self):
        path = self._path("model.ckpt")
        path.write_bytes(b"\xff" * 64)
        with self.assertRaises(ValueError):
            read_safetensors(path)


if __name__ == "__main__":
    unittest.main()
