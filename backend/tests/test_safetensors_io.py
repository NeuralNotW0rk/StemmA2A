import sys
import tempfile
import unittest
from pathlib import Path

backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

import torch
from safetensors.torch import load_file, save_file

from utils.safetensors_io import materialize_from_safetensors, read_safetensors


class _Tied(torch.nn.Module):
    """An embedding tied to an output projection, plus a persistent and a non-persistent buffer."""

    def __init__(self) -> None:
        super().__init__()
        self.embed = torch.nn.Embedding(10, 4)
        self.head = torch.nn.Linear(4, 10, bias=False)
        self.head.weight = self.embed.weight
        self.norm = torch.nn.LayerNorm(4)
        self.register_buffer("scale", torch.ones(4))
        self.register_buffer("cache", torch.zeros(3), persistent=False)


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

    def test_reads_only_named_tensors(self):
        path = self._path("model.safetensors")
        save_file({"keep": torch.ones(4), "skip": torch.zeros(4)}, str(path))
        self.assertEqual(set(read_safetensors(path, names=["keep"])), {"keep"})
        with self.assertRaises(KeyError):
            read_safetensors(path, names=["absent"])


class TestMaterializeFromSafetensors(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def _save(self, tensors: dict[str, torch.Tensor], name: str = "model.safetensors") -> Path:
        path = Path(self.tmp.name) / name
        save_file(tensors, str(path))
        return path

    def _saved_weights(self, source: torch.nn.Module, prefix: str = "") -> dict[str, torch.Tensor]:
        # As a Hugging Face checkpoint stores them: tied weights once, under a prefix, plus extras
        weights = {f"{prefix}{k}": v.clone() for k, v in source.state_dict().items() if k != "head.weight"}
        weights["decoder.unused"] = torch.randn(5)
        return weights

    def test_fills_weights_under_prefix_and_keeps_ties(self):
        source = _Tied()
        with torch.no_grad():
            source.scale.fill_(3.0)
        path = self._save(self._saved_weights(source, prefix="model."))

        target = _Tied()
        read = materialize_from_safetensors(target, [path], ("", "model."))
        self.assertGreater(read, 0)
        for key, value in source.state_dict().items():
            self.assertTrue(torch.equal(target.state_dict()[key], value), key)
        self.assertIs(target.head.weight, target.embed.weight)
        self.assertTrue(torch.equal(target.cache, torch.zeros(3)))

    def test_tied_weight_saved_under_other_name(self):
        source = _Tied()
        weights = {k: v.clone() for k, v in source.state_dict().items() if k != "embed.weight"}
        path = self._save(weights)
        target = _Tied()
        materialize_from_safetensors(target, [path])
        self.assertTrue(torch.equal(target.embed.weight, source.embed.weight))
        self.assertIs(target.head.weight, target.embed.weight)

    def test_casts_to_module_dtype(self):
        source = _Tied()
        path = self._save(self._saved_weights(source))
        target = _Tied().to(torch.float16)
        materialize_from_safetensors(target, [path])
        self.assertEqual(target.embed.weight.dtype, torch.float16)
        self.assertTrue(torch.equal(target.embed.weight, source.embed.weight.to(torch.float16)))

    def test_weights_split_across_files(self):
        weights = self._saved_weights(_Tied())
        names = sorted(weights)
        first = self._save({k: weights[k] for k in names[:2]}, "model-00001-of-00002.safetensors")
        second = self._save({k: weights[k] for k in names[2:]}, "model-00002-of-00002.safetensors")
        target = _Tied()
        materialize_from_safetensors(target, [first, second])
        for key in names:
            if key in target.state_dict():
                self.assertTrue(torch.equal(target.state_dict()[key], weights[key]), key)

    def test_missing_weight_raises(self):
        weights = self._saved_weights(_Tied())
        del weights["norm.bias"]
        path = self._save(weights)
        with self.assertRaises(KeyError):
            materialize_from_safetensors(_Tied(), [path])

    def test_shape_mismatch_raises(self):
        weights = self._saved_weights(_Tied())
        weights["scale"] = torch.ones(5)
        path = self._save(weights)
        with self.assertRaises(ValueError):
            materialize_from_safetensors(_Tied(), [path])


if __name__ == "__main__":
    unittest.main()
