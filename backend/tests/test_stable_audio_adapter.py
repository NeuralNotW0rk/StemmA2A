import json
import os
import struct
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add backend directory to sys.path
backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from engine.model_adapters.stable_audio_adapter import StableAudioAdapter
from engine.local_engine import LocalEngine
from param_graph.elements.base_elements import Asset
from utils.uid import XXH3_64, UIDMismatchError


class TestStableAudioAdapter(unittest.TestCase):
    def setUp(self):
        self.adapter = StableAudioAdapter()
        self.uid_gen = XXH3_64()

    def _create_dummy_safetensors(self, path: Path):
        header_json = b'{"__metadata__": {}}'
        header_len = struct.pack("<Q", len(header_json))
        with open(path, "wb") as f:
            f.write(header_len + header_json + b'{}')

    def test_register_model_with_directory_encoder(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            ckpt_path = tmp_path / "model.safetensors"
            self._create_dummy_safetensors(ckpt_path)

            config_path = tmp_path / "config.json"
            config_data = {
                "sample_rate": 44100,
                "sample_size": 2097152,
                "model": {
                    "conditioning": {
                        "configs": [{"id": "prompt", "type": "t5gemma"}]
                    }
                }
            }
            with open(config_path, "w") as f:
                json.dump(config_data, f)

            enc_dir = tmp_path / "encoder_dir"
            enc_dir.mkdir(parents=True, exist_ok=True)
            with open(enc_dir / "tokenizer.json", "w") as f:
                f.write('{"model": "t5"}')

            with patch("engine.model_adapters.stable_audio_adapter.load_ckpt_state_dict", return_value={}):
                model = self.adapter.register_model(
                    name="Test Stable Audio 3",
                    checkpoint_path=str(ckpt_path),
                    config_path=str(config_path),
                    encoder_path=str(enc_dir)
                )

            self.assertIsNotNone(model.encoder)
            self.assertEqual(model.encoder.uid, self.uid_gen.from_directory(enc_dir))
            expected_id = self.uid_gen.from_uids([model.checkpoint.uid, model.encoder.uid])
            self.assertEqual(model.id, expected_id)

    def test_register_model_with_archive_encoder(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            ckpt_path = tmp_path / "model.safetensors"
            self._create_dummy_safetensors(ckpt_path)

            config_path = tmp_path / "config.json"
            with open(config_path, "w") as f:
                json.dump({"sample_rate": 44100, "sample_size": 1000}, f)

            enc_archive = tmp_path / "encoder.zip"
            with open(enc_archive, "wb") as f:
                f.write(b"PK\x05\x06" + b"\x00" * 18)  # Empty zip archive bytes

            with patch("engine.model_adapters.stable_audio_adapter.load_ckpt_state_dict", return_value={}):
                model = self.adapter.register_model(
                    name="Test Archive Encoder",
                    checkpoint_path=str(ckpt_path),
                    config_path=str(config_path),
                    encoder_path=str(enc_archive),
                    encoder_uid="custom_encoder_uid.xxh3_64"
                )

            self.assertIsNotNone(model.encoder)
            self.assertEqual(model.encoder.uid, "custom_encoder_uid.xxh3_64")
            expected_id = self.uid_gen.from_uids([model.checkpoint.uid, "custom_encoder_uid.xxh3_64"])
            self.assertEqual(model.id, expected_id)

    def test_register_model_with_encoder_uid_only(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            ckpt_path = tmp_path / "model.safetensors"
            self._create_dummy_safetensors(ckpt_path)

            config_path = tmp_path / "config.json"
            with open(config_path, "w") as f:
                json.dump({"sample_rate": 44100, "sample_size": 1000}, f)

            with patch("engine.model_adapters.stable_audio_adapter.load_ckpt_state_dict", return_value={}):
                model = self.adapter.register_model(
                    name="Test UID Only",
                    checkpoint_path=str(ckpt_path),
                    config_path=str(config_path),
                    encoder_uid="remote_enc_123.xxh3_64",
                    encoder_size=5000
                )

            self.assertIsNotNone(model.encoder)
            self.assertEqual(model.encoder.uid, "remote_enc_123.xxh3_64")
            self.assertEqual(model.encoder.size, 5000)
            expected_id = self.uid_gen.from_uids([model.checkpoint.uid, "remote_enc_123.xxh3_64"])
            self.assertEqual(model.id, expected_id)

    def test_resolve_model_element_matches_by_checkpoint_and_encoder(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = LocalEngine(data_root=tmpdir)
            dummy_shared = MagicMock()
            dummy_shared.id = "shared_model_combined_id"
            dummy_shared.checkpoint = Asset(path="/models/ckpt.safetensors", uid="ckpt_uid_123.xxh3_64", size=100)
            dummy_shared.encoder = Asset(path="/models/enc", uid="enc_uid_456.xxh3_64", size=200)

            engine.shared_models = [dummy_shared]

            # Case A: Exact ID match
            query_exact = MagicMock()
            query_exact.id = "shared_model_combined_id"
            self.assertEqual(engine._resolve_model_element(query_exact), dummy_shared)

            # Case B: Diverged model ID from client project, but matching checkpoint + encoder UIDs
            query_diverged = MagicMock()
            query_diverged.id = "client_project_diff_id"
            query_diverged.checkpoint = Asset(path="", uid="ckpt_uid_123.xxh3_64", size=100)
            query_diverged.encoder = Asset(path="", uid="enc_uid_456.xxh3_64", size=200)

            resolved = engine._resolve_model_element(query_diverged)
            self.assertEqual(resolved, dummy_shared)


if __name__ == "__main__":
    unittest.main()
