import os
import sys
import json
import time
import shutil
import tempfile
import unittest
import unittest.mock
from pathlib import Path

backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from engine.model_store import ModelStore, MissingModelFiles, migrate_legacy_registry

CKPT = "ckpt.xxh3_64"
ENC = "enc.xxh3_64"


class TestModelStore(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.store = ModelStore(self.root)

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def _cached(self, uid: str, content: bytes = b"weights") -> Path:
        path = self.store.cache_path(uid)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        return path

    def _entry(self, **refs) -> dict:
        return {"name": "Model", "adapter": "stable_audio_tools", "config": {"sample_rate": 44100}, **refs}

    def test_resolve_prefers_the_store(self):
        self.assertIsNone(self.store.resolve(CKPT))
        self._cached(CKPT)
        self.assertEqual(self.store.resolve(CKPT), self.store.cache_path(CKPT))
        self.store.promote(self._entry(checkpoint={"uid": CKPT}))
        self.assertEqual(self.store.resolve(CKPT), self.store.model_path(CKPT))
        self.assertEqual(self.store.model_path(CKPT).relative_to(self.root).parts[0], "models")

    def test_promote_moves_cached_files_into_the_store(self):
        self._cached(CKPT, b"1234567")
        self._cached(ENC)
        normalized = self.store.promote(self._entry(
            checkpoint={"uid": CKPT, "size": 7, "path": "C:/host/only/model.ckpt"},
            encoder={"uid": ENC},
        ))
        # The host path doesn't exist here, so only the UID is kept
        self.assertEqual(normalized["checkpoint"], {"uid": CKPT, "size": 7})
        self.assertTrue(self.store.model_path(CKPT).exists())
        self.assertFalse(self.store.cache_path(CKPT).exists())
        self.assertEqual(self.store.load_registry(), [normalized])

    def test_promote_reports_missing_files_without_changing_anything(self):
        self._cached(CKPT)
        with self.assertRaises(MissingModelFiles) as raised:
            self.store.promote(self._entry(checkpoint={"uid": CKPT}, encoder={"uid": ENC}))
        self.assertEqual(raised.exception.uids, [ENC])
        self.assertTrue(self.store.cache_path(CKPT).exists())
        self.assertEqual(self.store.load_registry(), [])

    def test_promote_rejects_wrong_size(self):
        self._cached(CKPT, b"short")
        with self.assertRaises(ValueError):
            self.store.promote(self._entry(checkpoint={"uid": CKPT, "size": 999}))
        self.assertTrue(self.store.cache_path(CKPT).exists())

    def test_existing_path_is_used_in_place(self):
        external = self.root / "mounted" / "model.ckpt"
        external.parent.mkdir()
        external.write_bytes(b"weights")
        normalized = self.store.promote(self._entry(checkpoint={"path": str(external), "uid": CKPT}))
        self.assertEqual(normalized["checkpoint"]["path"], str(external))
        self.assertFalse(self.store.models_dir.joinpath("ck").exists())
        self.assertEqual(self.store.register_kwargs(normalized)["checkpoint_path"], str(external))

    def test_promoting_again_replaces_the_entry(self):
        self._cached(CKPT)
        self.store.promote(self._entry(checkpoint={"uid": CKPT}))
        self.store.promote({**self._entry(checkpoint={"uid": CKPT}), "name": "Renamed"})
        self.assertEqual([e["name"] for e in self.store.load_registry()], ["Renamed"])

    def test_register_kwargs(self):
        self._cached(CKPT)
        self._cached(ENC)
        entry = self.store.promote(self._entry(checkpoint={"uid": CKPT, "size": 7}, encoder={"uid": ENC}))
        kwargs = self.store.register_kwargs(entry)
        self.assertEqual(kwargs["checkpoint_path"], str(self.store.model_path(CKPT)))
        self.assertEqual(kwargs["checkpoint_uid"], CKPT)
        self.assertEqual(kwargs["checkpoint_size"], 7)
        self.assertEqual(kwargs["encoder_path"], str(self.store.model_path(ENC)))
        self.assertEqual(kwargs["config"], {"sample_rate": 44100})

        # A registry reference only resolves in the store, never the cache
        self.store.model_path(ENC).unlink()
        self._cached(ENC)
        with self.assertRaises(FileNotFoundError):
            self.store.register_kwargs(entry)

    def test_demote_returns_files_to_the_cache(self):
        self._cached(CKPT)
        self._cached(ENC)
        first = self.store.promote(self._entry(checkpoint={"uid": CKPT}, encoder={"uid": ENC}))
        # A second model shares the encoder
        self._cached("other.xxh3_64")
        self.store.promote(self._entry(checkpoint={"uid": "other.xxh3_64"}, encoder={"uid": ENC}))
        os.utime(self.store.model_path(CKPT), (0, 0))

        self.store.demote(first)
        self.assertTrue(self.store.cache_path(CKPT).exists())
        self.assertFalse(self.store.model_path(CKPT).exists())
        # Back in the cache with a fresh last-use time
        self.assertGreater(self.store.cache_path(CKPT).stat().st_mtime, time.time() - 60)
        self.assertTrue(self.store.model_path(ENC).exists())
        self.assertEqual(len(self.store.load_registry()), 1)

        with self.assertRaises(ValueError):
            self.store.demote(first)


class TestLocalEngineSharedModels(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.env = unittest.mock.patch.dict(os.environ, {"LOCAL_DATA_PATH": str(self.root)})
        self.env.start()
        os.environ.pop("RUNNING_IN_CONTAINER", None)

    def tearDown(self):
        self.env.stop()
        shutil.rmtree(self.root, ignore_errors=True)

    def test_add_and_remove_round_trip(self):
        import asyncio
        from engine.local_engine import LocalEngine

        store = ModelStore(self.root)
        cached = store.cache_path(CKPT)
        cached.parent.mkdir(parents=True)
        cached.write_bytes(b"weights")

        engine = LocalEngine(data_root=str(self.root / "project"))
        self.assertEqual(engine.shared_models, [])
        entry = {"name": "Shared SA", "adapter": "stable_audio_tools", "model_type": "stable_audio_tools",
                 "checkpoint": {"uid": CKPT, "size": 7}, "config": {"sample_rate": 44100}}

        # A file neither here nor cached is reported, so a remote client knows to upload it
        with self.assertRaises(MissingModelFiles):
            asyncio.run(engine.add_shared_model({**entry, "encoder": {"uid": ENC}}))

        shared = asyncio.run(engine.add_shared_model(entry))
        self.assertEqual(shared["checkpoint"]["path"], "REMOTE")
        self.assertEqual([m.id for m in engine.shared_models], [shared["id"]])
        self.assertEqual(engine.shared_models[0].checkpoint.path, str(store.model_path(CKPT)))

        # A fresh engine loads it from the registry
        self.assertEqual([m.id for m in LocalEngine(data_root=str(self.root / "p2")).shared_models], [shared["id"]])

        asyncio.run(engine.remove_shared_model(shared["id"]))
        self.assertEqual(engine.shared_models, [])
        self.assertTrue(cached.exists())
        with self.assertRaises(ValueError):
            asyncio.run(engine.remove_shared_model(shared["id"]))


class TestLegacyMigration(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.store = ModelStore(self.root)
        self.legacy = self.root / "backend" / "shared_models.json"
        self.legacy.parent.mkdir()
        self.external = self.root / "external" / "model.safetensors"
        self.external.parent.mkdir()
        self.external.write_bytes(b"weights")

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def test_migrates_flat_and_node_entries(self):
        cached = self.store.cache_path(CKPT)
        cached.parent.mkdir(parents=True)
        cached.write_bytes(b"weights")
        self.legacy.write_text(json.dumps([
            {   # Flat keys, with the checkpoint in the cache and a Windows path that isn't here
                "name": "Cached", "adapter": "stable_audio_tools", "model_type": "stable_audio_tools",
                "checkpoint_uid": CKPT, "checkpoint_path": "C:\\Models\\model.safetensors",
                "checkpoint_size": 7, "config": {"sample_rate": 44100},
            },
            {   # Path-only, with a config file
                "name": "Local", "adapter": "stable_audio_tools",
                "checkpoint_path": str(self.external), "config_path": str(self.external),
            },
            {   # A node copied from graph.json
                "data": {"name": "Node", "adapter": "stylegan2",
                         "checkpoint": {"path": str(self.external), "uid": "node.xxh3_64", "size": 7}},
            },
        ]))

        self.assertTrue(migrate_legacy_registry(self.root, self.legacy))
        entries = {e["name"]: e for e in self.store.load_registry()}
        self.assertEqual(entries["Cached"]["checkpoint"], {"uid": CKPT, "size": 7})
        self.assertEqual(entries["Cached"]["config"], {"sample_rate": 44100})
        self.assertTrue(self.store.model_path(CKPT).exists())
        self.assertFalse(cached.exists())
        self.assertEqual(entries["Local"]["checkpoint"], {"path": str(self.external)})
        self.assertEqual(entries["Local"]["config_file"], {"path": str(self.external)})
        self.assertEqual(entries["Node"]["checkpoint"]["uid"], "node.xxh3_64")
        self.assertEqual(entries["Node"]["checkpoint"]["path"], str(self.external))

        self.assertFalse(self.legacy.exists())
        self.assertTrue(self.legacy.with_name("shared_models.json.migrated").exists())

    def test_does_nothing_once_the_registry_exists(self):
        self.store.save_registry([])
        self.legacy.write_text("[]")
        self.assertFalse(migrate_legacy_registry(self.root, self.legacy))
        self.assertTrue(self.legacy.exists())


if __name__ == "__main__":
    unittest.main()
