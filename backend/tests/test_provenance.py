import sys
import time
import shutil
import tempfile
import unittest
from unittest.mock import MagicMock
from pathlib import Path

import numpy as np
import soundfile as sf
from PIL import Image as PILImage

backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

import app as app_module
from app import app
from param_graph.graph import ParameterGraph
from param_graph.elements.artifacts.audio_element import Audio
from param_graph.elements.artifacts.image_element import Image
from param_graph.elements.base_elements import Asset
from param_graph.elements.local_path import LocalPath
from engine.engine_provider import EngineProvider
from utils.provenance import embed_provenance, read_provenance, restore_context, ORIGIN_KEY

SAMPLE_RATE = 16000


def _write_wav(path: Path, freq: float = 440.0) -> np.ndarray:
    t = np.linspace(0, 0.25, int(SAMPLE_RATE * 0.25), endpoint=False)
    data = (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    sf.write(str(path), data, SAMPLE_RATE)
    return data


def _audio(element_id: str, path: Path, context: dict) -> Audio:
    return Audio(id=element_id, name=path.stem, context=context,
                 file=Asset(path=str(path), uid=element_id, extension=".wav"))


class TestProvenanceFormats(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_wav_round_trip_keeps_samples(self):
        path = self.tmp / "a.wav"
        _write_wav(path)
        before, _ = sf.read(str(path), dtype="float32")
        element = _audio("a.xxh3_64", path, {"prompt": "Amen break", "seed": 7, "job_id": "dropped"})

        self.assertTrue(embed_provenance(element))

        after, sr = sf.read(str(path), dtype="float32")
        self.assertEqual(sr, SAMPLE_RATE)
        np.testing.assert_array_equal(before, after)
        payload = read_provenance(path)
        self.assertEqual(payload["id"], "a.xxh3_64")
        self.assertEqual(payload["context"], {"prompt": "Amen break", "seed": 7})

    def test_reembedding_replaces_the_record(self):
        path = self.tmp / "a.wav"
        _write_wav(path)
        embed_provenance(_audio("a.xxh3_64", path, {"seed": 1}))
        size_once = path.stat().st_size
        embed_provenance(_audio("a.xxh3_64", path, {"seed": 2}))

        self.assertEqual(path.stat().st_size, size_once)
        self.assertEqual(read_provenance(path)["context"], {"seed": 2})

    def test_png_round_trip_is_readable_by_pillow(self):
        path = self.tmp / "a.png"
        pixels = (np.random.rand(16, 16, 3) * 255).astype(np.uint8)
        PILImage.fromarray(pixels).save(path)
        element = Image(id="img.xxh3_64", name="a", context={"truncation": 0.7},
                        file=Asset(path=str(path), uid="img.xxh3_64", extension=".png"))

        self.assertTrue(embed_provenance(element))
        embed_provenance(element)

        with PILImage.open(path) as img:
            np.testing.assert_array_equal(np.asarray(img.convert("RGB")), pixels)
            self.assertIn("stemma", img.text)
        self.assertEqual(read_provenance(path)["context"], {"truncation": 0.7})

    def test_untagged_and_unsupported_files(self):
        wav = self.tmp / "plain.wav"
        _write_wav(wav)
        self.assertIsNone(read_provenance(wav))

        latent = self.tmp / "latent.pt"
        latent.write_bytes(b"not audio")
        element = _audio("l.xxh3_64", latent, {})
        self.assertFalse(embed_provenance(element))
        self.assertEqual(latent.read_bytes(), b"not audio")

    def test_restore_context_points_at_the_original(self):
        payload = {"stemma": 1, "id": "orig.xxh3_64", "context": {"source_id": "src.xxh3_64"}}
        self.assertEqual(restore_context(payload, "copy.xxh3_64"),
                         {"source_id": "src.xxh3_64", ORIGIN_KEY: "orig.xxh3_64"})
        # A copy that hashes to the original's UID is the original, not a descendant of it
        self.assertNotIn(ORIGIN_KEY, restore_context(payload, "orig.xxh3_64"))


class TestProvenanceRoundTrip(unittest.TestCase):
    """An exported file, imported again, rejoins the lineage of the artifact it came from."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.old_graph = app_module.param_graph
        self.old_local_jobs = dict(app_module.local_jobs)
        self.old_trigger = app_module.trigger_embedding_update
        app_module.trigger_embedding_update = MagicMock()

        self.graph = ParameterGraph(str(self.tmp / "project"))
        self.graph.project_name = "ProvenanceTest"
        self.graph.save()
        app_module.param_graph = self.graph
        # Job status lookups need an engine, even for local jobs
        self.old_provider = app_module.engine_provider
        EngineProvider._engine_instance = None
        app_module.engine_provider = EngineProvider(data_root=str(self.tmp / "project"))
        self.client = app.test_client()

    def tearDown(self):
        app_module.param_graph = self.old_graph
        app_module.engine_provider = self.old_provider
        app_module.local_jobs = self.old_local_jobs
        app_module.trigger_embedding_update = self.old_trigger
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _run(self, route: str, body: dict) -> dict:
        resp = self.client.post(route, json=body)
        self.assertEqual(resp.status_code, 202)
        job_id = resp.get_json()["job_id"]
        deadline = time.time() + 60
        while time.time() < deadline:
            data = self.client.get(f"/job_status/{job_id}").get_json()
            if data.get("status") in ("completed", "failed"):
                self.assertEqual(data["status"], "completed", data.get("error"))
                return data
            time.sleep(0.01)
        self.fail(f"Job {job_id} timed out")

    def test_export_then_import_relinks_lineage(self):
        files = self.tmp / "project" / "process"
        files.mkdir(parents=True)
        _write_wav(files / "source.wav", 440.0)
        _write_wav(files / "gained.wav", 660.0)
        source = _audio("source.xxh3_64", files / "source.wav", {})
        origin = _audio("gained.xxh3_64", files / "gained.wav",
                        {"operation": "gain", "params": {"gain_db": 3.0}, "source_id": source.id})
        self.graph.add_element(source)
        self.graph.add_element(origin)
        self.graph.link(source, origin, relation="source")
        self.graph.save()

        export_dir = self.tmp / "exported"
        self._run("/export", {"names": [origin.id], "custom_path": str(export_dir)})

        self.graph.add_element(LocalPath(id="exported_path", name="exported", path=str(export_dir)))
        self._run("/expand_path", {"path_node_id": "exported_path"})

        imported = [n for n, d in self.graph.G.nodes(data=True)
                    if d.get("type") == "audio" and n not in (source.id, origin.id)]
        self.assertEqual(len(imported), 1)
        copy_id = imported[0]
        context = self.graph.G.nodes[copy_id]["context"]
        self.assertEqual(context[ORIGIN_KEY], origin.id)
        self.assertEqual(context["source_id"], source.id)
        self.assertTrue(self.graph.G.has_edge(origin.id, copy_id))
        self.assertTrue(self.graph.G.has_edge(source.id, copy_id))


if __name__ == "__main__":
    unittest.main()
