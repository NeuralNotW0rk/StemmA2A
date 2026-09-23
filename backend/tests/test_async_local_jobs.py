import os
import json
import time
import shutil
import tempfile
import unittest
from unittest.mock import AsyncMock, MagicMock
from pathlib import Path
import numpy as np
import soundfile as sf

import app as app_module
from app import app
from param_graph.graph import ParameterGraph
from param_graph.elements.artifacts.audio_element import Audio
from param_graph.elements.base_elements import Asset
from param_graph.elements.local_path import LocalPath
from engine.local_engine import LocalEngine
from engine.engine_provider import EngineProvider

class TestAsyncLocalJobs(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.old_graph = app_module.param_graph
        self.old_provider = app_module.engine_provider
        self.old_local_jobs = dict(app_module.local_jobs)

        # Set up a clean test graph and engine provider
        self.test_graph = ParameterGraph(self.tmp_dir)
        self.test_graph.project_name = "AsyncJobsTestProject"
        self.test_graph.save()
        app_module.param_graph = self.test_graph

        EngineProvider._engine_instance = None
        self.test_provider = EngineProvider(data_root=self.tmp_dir)
        app_module.engine_provider = self.test_provider
        self.old_trigger = app_module.trigger_embedding_update
        app_module.trigger_embedding_update = MagicMock()

        self.client = app.test_client()

    def tearDown(self):
        app_module.param_graph = self.old_graph
        app_module.engine_provider = self.old_provider
        app_module.local_jobs = self.old_local_jobs
        app_module.trigger_embedding_update = self.old_trigger
        try:
            shutil.rmtree(self.tmp_dir)
        except Exception:
            pass

    def _poll_job(self, job_id: str, timeout_sec: float = 60.0) -> dict:
        start_time = time.time()
        while time.time() - start_time < timeout_sec:
            resp = self.client.get(f"/job_status/{job_id}")
            self.assertEqual(resp.status_code, 200)
            data = resp.get_json()
            if data.get("status") in ("completed", "failed"):
                return data
            time.sleep(0.01)
        self.fail(f"Job {job_id} timed out after {timeout_sec}s")

    def test_async_register_model(self):
        """Test asynchronous model registration and layer extraction."""
        payload = {
            "name": "Async Test StyleGAN",
            "model_type": "stylegan2",
            "checkpoint_path": "",
            "size": 256,
            "channel_multiplier": 2
        }
        resp = self.client.post("/register_model", json=payload)
        self.assertEqual(resp.status_code, 202)
        resp_data = resp.get_json()
        self.assertTrue(resp_data["success"])
        job_id = resp_data["job_id"]

        job_data = self._poll_job(job_id)
        self.assertEqual(job_data["status"], "completed", f"Job failed with error: {job_data.get('error')}")
        self.assertIn("model", job_data["result"])
        model_element = job_data["result"]["model"]
        self.assertTrue(model_element["id"].startswith("stylegan2_") or model_element["id"].startswith("model_"))

        # Verify model exists in graph
        self.assertTrue(self.test_graph.G.has_node(model_element["id"]))

    def test_async_create_grating(self):
        """Test asynchronous grating creation with feature clustering."""
        # First register a model
        reg_resp = self.client.post("/register_model", json={
            "name": "Grating Target StyleGAN",
            "model_type": "stylegan2",
            "checkpoint_path": "",
            "size": 256,
            "channel_multiplier": 2
        })
        model_job_id = reg_resp.get_json()["job_id"]
        model_job = self._poll_job(model_job_id)
        model_id = model_job["result"]["id"]

        # Mock heavy clustering computation on CPU
        engine = self.test_provider.get_engine()
        engine.cluster_features = AsyncMock(return_value=[i % 2 for i in range(512)])

        # Request grating creation
        payload = {
            "model_id": model_id,
            "name": "Async Grating",
            "elements": [
                {
                    "address": "conv1.conv",
                    "kernel_type": "erode",
                    "params": {"radius": 2},
                    "perform_clustering": True,
                    "num_clusters": 2,
                    "cluster": 1
                }
            ]
        }
        resp = self.client.post("/create_grating", json=payload)
        self.assertEqual(resp.status_code, 202)
        job_id = resp.get_json()["job_id"]

        job_data = self._poll_job(job_id)
        self.assertEqual(job_data["status"], "completed", f"Job failed with error: {job_data.get('error')}")
        grating = job_data["result"]["grating"]
        self.assertTrue(grating["id"].startswith("grating_"))
        self.assertEqual(grating["base_model_id"], model_id)
        self.assertTrue(self.test_graph.G.has_node(grating["id"]))

    def test_async_export_shared_model(self):
        """Test asynchronous export of model to shared catalog config."""
        reg_resp = self.client.post("/register_model", json={
            "name": "Export Target StyleGAN",
            "model_type": "stylegan2",
            "checkpoint_path": "",
            "size": 256,
            "channel_multiplier": 2
        })
        model_job = self._poll_job(reg_resp.get_json()["job_id"])
        model_id = model_job["result"]["id"]

        resp = self.client.post("/export_shared_model", json={"model_id": model_id})
        self.assertEqual(resp.status_code, 202)
        job_id = resp.get_json()["job_id"]

        job_data = self._poll_job(job_id)
        self.assertEqual(job_data["status"], "completed", f"Job failed with error: {job_data.get('error')}")
        self.assertIn("Successfully exported", job_data["result"]["message"])

    def test_async_expand_path_and_export_audio(self):
        """Test asynchronous directory scanning (expand_path) and audio export."""
        # Create audio files in a subfolder
        audio_dir = Path(self.tmp_dir) / "source_wavs"
        audio_dir.mkdir(parents=True, exist_ok=True)

        sample_rate = 16000
        t = np.linspace(0, 0.5, int(sample_rate * 0.5), endpoint=False)
        audio_data1 = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        audio_data2 = (0.5 * np.sin(2 * np.pi * 880 * t)).astype(np.float32)

        wav1 = audio_dir / "test1.wav"
        wav2 = audio_dir / "test2.wav"
        sf.write(str(wav1), audio_data1, sample_rate)
        sf.write(str(wav2), audio_data2, sample_rate)

        # Add LocalPath node to graph
        path_node_id = "local_path_1"
        local_path = LocalPath(id=path_node_id, name="source_wavs", path=str(audio_dir))
        self.test_graph.add_element(local_path)
        self.test_graph.save()

        # Test POST /expand_path
        resp = self.client.post("/expand_path", json={"path_node_id": path_node_id})
        self.assertEqual(resp.status_code, 202)
        job_id = resp.get_json()["job_id"]

        job_data = self._poll_job(job_id)
        self.assertEqual(job_data["status"], "completed", f"Job failed with error: {job_data.get('error')}")
        self.assertIn("directory_id", job_data["result"])
        directory_id = job_data["result"]["directory_id"]
        self.assertTrue(self.test_graph.G.has_node(directory_id))

        # Check audio nodes were created
        audio_nodes = [
            n for n, d in self.test_graph.G.nodes(data=True)
            if d.get("type") == "audio"
        ]
        self.assertEqual(len(audio_nodes), 2)
        audio_names = [self.test_graph.G.nodes[n]["name"] for n in audio_nodes]

        # Test POST /export with the discovered audio names
        export_out_dir = Path(self.tmp_dir) / "exported_wavs"
        resp = self.client.post("/export", json={
            "names": audio_names,
            "custom_path": str(export_out_dir)
        })
        self.assertEqual(resp.status_code, 202)
        export_job_id = resp.get_json()["job_id"]

        export_job_data = self._poll_job(export_job_id)
        self.assertEqual(export_job_data["status"], "completed", f"Job failed with error: {export_job_data.get('error')}")
        self.assertTrue(export_out_dir.exists())
        self.assertTrue((export_out_dir / "test1.wav").exists())
        self.assertTrue((export_out_dir / "test2.wav").exists())

    def test_async_rescan_source(self):
        """Test asynchronous source rescanning."""
        audio_dir = Path(self.tmp_dir) / "rescan_wavs"
        audio_dir.mkdir(parents=True, exist_ok=True)

        sample_rate = 16000
        t = np.linspace(0, 0.2, int(sample_rate * 0.2), endpoint=False)
        audio_data1 = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        wav1 = audio_dir / "item1.wav"
        sf.write(str(wav1), audio_data1, sample_rate)

        # Add external source
        resp = self.client.post("/add_external_source", json={"source_path": str(audio_dir)})
        self.assertEqual(resp.status_code, 200)

        # Add another file to audio_dir
        audio_data2 = (0.5 * np.sin(2 * np.pi * 660 * t)).astype(np.float32)
        wav2 = audio_dir / "item2.wav"
        sf.write(str(wav2), audio_data2, sample_rate)

        # Rescan source
        rescan_resp = self.client.post("/rescan_source", json={"source_name": "rescan_wavs"})
        self.assertEqual(rescan_resp.status_code, 202)
        rescan_job_id = rescan_resp.get_json()["job_id"]

        rescan_job = self._poll_job(rescan_job_id)
        self.assertEqual(rescan_job["status"], "completed", f"Job failed with error: {rescan_job.get('error')}")

    def test_async_update_embeddings(self):
        """Test asynchronous embedding generation."""
        # Create an audio node in the graph
        audio_dir = Path(self.tmp_dir) / "embed_wavs"
        audio_dir.mkdir(parents=True, exist_ok=True)
        wav_path = audio_dir / "embed1.wav"
        sample_rate = 16000
        t = np.linspace(0, 0.2, int(sample_rate * 0.2), endpoint=False)
        audio_data = (0.5 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
        sf.write(str(wav_path), audio_data, sample_rate)

        # Store in CAS storage
        cas_path = Path(self.tmp_dir) / "cas_embed1.wav"
        shutil.copyfile(str(wav_path), str(cas_path))

        audio_node = Audio(
            id="audio_embed_1",
            name="embed1",
            file=Asset(path=str(cas_path), uid="embed1_uid"),
            sample_rate=sample_rate,
            duration=0.2,
            context={}
        )
        self.test_graph.add_element(audio_node)
        self.test_graph.save()

        # Restore real embedding update for this specific test
        app_module.trigger_embedding_update = self.old_trigger

        resp = self.client.post("/update_embeddings")
        self.assertEqual(resp.status_code, 202)
        job_id = resp.get_json()["job_id"]

        job_data = self._poll_job(job_id, timeout_sec=25.0)
        self.assertEqual(job_data["status"], "completed", f"Job failed with error: {job_data.get('error')}")

if __name__ == "__main__":
    unittest.main()
