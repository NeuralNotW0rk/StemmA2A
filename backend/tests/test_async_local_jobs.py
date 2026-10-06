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

    def test_async_group_export_with_dynamic_labels(self):
        """Test asynchronous group export where exported filenames are based on simplified dynamic labels."""
        audio_dir = Path(self.tmp_dir) / "group_source_wavs"
        audio_dir.mkdir(parents=True, exist_ok=True)

        sample_rate = 16000
        t = np.linspace(0, 0.1, int(sample_rate * 0.1), endpoint=False)
        audio_data = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

        wav1 = audio_dir / "orig1.wav"
        wav2 = audio_dir / "orig2.wav"
        wav3 = audio_dir / "orig3.wav"
        sf.write(str(wav1), audio_data, sample_rate)
        sf.write(str(wav2), audio_data, sample_rate)
        sf.write(str(wav3), audio_data, sample_rate)

        # Create audio nodes in CAS cache with varying seed and shared prompt
        m1 = Audio(
            id="m1_uid",
            name="orig1",
            file=Asset(path=str(wav1), uid="m1_uid"),
            sample_rate=sample_rate,
            context={"prompt": "acid lead", "seed": 5001, "steps": 40}
        )
        m2 = Audio(
            id="m2_uid",
            name="orig2",
            file=Asset(path=str(wav2), uid="m2_uid"),
            sample_rate=sample_rate,
            context={"prompt": "acid lead", "seed": 5002, "steps": 40}
        )
        m3 = Audio(
            id="m3_uid",
            name="orig3",
            file=Asset(path=str(wav3), uid="m3_uid"),
            sample_rate=sample_rate,
            context={"prompt": "acid lead", "seed": 5003, "steps": 40}
        )
        from param_graph.elements.collections.group_element import Group
        grp = Group(id="grp_export_test", member_ids=["m1_uid", "m2_uid", "m3_uid"], member_type="audio")

        self.test_graph.add_element(m1)
        self.test_graph.add_element(m2)
        self.test_graph.add_element(m3)
        self.test_graph.add_element(grp)
        self.test_graph.save()

        # 1. Export by group ID
        export_out_dir = Path(self.tmp_dir) / "group_export_out"
        resp = self.client.post("/export", json={
            "names": ["grp_export_test"],
            "custom_path": str(export_out_dir)
        })
        self.assertEqual(resp.status_code, 202)
        job_id = resp.get_json()["job_id"]
        job_data = self._poll_job(job_id)
        self.assertEqual(job_data["status"], "completed", f"Job failed: {job_data.get('error')}")

        self.assertTrue(export_out_dir.exists())
        self.assertTrue((export_out_dir / "seed_5001.wav").exists())
        self.assertTrue((export_out_dir / "seed_5002.wav").exists())
        self.assertTrue((export_out_dir / "seed_5003.wav").exists())

        # 2. Export by member IDs directly
        export_out_dir2 = Path(self.tmp_dir) / "group_export_members_out"
        resp2 = self.client.post("/export", json={
            "names": ["m1_uid", "m2_uid"],
            "custom_path": str(export_out_dir2)
        })
        self.assertEqual(resp2.status_code, 202)
        job_id2 = resp2.get_json()["job_id"]
        job_data2 = self._poll_job(job_id2)
        self.assertEqual(job_data2["status"], "completed", f"Job failed: {job_data2.get('error')}")

        self.assertTrue((export_out_dir2 / "seed_5001.wav").exists())
        self.assertTrue((export_out_dir2 / "seed_5002.wav").exists())

        # 3. Export a group with long multi-parameter dynamic labels (untruncated info in filename)
        long_m1 = Audio(
            id="long_m1_uid",
            name="long1",
            file=Asset(path=str(wav1), uid="long_m1_uid"),
            sample_rate=sample_rate,
            context={"prompt": "acid lead", "seed": 7001, "steps": 50, "cfg_scale": 7.5, "noise_level": 0.65}
        )
        long_m2 = Audio(
            id="long_m2_uid",
            name="long2",
            file=Asset(path=str(wav2), uid="long_m2_uid"),
            sample_rate=sample_rate,
            context={"prompt": "acid lead", "seed": 7002, "steps": 80, "cfg_scale": 9.0, "noise_level": 0.85}
        )
        grp_long = Group(id="grp_export_long", member_ids=["long_m1_uid", "long_m2_uid"], member_type="audio")
        self.test_graph.add_element(long_m1)
        self.test_graph.add_element(long_m2)
        self.test_graph.add_element(grp_long)
        self.test_graph.save()

        export_out_dir3 = Path(self.tmp_dir) / "group_export_long_out"
        resp3 = self.client.post("/export", json={
            "names": ["grp_export_long"],
            "custom_path": str(export_out_dir3)
        })
        self.assertEqual(resp3.status_code, 202)
        job_id3 = resp3.get_json()["job_id"]
        job_data3 = self._poll_job(job_id3)
        self.assertEqual(job_data3["status"], "completed", f"Job failed: {job_data3.get('error')}")

        self.assertTrue(export_out_dir3.exists())
        exported_files = [p.name for p in export_out_dir3.iterdir()]
        # Verify that all varying parameter names and values are in the filename (untruncated)
        self.assertTrue(any("cfg_scale_7.5" in f and "noise_level_0.65" in f and "seed_7001" in f and "steps_50" in f for f in exported_files))
        self.assertTrue(any("cfg_scale_9.0" in f and "noise_level_0.85" in f and "seed_7002" in f and "steps_80" in f for f in exported_files))

    def test_concurrent_embedding_triggers_coalescing(self):
        """Verify that rapid concurrent trigger_embedding_update calls coalesce into a single worker loop without parallel redundant loads."""
        # Restore real trigger
        app_module.trigger_embedding_update = self.old_trigger

        # Call trigger_embedding_update 20 times concurrently from multiple threads
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(app_module.trigger_embedding_update, False, True, None) for _ in range(20)]
            for f in futures:
                f.result()

        # Wait for worker loop to complete
        for _ in range(50):
            with app_module._embedding_worker_lock:
                if not app_module._embedding_worker_running:
                    break
            time.sleep(0.1)

        self.assertFalse(app_module._embedding_worker_running)


if __name__ == "__main__":
    unittest.main()



