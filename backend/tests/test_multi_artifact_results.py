import asyncio
import os
import shutil
import tempfile
import unittest
import uuid
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import torch

import app as app_module
from engine.engine_provider import EngineProvider
from engine.remote_engine import RemoteEngine
from engine.results import (
    ArtifactLink,
    OperationResult,
    is_multi_result,
    map_result_elements,
    result_artifacts,
    root_ids,
    serialize_result,
)
from param_graph.elements.artifacts.grating_element import Grating
from param_graph.elements.artifacts.image_element import Image
from param_graph.elements.artifacts.latent_element import Latent
from param_graph.elements.base_elements import Asset
from param_graph.elements.collections.group_element import Group
from param_graph.elements.models.stylegan_element import StyleGANModel
from param_graph.graph import ParameterGraph


def _latent(id_, path):
    return Latent(id=id_, name="pivot", context={"inversion_method": "pti"}, file=Asset(path=path, uid=id_, extension=".pt"))


def _grating(id_, path, model_id, pivot_id):
    return Grating(
        id=id_, name="pti_grating", context={"pivot_latent": pivot_id},
        file=Asset(path=path, uid=id_, extension=".safetensors"), base_model_id=model_id, elements=[]
    )


class TestResultHelpers(unittest.TestCase):
    def setUp(self):
        self.latent = _latent("lat_1", "/tmp/lat_1.pt")
        self.grating = _grating("grt_1", "/tmp/grt_1.safetensors", "model_1", "lat_1")
        self.multi = {"artifacts": [self.latent, self.grating], "links": [ArtifactLink("lat_1", "grt_1", "pivot")]}

    def test_serialize_and_split(self):
        data = serialize_result(self.multi)
        self.assertTrue(is_multi_result(data))
        self.assertEqual([a["id"] for a in data["artifacts"]], ["lat_1", "grt_1"])
        self.assertEqual(data["links"], [{"source": "lat_1", "target": "grt_1", "relation": "pivot"}])
        self.assertEqual(root_ids(data), ["lat_1"])

        primary, others, links = result_artifacts(data)
        self.assertEqual(primary["id"], "lat_1")
        self.assertEqual([a["id"] for a in others], ["grt_1"])
        self.assertEqual(len(links), 1)

    def test_single_results_are_unchanged(self):
        single = serialize_result(self.latent)
        self.assertFalse(is_multi_result(single))
        self.assertEqual(result_artifacts(single), (single, [], []))
        self.assertEqual(result_artifacts({"artifact": single}), (single, [], []))

    def test_map_result_elements(self):
        tag = lambda d: {**d, "name": "mapped"}
        mapped = map_result_elements(serialize_result(self.multi), tag)
        self.assertEqual([a["name"] for a in mapped["artifacts"]], ["mapped", "mapped"])
        self.assertEqual(len(mapped["links"]), 1)
        self.assertEqual(map_result_elements(serialize_result(self.latent), tag)["name"], "mapped")
        self.assertEqual(map_result_elements({"status": "x"}, tag), {"status": "x"})


class TestLocalEngineMultiArtifactInvert(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        EngineProvider._engine_instance = None
        self.engine = EngineProvider(data_root=self.tmp).get_engine()

    def tearDown(self):
        EngineProvider._engine_instance = None
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_invert_persists_every_output_and_serializes(self):
        src_dir = tempfile.mkdtemp(dir=self.tmp)
        grating_path = os.path.join(src_dir, "grt.safetensors")
        Path(grating_path).write_bytes(b"grating-bytes")
        w = torch.randn(1, 8)
        result = OperationResult(
            outputs=[(_latent("lat_2", ""), w), (_grating("grt_2", grating_path, "m", "lat_2"), None)],
            links=[ArtifactLink("lat_2", "grt_2", "pivot")]
        )
        adapter = MagicMock()
        adapter.invert.return_value = result

        with patch.object(self.engine, "_resolve_model_element", side_effect=lambda m: m), \
             patch.object(self.engine, "_get_adapter_class", return_value=object), \
             patch.object(self.engine.model_cache, "get", return_value=adapter):
            out = asyncio.run(self.engine._invert_logic(model_element=MagicMock(adapter="stylegan2")))

        latent, grating = out["artifacts"]
        self.assertTrue(torch.equal(torch.load(latent.file.path), w))
        self.assertEqual(Path(grating.file.path).read_bytes(), b"grating-bytes")
        self.assertTrue(latent.file.path.startswith(self.tmp) and grating.file.path.startswith(self.tmp))

        self.engine.job_statuses["job_multi"] = {"status": "completed", "result": out}
        status = asyncio.run(self.engine.get_job_status("job_multi"))
        self.assertEqual([a["id"] for a in status["result"]["artifacts"]], ["lat_2", "grt_2"])
        self.assertEqual(status["result"]["links"][0]["relation"], "pivot")


class TestRemoteEngineMultiArtifactDownload(unittest.IsolatedAsyncioTestCase):
    async def test_downloads_every_artifact(self):
        engine = RemoteEngine(remote_url="http://mock-remote:5001", data_root=tempfile.mkdtemp())
        token = uuid.uuid4().hex[:8]
        ids = [f"lat_{token}", f"grt_{token}"]
        downloaded = []

        async def mock_download(asset_id, local_path, session):
            downloaded.append(asset_id)
            local_path.parent.mkdir(parents=True, exist_ok=True)
            local_path.write_bytes(b"data")
            return True

        engine._download_single_asset = mock_download
        result = serialize_result({
            "artifacts": [_latent(ids[0], f"cache/{ids[0][:2]}/{ids[0]}"), _grating(ids[1], f"cache/{ids[1][:2]}/{ids[1]}", "m", ids[0])],
            "links": [ArtifactLink(ids[0], ids[1], "pivot")]
        })

        response = AsyncMock()
        response.raise_for_status = MagicMock()
        response.json = AsyncMock(return_value={"status": "completed", "result": result})
        get_ctx = MagicMock()
        get_ctx.__aenter__ = AsyncMock(return_value=response)
        get_ctx.__aexit__ = AsyncMock(return_value=None)
        session = MagicMock()
        session.__aenter__ = AsyncMock(return_value=session)
        session.__aexit__ = AsyncMock(return_value=None)
        session.get = MagicMock(return_value=get_ctx)

        tmp_root = Path(app_module.__file__).parent / "tmp"
        try:
            with patch("aiohttp.ClientSession", return_value=session):
                status = await engine.get_job_status(f"job_{token}")
            self.assertEqual(status["status"], "completed")
            self.assertEqual(sorted(downloaded), sorted(ids))
            anchored = status["result"]["artifacts"]
            self.assertEqual([a["id"] for a in anchored], ids)
            self.assertTrue(all(os.path.exists(a["file"]["path"]) for a in anchored))
            self.assertEqual(status["result"]["links"][0]["relation"], "pivot")
        finally:
            for asset_id in ids:
                path = tmp_root / "cache" / asset_id[:2] / asset_id
                if path.exists():
                    path.unlink()


class TestJobStatusMultiArtifact(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.graph = ParameterGraph(self.tmp)
        EngineProvider._engine_instance = None
        self.provider = EngineProvider(data_root=self.tmp)
        self.engine = self.provider.get_engine()

        self.model = StyleGANModel(
            id="model_pti", name="Model", context={},
            checkpoint=Asset(path="ckpt", uid="ckpt_uid", extension=".pt"), adapter="stylegan2"
        )
        self.image = Image(id="img_src", name="source", context={}, file=Asset(path="src.png", uid="img_src", extension=".png"))
        self.graph.add_element(self.model)
        self.graph.add_element(self.image)
        self.graph.save()

        self.old = (app_module.param_graph, app_module.engine_provider, app_module.trigger_embedding_update, dict(app_module.active_jobs))
        app_module.param_graph = self.graph
        app_module.engine_provider = self.provider
        app_module.trigger_embedding_update = MagicMock()
        self.client = app_module.app.test_client()

    def tearDown(self):
        app_module.param_graph, app_module.engine_provider, app_module.trigger_embedding_update, active = self.old
        app_module.active_jobs.clear()
        app_module.active_jobs.update(active)
        EngineProvider._engine_instance = None
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _complete(self, job_id, group_id=None):
        files = {}
        for name in ("lat_3", "grt_3"):
            files[name] = os.path.join(self.tmp, f"{name}.bin")
            Path(files[name]).write_bytes(name.encode())
        result = serialize_result({
            "artifacts": [_latent("lat_3", files["lat_3"]), _grating("grt_3", files["grt_3"], "model_pti", "lat_3")],
            "links": [ArtifactLink("lat_3", "grt_3", "pivot")]
        })
        self.engine.get_job_status = AsyncMock(return_value={"status": "completed", "result": result})
        app_module.active_jobs[job_id] = {
            "parent_id": None,
            "group_id": group_id,
            "linked_elements": [self.model, self.image],
            "validated_params": {"model_id": "model_pti", "operation": "invert"},
            "operation": "invert",
            "room": None,
        }
        return self.client.get(f"/job_status/{job_id}")

    def _relations(self, source, target):
        return {attrs.get("relation") for u, v, attrs in self.graph.G.edges(data=True) if u == source and v == target}

    def test_both_artifacts_are_stored_and_linked(self):
        resp = self._complete("job_pti")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertEqual(data["node_id"], "lat_3")
        self.assertEqual([a["id"] for a in data["artifacts"]], ["lat_3", "grt_3"])

        self.assertTrue(self.graph.G.has_node("lat_3") and self.graph.G.has_node("grt_3"))
        self.assertEqual(self._relations("img_src", "lat_3"), {"source"})
        self.assertEqual(self._relations("model_pti", "lat_3"), {"source"})
        self.assertEqual(self._relations("lat_3", "grt_3"), {"pivot"})
        self.assertEqual(self._relations("model_pti", "grt_3"), {"binds_to"})
        # The grating depends on the image only through its pivot
        self.assertEqual(self._relations("img_src", "grt_3"), set())

        grating = self.graph.get_element("grt_3")
        self.assertEqual(grating.context["operation"], "invert")
        self.assertTrue(os.path.exists(grating.file.path))
        self.assertNotEqual(grating.file.path, self.graph.get_element("lat_3").file.path)

    def test_only_roots_join_the_batch_group(self):
        self.graph.add_element(Group(id="grp_pti", member_type="latent"))
        self.graph.save()
        self._complete("job_pti_grouped", group_id="grp_pti")
        members = self.graph.G.nodes["grp_pti"].get("member_ids")
        self.assertEqual(members, ["lat_3"])
        self.assertEqual(self.graph.G.nodes["lat_3"].get("parent"), "grp_pti")
        self.assertIsNone(self.graph.G.nodes["grt_3"].get("parent"))


if __name__ == "__main__":
    unittest.main()
