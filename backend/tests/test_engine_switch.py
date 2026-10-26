import asyncio
import socket
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from aiohttp import web
from aiohttp.test_utils import TestServer

import app as app_module
from app import app
from engine.engine_provider import EngineProvider
from engine.remote_engine import check_engine_service


def _fake_local(data_root=None):
    engine = MagicMock(name="LocalEngine", spec=["data_root"])
    engine.data_root = Path(data_root or "tmp")
    return engine


def _fake_remote(remote_url, timeout=300, data_root=None):
    engine = MagicMock(name="RemoteEngine", spec=["data_root", "remote_url"])
    engine.data_root = Path(data_root or "tmp")
    engine.remote_url = remote_url
    return engine


class EngineSwitchTestBase(unittest.TestCase):
    def setUp(self):
        patchers = [
            patch("engine.engine_provider.LocalEngine", side_effect=_fake_local),
            patch("engine.engine_provider.RemoteEngine", side_effect=_fake_remote),
        ]
        self.local_cls, self.remote_cls = [p.start() for p in patchers]
        for p in patchers:
            self.addCleanup(p.stop)
        self.addCleanup(setattr, EngineProvider, "_engine_instance", EngineProvider._engine_instance)
        self.addCleanup(setattr, EngineProvider, "_engines", EngineProvider._engines)
        EngineProvider._engine_instance = None


class TestEngineProviderSwitch(EngineSwitchTestBase):
    def test_switch_keeps_engines_for_reuse(self):
        provider = EngineProvider(data_root="project")
        local = provider.get_engine()
        self.assertIsNone(provider.remote_url)

        remote = provider.switch("http://gpu:5001")
        self.assertEqual(provider.remote_url, "http://gpu:5001")
        self.assertEqual(remote.data_root, Path("project"))

        self.assertIs(provider.switch(None), local)
        self.assertIs(provider.switch("http://gpu:5001"), remote)
        self.assertEqual(self.local_cls.call_count, 1)
        self.assertEqual(self.remote_cls.call_count, 1)

    def test_set_data_root_retargets_every_engine(self):
        provider = EngineProvider(data_root="a")
        provider.switch("http://gpu:5001")
        with patch("pathlib.Path.mkdir"):
            provider.set_data_root("b")
        self.assertEqual(provider.get_engine().data_root, Path("b"))
        self.assertEqual(provider.switch(None).data_root, Path("b"))

    def test_reset_discards_kept_engines(self):
        EngineProvider(remote_url="http://gpu:5001", data_root="a").switch(None)
        EngineProvider._engine_instance = None
        provider = EngineProvider(remote_url="http://gpu:5001", data_root="b")
        self.assertEqual(provider.get_engine().data_root, Path("b"))
        self.assertEqual(self.remote_cls.call_count, 2)


class TestEngineEndpoint(EngineSwitchTestBase):
    def setUp(self):
        super().setUp()
        for name in ("engine_provider", "execution_url", "remote_engine_url"):
            self.addCleanup(setattr, app_module, name, getattr(app_module, name))
        self.addCleanup(app_module.active_jobs.clear)
        self.addCleanup(app_module.local_jobs.clear)
        app_module.active_jobs.clear()
        app_module.local_jobs.clear()
        app_module.execution_url = None
        app_module.remote_engine_url = "http://gpu:5001"
        app_module.engine_provider = EngineProvider(data_root="project")
        self.client = app.test_client()

    def test_switch_to_remote_and_back(self):
        res = self.client.post("/engine", json={"mode": "remote"})
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.get_json()["mode"], "remote")
        self.assertEqual(app_module.engine_provider.remote_url, "http://gpu:5001")

        res = self.client.post("/engine", json={"mode": "local"})
        self.assertEqual(res.get_json()["mode"], "local")
        # The remote URL is remembered for switching back
        self.assertEqual(res.get_json()["remote_url"], "http://gpu:5001")
        self.assertIsNone(app_module.engine_provider.remote_url)

    def test_new_remote_url_is_normalized_and_remembered(self):
        res = self.client.post("/engine", json={"mode": "remote", "remote_url": " http://other:5001/ "})
        self.assertEqual(res.status_code, 200)
        self.assertEqual(app_module.engine_provider.remote_url, "http://other:5001")
        self.assertEqual(self.client.get("/engine").get_json()["remote_url"], "http://other:5001")

    def test_remote_without_url_is_rejected(self):
        app_module.remote_engine_url = None
        res = self.client.post("/engine", json={"mode": "remote"})
        self.assertEqual(res.status_code, 400)
        self.assertIsNone(app_module.engine_provider.remote_url)

    def test_unfinished_jobs_block_switch_unless_forced(self):
        app_module.active_jobs["engine-job"] = {}
        app_module.local_jobs["host-job"] = {"status": "running"}
        app_module.local_jobs["done-job"] = {"status": "completed"}

        res = self.client.post("/engine", json={"mode": "remote"})
        self.assertEqual(res.status_code, 409)
        self.assertEqual(res.get_json()["unfinished_jobs"], 2)
        self.assertIsNone(app_module.engine_provider.remote_url)

        res = self.client.post("/engine", json={"mode": "remote", "force": True})
        self.assertEqual(res.status_code, 200)
        self.assertEqual(app_module.engine_provider.remote_url, "http://gpu:5001")

    def test_reselecting_active_engine_ignores_unfinished_jobs(self):
        app_module.active_jobs["engine-job"] = {}
        res = self.client.post("/engine", json={"mode": "local"})
        self.assertEqual(res.status_code, 200)

    def test_choice_without_project_applies_on_next_load(self):
        app_module.engine_provider = None
        res = self.client.post("/engine", json={"mode": "remote"})
        self.assertEqual(res.status_code, 200)
        with patch("pathlib.Path.mkdir"):
            app_module.initialize_engine("project")
        self.assertEqual(app_module.engine_provider.remote_url, "http://gpu:5001")

    def test_project_load_reuses_engines(self):
        with patch("pathlib.Path.mkdir"):
            app_module.initialize_engine("first")
            local = app_module.engine_provider.get_engine()
            app_module.initialize_engine("second")
        self.assertIs(app_module.engine_provider.get_engine(), local)
        self.assertEqual(local.data_root, Path("second"))
        self.assertEqual(self.local_cls.call_count, 1)

    def test_check_probes_only_the_active_remote_engine(self):
        probe = AsyncMock(return_value={"reachable": True, "device": "cuda"})
        with patch.object(app_module, "check_engine_service", probe):
            self.assertNotIn("connection", self.client.get("/engine?check=1").get_json())
            self.client.post("/engine", json={"mode": "remote"})
            self.assertNotIn("connection", self.client.get("/engine").get_json())
            res = self.client.get("/engine?check=1").get_json()
        self.assertEqual(res["connection"], {"reachable": True, "device": "cuda"})
        probe.assert_awaited_once_with("http://gpu:5001")


class TestCheckEngineService(unittest.TestCase):
    def _probe(self, handler) -> dict:
        async def run():
            web_app = web.Application()
            web_app.router.add_get("/health", handler)
            async with TestServer(web_app) as server:
                return await check_engine_service(str(server.make_url("")).rstrip("/"))
        return asyncio.run(run())

    def test_healthy_service(self):
        result = self._probe(lambda _: web.json_response({"status": "healthy", "device": "cuda"}))
        self.assertEqual(result, {"reachable": True, "device": "cuda"})

    def test_login_redirect(self):
        async def handler(_):
            raise web.HTTPFound("https://example.cloudflareaccess.com/login")
        result = self._probe(handler)
        self.assertFalse(result["reachable"])
        self.assertIn("Cloudflare Access", result["error"])

    def test_error_status(self):
        result = self._probe(lambda _: web.Response(status=502))
        self.assertEqual(result, {"reachable": False, "error": "Engine service responded with HTTP 502."})

    def test_non_engine_response(self):
        result = self._probe(lambda _: web.Response(text="<html>hello</html>"))
        self.assertEqual(result["error"], "The response isn't from an engine service.")

    def test_unreachable(self):
        # A port that was free a moment ago, so nothing is listening on it
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        result = asyncio.run(check_engine_service(f"http://127.0.0.1:{port}"))
        self.assertFalse(result["reachable"])
        self.assertTrue(result["error"].startswith("Cannot connect"))


if __name__ == "__main__":
    unittest.main()
