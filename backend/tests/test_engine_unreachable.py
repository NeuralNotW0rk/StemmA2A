import asyncio
import socket
import tempfile
import time
import unittest
import uuid
from unittest.mock import AsyncMock, MagicMock

import aiohttp

from engine.remote_engine import RemoteEngine, _request_error
from operations.task_manager import TaskManager
from utils.errors import EngineUnreachableError, job_traceback


def _response_error(status: int) -> aiohttp.ClientResponseError:
    request_info = MagicMock(url="http://engine/execute")
    return aiohttp.ClientResponseError(request_info, (), status=status, message="Error")


class RequestErrorTest(unittest.TestCase):
    def test_connection_failure_is_unreachable(self):
        self.assertIsInstance(_request_error(aiohttp.ClientConnectionError()), EngineUnreachableError)
        self.assertIsInstance(_request_error(asyncio.TimeoutError()), EngineUnreachableError)

    def test_proxy_reporting_a_down_origin_is_unreachable(self):
        self.assertIsInstance(_request_error(_response_error(530)), EngineUnreachableError)
        self.assertIsInstance(_request_error(_response_error(502)), EngineUnreachableError)

    def test_failed_request_is_not_unreachable(self):
        error = _request_error(_response_error(500))
        self.assertNotIsInstance(error, EngineUnreachableError)
        self.assertIn("500", str(error))

    def test_execute_without_a_server_raises_unreachable(self):
        # A port that was free a moment ago, so nothing is listening on it
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        with tempfile.TemporaryDirectory() as data_root:
            engine = RemoteEngine(f"http://127.0.0.1:{port}", timeout=5, data_root=data_root)
            with self.assertRaises(EngineUnreachableError):
                asyncio.run(engine.execute("generate", job_id="job_x"))


class ResultDownloadTest(unittest.IsolatedAsyncioTestCase):
    """A finished job whose result can't be downloaded fails only when the download itself failed."""

    async def _status_with_download_reply(self, download_status: int) -> dict:
        from aiohttp import web
        from aiohttp.test_utils import TestServer

        asset_id = f"test_unreach_{uuid.uuid4().hex[:8]}.xxh3_64"
        element = {
            "id": asset_id, "type": "image", "name": "result", "context": {},
            "file": {"uid": asset_id, "path": f"cache/te/{asset_id}", "size": 16},
            "width": 8, "height": 8, "metadata": {},
        }

        async def job_status(_request: web.Request) -> web.Response:
            return web.json_response({"status": "completed", "result": element})

        async def download(_request: web.Request) -> web.Response:
            return web.Response(status=download_status, text="origin down")

        web_app = web.Application()
        web_app.router.add_get("/job_status/{job_id}", job_status)
        web_app.router.add_get("/download_asset/{asset_id}", download)
        with tempfile.TemporaryDirectory() as data_root:
            async with TestServer(web_app) as server:
                engine = RemoteEngine(str(server.make_url("")).rstrip("/"), timeout=5, data_root=data_root)
                return await engine.get_job_status("job_x")

    async def test_download_cut_off_by_the_proxy_is_retried(self):
        with self.assertRaises(EngineUnreachableError):
            await self._status_with_download_reply(530)

    async def test_download_refused_by_the_engine_fails_the_job(self):
        status = await self._status_with_download_reply(500)
        self.assertEqual(status["status"], "failed")


class JobStatusGracePeriodTest(unittest.TestCase):
    """A remote job keeps being reported as running while its engine is briefly unreachable."""

    def setUp(self):
        import app as app_module

        self.app_module = app_module
        for name in ("engine_provider", "param_graph", "REMOTE_JOB_GRACE_SECONDS"):
            self.addCleanup(setattr, app_module, name, getattr(app_module, name))
        self.addCleanup(app_module._remote_poll_state.clear)
        self.addCleanup(app_module._locally_cancelled_jobs.clear)
        self.addCleanup(app_module.active_jobs.clear)
        app_module.param_graph = MagicMock()
        self.engine = MagicMock()
        app_module.engine_provider = MagicMock(get_engine=MagicMock(return_value=self.engine))
        app_module.active_jobs["job_r"] = {"operation": "generate"}
        self.client = app_module.app.test_client()

    def _poll(self, outcome: object) -> tuple[int, dict]:
        side_effect = outcome if isinstance(outcome, Exception) else None
        self.engine.get_job_status = AsyncMock(return_value=outcome, side_effect=side_effect)
        res = self.client.get("/job_status/job_r")
        return res.status_code, res.get_json()

    def test_lost_connection_reports_running_with_last_progress(self):
        self._poll({"status": "running", "progress": {"value": 3, "total": 10, "description": "Step 3"}})

        code, body = self._poll(EngineUnreachableError("offline"))
        self.assertEqual(code, 200)
        self.assertEqual(body["status"], "running")
        self.assertTrue(body["reconnecting"])
        self.assertEqual((body["progress"]["value"], body["progress"]["total"]), (3, 10))
        self.assertIn("retrying", body["progress"]["description"])
        # The job's context is kept, so its result can still be stored once the connection is back
        self.assertIn("job_r", self.app_module.active_jobs)

        code, body = self._poll({"status": "running", "progress": {"value": 4, "total": 10}})
        self.assertEqual(body["progress"]["value"], 4)
        self.assertIsNone(self.app_module._remote_poll_state["job_r"]["unreachable_since"])

    def test_queued_job_stays_pending_while_disconnected(self):
        self._poll({"status": "pending", "progress": None})
        _, body = self._poll(EngineUnreachableError("offline"))
        self.assertEqual(body["status"], "pending")

    def test_cancel_while_disconnected_cancels_locally(self):
        self._poll(EngineUnreachableError("offline"))
        self.engine.cancel_job = AsyncMock(side_effect=EngineUnreachableError("offline"))

        res = self.client.post("/jobs/job_r/cancel")
        self.assertEqual(res.status_code, 202)
        self.assertIn("couldn't be reached", res.get_json()["message"])
        self.assertNotIn("job_r", self.app_module.active_jobs)

        # Stays cancelled, even once the engine answers again with the job's progress
        code, body = self._poll({"status": "running", "progress": {"value": 5, "total": 10}})
        self.assertEqual((code, body["status"]), (200, "cancelled"))

    def test_job_fails_once_the_grace_period_runs_out(self):
        self.app_module.REMOTE_JOB_GRACE_SECONDS = 60
        self._poll(EngineUnreachableError("offline"))
        self.app_module._remote_poll_state["job_r"]["unreachable_since"] -= 61

        code, body = self._poll(EngineUnreachableError("offline"))
        self.assertEqual(code, 500)
        self.assertEqual(body["status"], "failed")
        self.assertIn("offline", body["error"])
        self.assertIsNone(body["traceback"])
        self.assertNotIn("job_r", self.app_module.active_jobs)
        self.assertNotIn("job_r", self.app_module._remote_poll_state)


class JobTracebackTest(unittest.TestCase):
    def test_unreachable_has_no_traceback(self):
        self.assertIsNone(job_traceback(EngineUnreachableError("offline")))

    def test_other_errors_keep_their_traceback(self):
        try:
            raise ValueError("bad value")
        except ValueError as e:
            tb = job_traceback(e)
        self.assertIn("Traceback", tb)
        self.assertIn("bad value", tb)

    def test_failed_task_records_message_only(self):
        manager = TaskManager()

        def task() -> None:
            raise EngineUnreachableError("Cannot reach the remote engine service.")

        job_id = manager.submit_task(task)
        deadline = time.time() + 5
        while manager.get_job(job_id)["status"] != "failed" and time.time() < deadline:
            time.sleep(0.01)
        job = manager.get_job(job_id)
        self.assertEqual(job["status"], "failed")
        self.assertEqual(job["error"], "Cannot reach the remote engine service.")
        self.assertIsNone(job["traceback"])


if __name__ == "__main__":
    unittest.main()
