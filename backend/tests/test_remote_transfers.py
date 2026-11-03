import asyncio
from concurrent.futures import ThreadPoolExecutor
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from engine.remote_engine import RemoteEngine


class TestRemoteTransferDeduplication(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.data_root = self.temp_dir.name
        self.engine = RemoteEngine(remote_url="http://mock-remote:5001", data_root=self.data_root)

    async def asyncTearDown(self):
        self.temp_dir.cleanup()

    async def test_concurrent_uploads_deduplicated(self):
        upload_call_count = 0

        async def mock_upload_single(asset_uid: str, asset_path: str, session):
            nonlocal upload_call_count
            upload_call_count += 1
            # Simulate network delay during upload
            await asyncio.sleep(0.05)
            return True

        self.engine._upload_single_asset = mock_upload_single

        mock_session = MagicMock()
        missing_uids = ["checkpoint_uid_123.xxh3_64"]
        local_assets = {"checkpoint_uid_123.xxh3_64": "/path/to/model.pkl"}

        # Simulate 8 concurrent generation requests trying to upload the same missing asset
        coros = [
            self.engine.upload_missing_assets(missing_uids, local_assets, mock_session)
            for _ in range(8)
        ]
        results = await asyncio.gather(*coros)

        # All 8 requests should succeed
        self.assertEqual(len(results), 8)
        self.assertTrue(all(results))

        # Only 1 upload task should have been dispatched over the wire
        self.assertEqual(upload_call_count, 1)

        # In-flight tracker should be empty after completion
        self.assertEqual(len(self.engine._active_uploads), 0)

    async def test_concurrent_downloads_deduplicated(self):
        download_call_count = 0
        import uuid
        unique_token = uuid.uuid4().hex[:8]
        asset_id = f"test_dl_{unique_token}.xxh3_64"

        # Ensure no leftover files from prior test runs
        tmp_root = Path(__file__).parent.parent / "tmp"
        target_path = tmp_root / "cache" / unique_token[:2] / asset_id
        if target_path.exists():
            target_path.unlink()

        async def mock_download_single(asset_id: str, local_path: Path, session):
            nonlocal download_call_count
            download_call_count += 1
            # Simulate network delay during download
            await asyncio.sleep(0.05)
            # Write a dummy file to simulate downloaded result
            local_path.parent.mkdir(parents=True, exist_ok=True)
            local_path.write_bytes(b"dummy image data")
            return True

        self.engine._download_single_asset = mock_download_single

        # Prepare mock response for get_job_status
        fake_job_id = f"test_job_{unique_token}"
        element_dict = {
            "id": asset_id,
            "type": "image",
            "name": "test_image",
            "context": {},
            "file": {
                "uid": asset_id,
                "path": f"cache/{unique_token[:2]}/{asset_id}",
                "size": 16
            },
            "width": 512,
            "height": 512,
            "metadata": {}
        }

        mock_status_response = AsyncMock()
        mock_status_response.raise_for_status = MagicMock()
        mock_status_response.json = AsyncMock(return_value={
            "status": "completed",
            "result": element_dict
        })

        mock_get_ctx = MagicMock()
        mock_get_ctx.__aenter__ = AsyncMock(return_value=mock_status_response)
        mock_get_ctx.__aexit__ = AsyncMock(return_value=None)

        mock_session = MagicMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        mock_session.get = MagicMock(return_value=mock_get_ctx)

        try:
            with patch("aiohttp.ClientSession", return_value=mock_session):
                coros = [
                    self.engine.get_job_status(fake_job_id)
                    for _ in range(6)
                ]
                results = await asyncio.gather(*coros)

            self.assertEqual(len(results), 6)
            self.assertTrue(all(r.get("status") == "completed" for r in results))

            # Only 1 download task should have been dispatched
            self.assertEqual(download_call_count, 1)

            # In-flight tracker should be empty after completion
            self.assertEqual(len(self.engine._active_downloads), 0)
        finally:
            if target_path.exists():
                try:
                    target_path.unlink()
                except Exception:
                    pass

    async def test_upload_failure_cleans_up_state(self):
        async def failing_upload_single(asset_uid: str, asset_path: str, session):
            await asyncio.sleep(0.01)
            return False

        self.engine._upload_single_asset = failing_upload_single

        mock_session = MagicMock()
        missing_uids = ["failed_uid.xxh3_64"]
        local_assets = {"failed_uid.xxh3_64": "/path/to/missing.pkl"}

        result = await self.engine.upload_missing_assets(missing_uids, local_assets, mock_session)
        self.assertFalse(result)
        # Verify tracker was cleaned up
        self.assertEqual(len(self.engine._active_uploads), 0)

    def test_multi_threaded_cross_event_loop_downloads(self):
        """
        Simulates Flask/Werkzeug multi-threaded server handling concurrent /job_status requests
        across separate OS threads each running their own independent asyncio event loop.
        """
        import uuid
        unique_token = uuid.uuid4().hex[:8]
        asset_id = f"test_mt_dl_{unique_token}.xxh3_64"

        tmp_root = Path(__file__).parent.parent / "tmp"
        target_path = tmp_root / "cache" / unique_token[:2] / asset_id
        if target_path.exists():
            target_path.unlink()

        download_count = 0
        count_lock = threading.Lock()

        async def mock_download_single(asset_id: str, local_path: Path, session):
            nonlocal download_count
            with count_lock:
                download_count += 1
            await asyncio.sleep(0.05)
            local_path.parent.mkdir(parents=True, exist_ok=True)
            local_path.write_bytes(b"multi threaded image data")
            return True

        self.engine._download_single_asset = mock_download_single

        fake_job_id = f"test_mt_job_{unique_token}"
        element_dict = {
            "id": asset_id,
            "type": "image",
            "name": "test_image",
            "context": {},
            "file": {
                "uid": asset_id,
                "path": f"cache/{unique_token[:2]}/{asset_id}",
                "size": 24
            },
            "width": 512,
            "height": 512,
            "metadata": {}
        }

        mock_status_response = AsyncMock()
        mock_status_response.raise_for_status = MagicMock()
        mock_status_response.json = AsyncMock(return_value={
            "status": "completed",
            "result": element_dict
        })

        mock_get_ctx = MagicMock()
        mock_get_ctx.__aenter__ = AsyncMock(return_value=mock_status_response)
        mock_get_ctx.__aexit__ = AsyncMock(return_value=None)

        mock_session = MagicMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        mock_session.get = MagicMock(return_value=mock_get_ctx)

        def _thread_worker() -> dict:
            # Each thread creates its own fresh event loop, matching Flask/asgiref behavior
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                with patch("aiohttp.ClientSession", return_value=mock_session):
                    return loop.run_until_complete(self.engine.get_job_status(fake_job_id))
            finally:
                loop.close()

        try:
            num_threads = 8
            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                futures = [executor.submit(_thread_worker) for _ in range(num_threads)]
                results = [f.result() for f in futures]

            self.assertEqual(len(results), num_threads)
            self.assertTrue(all(r.get("status") == "completed" for r in results))
            # Exactly 1 download performed despite 8 separate threads and loops
            self.assertEqual(download_count, 1)
            self.assertEqual(len(self.engine._active_downloads), 0)
        finally:
            if target_path.exists():
                try:
                    target_path.unlink()
                except Exception:
                    pass


class TestRemoteSharedModels(unittest.IsolatedAsyncioTestCase):
    async def test_add_shared_model_uploads_missing_files(self):
        from aiohttp import web
        from aiohttp.test_utils import TestServer

        received = []
        uploaded = []

        async def add_shared_model(request):
            received.append(await request.json())
            if not uploaded:
                return web.json_response({"error": "Missing assets", "missing_uids": ["ckpt.xxh3_64"]}, status=422)
            return web.json_response({"shared_model": {"id": "model1"}, "success": True})

        web_app = web.Application()
        web_app.router.add_post("/shared_models", add_shared_model)
        async with TestServer(web_app) as server:
            engine = RemoteEngine(remote_url=str(server.make_url("")).rstrip("/"))

            async def mock_upload_single(asset_uid, asset_path, session):
                uploaded.append((asset_uid, asset_path))
                return True

            engine._upload_single_asset = mock_upload_single
            entry = {
                "name": "Model", "adapter": "stable_audio_tools", "config": {"sample_rate": 44100},
                "checkpoint": {"uid": "ckpt.xxh3_64", "size": 7, "path": "C:/Models/model.ckpt"},
            }
            shared = await engine.add_shared_model(entry, {"ckpt.xxh3_64": "C:/Models/model.ckpt"})

        self.assertEqual(shared, {"id": "model1"})
        self.assertEqual(uploaded, [("ckpt.xxh3_64", "C:/Models/model.ckpt")])
        self.assertEqual(len(received), 2)
        # Host paths never reach the engine service
        self.assertEqual(received[0]["checkpoint"], {"uid": "ckpt.xxh3_64", "size": 7})


class TestDirectoryUpload(unittest.IsolatedAsyncioTestCase):
    """A directory goes up file by file, unarchived, and is stored as the directory on the engine."""

    async def asyncSetUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        root = Path(self.temp_dir.name)
        self.source = root / "encoder"
        (self.source / "sub").mkdir(parents=True)
        (self.source / "config.json").write_bytes(b'{"a": 1}')
        (self.source / "sub" / "weights.bin").write_bytes(b"0123456789")
        self.engine_data = root / "engine"
        self.requests: list[str] = []
        self.fail_next_member: str | None = None
        # Members after which the engine reports the asset as already present
        self.present_after: int | None = None

    async def asyncTearDown(self):
        self.temp_dir.cleanup()

    async def _upload(self, uid: str, via_engine: list | None = None) -> bool:
        from aiohttp import web
        from aiohttp.test_utils import TestServer
        from engine import uploads

        async def upload(request):
            form = await request.post()
            relpath = form.get("relpath")
            if self.present_after is not None and len(self.requests) >= self.present_after:
                return web.json_response({"already_present": True})
            self.requests.append(relpath)
            if relpath and relpath == self.fail_next_member:
                self.fail_next_member = None
                return web.json_response({"error": "dropped"}, status=502)
            uploads.write_chunk(
                self.engine_data, form["uid"], form["file"].file.read(), int(form["chunk_index"]),
                int(form["total_chunks"]), int(form["total_size"]), relpath,
            )
            return web.json_response({})

        async def finalize(request):
            body = await request.json()
            try:
                uploads.finalize_directory(self.engine_data, body["uid"], body["total_size"])
            except uploads.UploadError as e:
                return web.json_response({"error": str(e)}, status=409)
            return web.json_response({})

        web_app = web.Application()
        web_app.router.add_post("/upload", upload)
        web_app.router.add_post("/upload/finalize", finalize)
        async with TestServer(web_app) as server:
            engine = RemoteEngine(remote_url=str(server.make_url("")).rstrip("/"))
            if via_engine is not None:
                via_engine.append(engine)
            import aiohttp
            async with aiohttp.ClientSession() as session:
                with patch("engine.remote_engine.asyncio.sleep", AsyncMock()):
                    if via_engine is not None:
                        return await engine.upload_missing_assets([uid], {uid: str(self.source)}, session, "job1")
                    return await engine._upload_single_asset(uid, str(self.source), session)

    async def test_directory_is_sent_unarchived_and_stored_as_a_directory(self):
        from engine import uploads
        from utils.uid import XXH3_64

        uid = XXH3_64().from_directory(self.source)
        self.assertTrue(await self._upload(uid))
        self.assertEqual(self.requests, ["config.json", "sub/weights.bin"])
        stored = uploads.destination_path(self.engine_data, uid)
        self.assertTrue(stored.is_dir())
        self.assertEqual(XXH3_64().from_directory(stored), uid)

    async def test_retry_resumes_from_the_failed_file(self):
        from utils.uid import XXH3_64

        self.fail_next_member = "sub/weights.bin"
        self.assertTrue(await self._upload(XXH3_64().from_directory(self.source)))
        # config.json is not sent again on the retry
        self.assertEqual(self.requests, ["config.json", "sub/weights.bin", "sub/weights.bin"])

    async def test_transfers_report_progress_for_their_job(self):
        from utils.uid import XXH3_64

        uid = XXH3_64().from_directory(self.source)
        engines = []
        self.assertTrue(await self._upload(uid, via_engine=engines))
        [transfer] = engines[0].get_transfers()
        self.assertEqual(transfer["direction"], "upload")
        self.assertEqual(transfer["name"], "encoder")
        self.assertEqual(transfer["state"], "done")
        self.assertEqual((transfer["bytes_done"], transfer["total_bytes"]), (18, 18))
        self.assertEqual((transfer["files_done"], transfer["total_files"]), (2, 2))
        self.assertEqual(transfer["job_ids"], ["job1"])

        # Finished transfers drop off after a while
        with patch("engine.remote_engine.time.time", return_value=transfer["finished_at"] + 3600):
            self.assertEqual(engines[0].get_transfers(), [])

    async def test_failed_transfer_reports_its_error(self):
        engines = []
        self.assertFalse(await self._upload("0000000000000000.xxh3_64", via_engine=engines))
        [transfer] = engines[0].get_transfers()
        self.assertEqual(transfer["state"], "failed")
        self.assertIn("hash to", transfer["error"])

    async def test_upload_stops_when_the_engine_already_has_the_asset(self):
        from utils.uid import XXH3_64

        self.present_after = 1
        engines = []
        self.assertTrue(await self._upload(XXH3_64().from_directory(self.source), via_engine=engines))
        # The second file was never sent, and nothing was finalized
        self.assertEqual(self.requests, ["config.json"])
        [transfer] = engines[0].get_transfers()
        self.assertEqual(transfer["state"], "done")

    async def test_contents_that_do_not_match_the_uid_are_not_retried(self):
        from engine import uploads

        self.assertFalse(await self._upload("0000000000000000.xxh3_64"))
        self.assertEqual(self.requests, ["config.json", "sub/weights.bin"])
        self.assertFalse(uploads.destination_path(self.engine_data, "0000000000000000.xxh3_64").exists())


if __name__ == "__main__":
    unittest.main()
