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


if __name__ == "__main__":
    unittest.main()
