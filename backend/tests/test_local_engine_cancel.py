import asyncio
import queue
import sys
import threading
import unittest
from pathlib import Path

backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from engine.local_engine import LocalEngine


class _BlockingEngine(LocalEngine):
    """A LocalEngine with only its job queue set up (no encoders) and an operation the test controls."""

    def __init__(self):
        self.job_queue = queue.Queue()
        self.job_statuses = {}
        self.job_status_lock = threading.Lock()
        self.is_sleeping = False
        self.idle_timeout = 60
        self.started = threading.Event()
        self.release = threading.Event()
        self.fail = False
        self.worker_thread = threading.Thread(target=self._worker, daemon=True)
        self.worker_thread.start()

    async def _block_logic(self, **kwargs):
        self.started.set()
        self.release.wait(timeout=5)
        if self.fail:
            raise RuntimeError("boom")
        return "result"


class TestLocalEngineCancel(unittest.TestCase):
    def setUp(self):
        self.engine = _BlockingEngine()

    def _run(self, job_id):
        asyncio.run(self.engine.execute("block", job_id=job_id))

    def _status(self, job_id):
        return asyncio.run(self.engine.get_job_status(job_id))

    def _finish(self):
        self.engine.release.set()
        self.engine.job_queue.join()

    def test_completes_normally(self):
        self._run("job")
        self._finish()
        self.assertEqual(self._status("job"), {"status": "completed", "result": "result"})

    def test_cancelled_while_running_stays_cancelled(self):
        self._run("job")
        self.assertTrue(self.engine.started.wait(timeout=5))
        asyncio.run(self.engine.cancel_job("job"))
        self._finish()
        self.assertEqual(self._status("job"), {"status": "cancelled"})

    def test_failure_after_cancel_stays_cancelled(self):
        self.engine.fail = True
        self._run("job")
        self.assertTrue(self.engine.started.wait(timeout=5))
        asyncio.run(self.engine.cancel_job("job"))
        self._finish()
        self.assertEqual(self._status("job"), {"status": "cancelled"})

    def test_cancelled_while_queued_never_runs(self):
        self._run("first")
        self.assertTrue(self.engine.started.wait(timeout=5))
        self._run("second")
        asyncio.run(self.engine.cancel_job("second"))
        self.engine.started.clear()
        self._finish()
        self.assertFalse(self.engine.started.is_set())
        self.assertEqual(self._status("first")["status"], "completed")
        self.assertEqual(self._status("second"), {"status": "cancelled"})


if __name__ == "__main__":
    unittest.main()
