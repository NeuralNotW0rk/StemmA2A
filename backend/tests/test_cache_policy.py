import os
import sys
import time
import shutil
import tempfile
import unittest
from pathlib import Path

backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from engine.cache_policy import CachePolicy, MIB, size_weight
from utils.uid import path_from_uid

HOUR = 3600
DAY = 24 * HOUR


class TestCachePolicy(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.now = time.time()
        self.active_jobs: set[str] = set()

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def _policy(self, max_bytes: int, min_age: float = DAY) -> CachePolicy:
        return CachePolicy(
            data_root=self.root,
            max_bytes=max_bytes,
            min_age_seconds=min_age,
            is_job_active=lambda job_id: job_id in self.active_jobs,
        )

    def _entry(self, uid: str, size: int, idle: float) -> Path:
        path = self.root / path_from_uid(uid)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            f.truncate(size)  # Sparse, so large entries cost no real disk
        os.utime(path, (self.now - idle, self.now - idle))
        return path

    def _evicted(self, report: dict) -> list[str]:
        return [e["uid"] for e in report["evicted"]]

    def test_size_weight_is_logarithmic(self):
        self.assertEqual(size_weight(10), 1.0)
        self.assertEqual(size_weight(MIB), 1.0)
        self.assertAlmostEqual(size_weight(64 * MIB), 7.0)
        self.assertAlmostEqual(size_weight(1024 * MIB), 11.0)

    def test_under_cap_evicts_nothing(self):
        self._entry("a.xxh3_64", MIB, idle=10 * DAY)
        report = self._policy(max_bytes=10 * MIB).sweep()
        self.assertEqual(report["evicted"], [])

    def test_large_files_outlast_small_ones_by_a_bounded_factor(self):
        # 64 MiB (weight 7) idle 5 days scores 5/7; 1 MiB idle 2 days scores 2: the small file goes first
        big = self._entry("big.xxh3_64", 64 * MIB, idle=5 * DAY)
        self._entry("small.xxh3_64", MIB, idle=2 * DAY)
        report = self._policy(max_bytes=64 * MIB).sweep(dry_run=True)
        self.assertEqual(self._evicted(report), ["small.xxh3_64", "big.xxh3_64"])

        # Idle 30 days, the big file scores ~4.3 and goes before a small file idle 2 days
        os.utime(big, (self.now - 30 * DAY, self.now - 30 * DAY))
        report = self._policy(max_bytes=64 * MIB).sweep(dry_run=True)
        self.assertEqual(self._evicted(report), ["big.xxh3_64"])

    def test_stops_at_low_water_mark(self):
        for i in range(10):
            self._entry(f"e{i}.xxh3_64", MIB, idle=(2 + i) * DAY)
        report = self._policy(max_bytes=9 * MIB).sweep()
        # 10 MiB down to at most 0.8 * 9 = 7.2 MiB: the three idlest entries go
        self.assertEqual(self._evicted(report), ["e9.xxh3_64", "e8.xxh3_64", "e7.xxh3_64"])

    def test_recent_and_leased_entries_are_kept(self):
        self._entry("recent.xxh3_64", MIB, idle=HOUR)
        self._entry("leased.xxh3_64", MIB, idle=10 * DAY)
        self._entry("old.xxh3_64", MIB, idle=5 * DAY)
        self.active_jobs.add("job1")
        policy = self._policy(max_bytes=MIB)
        policy.lease("job1", ["leased.xxh3_64"])
        report = policy.sweep()
        self.assertEqual(self._evicted(report), ["old.xxh3_64"])
        self.assertIn("warning", report)

        # Once the job finishes its lease lapses
        self.active_jobs.clear()
        report = policy.sweep()
        self.assertEqual(self._evicted(report), ["leased.xxh3_64"])

    def test_stale_upload_parts_are_removed(self):
        stale = self.root / "cache" / "ab" / "stale.xxh3_64.part"
        fresh = self.root / "cache" / "ab" / "fresh.xxh3_64.part"
        stale.parent.mkdir(parents=True)
        stale.write_bytes(b"x")
        fresh.write_bytes(b"x")
        os.utime(stale, (self.now - 2 * DAY, self.now - 2 * DAY))
        report = self._policy(max_bytes=0).sweep()
        self.assertEqual(report["stale_parts_removed"], 1)
        self.assertFalse(stale.exists())
        self.assertTrue(fresh.exists())

    def test_dry_run_deletes_nothing(self):
        path = self._entry("a.xxh3_64", 2 * MIB, idle=10 * DAY)
        report = self._policy(max_bytes=MIB).sweep(dry_run=True)
        self.assertEqual(self._evicted(report), ["a.xxh3_64"])
        self.assertTrue(path.exists())


if __name__ == "__main__":
    unittest.main()
