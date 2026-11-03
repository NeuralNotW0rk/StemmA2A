"""
Size-capped eviction for the engine service's content-addressed cache (<data>/cache/xx/<uid>).

The policy only looks at each entry's size and when it was last used, never at what it holds:

- An entry is whatever sits at cache/xx/<uid>: a file, or a directory asset as a whole, whose size
  is the total of its files. Its members are never entries of their own.
- Last use is the entry's mtime, which is bumped on every cache hit (see touch()).
- Once the cache grows past max_bytes, entries are evicted until it is back under
  LOW_WATER_RATIO * max_bytes. The entry with the highest idle_time / size_weight goes first, where
  the weight grows with log2 of the size, so large files outlast small ones by a bounded factor.
- Entries used within min_age are never evicted, and neither are leased ones. A lease is held by a
  job that is still pending or running.

Evicting an entry is always recoverable: /execute reports it missing and the client uploads it again.
Shared models live outside the cache, in the model store (see model_store.py), so the cache has no
entries it must keep for their own sake.
"""
import math
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

from utils.filesystem import remove_path

MIB = 1024 * 1024
GIB = 1024 * MIB
LOW_WATER_RATIO = 0.8
# Upload staging files older than this are left over from failed uploads
STALE_PART_SECONDS = 24 * 3600


def size_weight(size: int) -> float:
    """1 for files up to 1 MiB, then +1 for every doubling (100 MiB ~7.6, 5 GiB ~13.3)."""
    return 1.0 + math.log2(max(size, MIB) / MIB)


def touch(path: Path) -> None:
    """Records a cache hit by bumping the entry's mtime."""
    try:
        os.utime(path, None)
    except FileNotFoundError:
        pass


def size_and_mtime(path: Path) -> tuple[int, float]:
    """A file's size and mtime, or a directory's total size and the newest mtime inside it."""
    stat = path.stat()
    if not path.is_dir():
        return stat.st_size, stat.st_mtime
    size, mtime = 0, stat.st_mtime
    for member in path.glob("**/*"):
        member_stat = member.stat()
        mtime = max(mtime, member_stat.st_mtime)
        if member.is_file():
            size += member_stat.st_size
    return size, mtime




@dataclass
class CacheEntry:
    path: Path
    size: int
    last_used: float

    @property
    def uid(self) -> str:
        return self.path.name

    def score(self, now: float) -> float:
        return (now - self.last_used) / size_weight(self.size)


class CachePolicy:
    def __init__(
        self,
        data_root: Path,
        max_bytes: int,
        min_age_seconds: float,
        is_job_active: Callable[[str], bool],
    ):
        self.data_root = Path(data_root)
        self.cache_dir = self.data_root / "cache"
        self.max_bytes = max_bytes
        self.min_age_seconds = min_age_seconds
        self._is_job_active = is_job_active
        # Held while checking for assets and taking leases in /execute, and while sweeping, so an
        # entry can't be evicted between being found and being leased
        self.lock = threading.RLock()
        self._leases: dict[str, set[str]] = {}

    def lease(self, job_id: str, uids: Iterable[str]) -> None:
        """Protects the job's inputs until the job is no longer pending or running."""
        with self.lock:
            self._leases[job_id] = set(uids)

    def _leased_uids(self) -> set[str]:
        for job_id in [j for j in self._leases if not self._is_job_active(j)]:
            del self._leases[job_id]
        leased: set[str] = set()
        for uids in self._leases.values():
            leased |= uids
        return leased

    def _scan(self, now: float) -> tuple[list[CacheEntry], list[Path]]:
        """Returns the cache entries and the stale upload staging files and directories."""
        entries, stale_parts = [], []
        if not self.cache_dir.is_dir():
            return entries, stale_parts
        for shard in self.cache_dir.iterdir():
            if not shard.is_dir():
                continue
            for path in shard.iterdir():
                try:
                    size, mtime = size_and_mtime(path)
                except FileNotFoundError:
                    continue
                if path.name.endswith(".part"):
                    if now - mtime > STALE_PART_SECONDS:
                        stale_parts.append(path)
                    continue
                entries.append(CacheEntry(path, size, path.stat().st_mtime))
        return entries, stale_parts

    def sweep(self, dry_run: bool = False) -> dict:
        """Evicts entries until the cache is under its low-water mark. With dry_run, only reports what would go."""
        with self.lock:
            now = time.time()
            entries, stale_parts = self._scan(now)
            total = sum(e.size for e in entries)
            report = {
                "total_bytes": total,
                "entry_count": len(entries),
                "max_bytes": self.max_bytes,
                "evicted": [],
                "evicted_bytes": 0,
                "stale_parts_removed": 0,
                "dry_run": dry_run,
            }

            if not dry_run:
                for path in stale_parts:
                    remove_path(path)
                report["stale_parts_removed"] = len(stale_parts)

            if self.max_bytes <= 0 or total <= self.max_bytes:
                return report

            leased = self._leased_uids()
            candidates = sorted(
                (e for e in entries if e.uid not in leased and now - e.last_used >= self.min_age_seconds),
                key=lambda e: e.score(now),
                reverse=True,
            )

            target = self.max_bytes * LOW_WATER_RATIO
            remaining = total
            for entry in candidates:
                if remaining <= target:
                    break
                if not dry_run:
                    try:
                        remove_path(entry.path)
                    except FileNotFoundError:
                        pass
                    except PermissionError:
                        # Open elsewhere (e.g. being served on Windows); try again next sweep. A
                        # directory left partly removed fails its size check in /execute and is
                        # uploaded again
                        continue
                remaining -= entry.size
                report["evicted"].append({"uid": entry.uid, "size": entry.size, "idle_hours": round((now - entry.last_used) / 3600, 1)})
                report["evicted_bytes"] += entry.size

            if remaining > target:
                report["warning"] = (
                    f"Cache is still {remaining / GIB:.1f} GiB after eviction; the rest is leased or "
                    f"used within the last {self.min_age_seconds / 3600:.0f} h."
                )
            return report

    def start_background_sweeps(self, interval_seconds: float) -> threading.Thread:
        def _loop() -> None:
            while True:
                try:
                    report = self.sweep()
                    if report["evicted"] or report["stale_parts_removed"] or report.get("warning"):
                        print(
                            f"[Engine Cache] Evicted {len(report['evicted'])} entries "
                            f"({report['evicted_bytes'] / GIB:.2f} GiB), removed "
                            f"{report['stale_parts_removed']} stale upload(s). "
                            f"Cache: {(report['total_bytes'] - report['evicted_bytes']) / GIB:.2f} / "
                            f"{report['max_bytes'] / GIB:.0f} GiB."
                        )
                    if report.get("warning"):
                        print(f"[Engine Cache] Warning: {report['warning']}")
                except Exception as e:
                    import traceback
                    print(f"[Engine Cache] Sweep failed: {e}")
                    traceback.print_exc()
                time.sleep(interval_seconds)

        thread = threading.Thread(target=_loop, name="engine-cache-sweeper", daemon=True)
        thread.start()
        return thread
