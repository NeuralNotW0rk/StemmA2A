"""
A drop folder for putting assets into the engine service's cache by hand, for when copying a file
onto the engine host is faster than uploading it.

Put a file or directory in <data>/inbox/ named exactly by its UID, as the client logs it (for
example f28c7dd48e02c8a6.xxh3_64). Once it is unchanged between two scans it is moved into
cache/xx/<uid> and marked as just used, so it then ages out under the normal policy (see
cache_policy.py). Names ending in .part are ignored: copy under <uid>.part and rename when the copy
is done if your copy tool sets the final size before it has written everything.

A directory is checked against its UID before it is moved. A file can't be, because its UID comes
from the model adapter rather than its bytes. A wrong or incomplete file is caught by /execute,
which compares it with the size the client recorded, evicts it and asks for an upload instead.
"""
import os
import re
import threading
import time
from pathlib import Path
from typing import Callable

from engine.cache_policy import size_and_mtime
from utils.filesystem import remove_path
from utils.uid import XXH3_64, path_from_uid

INBOX_DIR = "inbox"
UID_PATTERN = re.compile(r"^[0-9a-f]+\.[a-z0-9_]+$")


class Inbox:
    def __init__(self, data_root: Path, lock: threading.RLock, model_path: Callable[[str], Path]):
        self.data_root = Path(data_root)
        self.inbox_dir = self.data_root / INBOX_DIR
        # The cache policy's lock, so an entry can't appear between /execute's check and its lease
        self.lock = lock
        # Where the model store keeps a UID; assets already there aren't copied into the cache
        self._model_path = model_path
        # Size and newest mtime of each entry at the previous scan
        self._last_seen: dict[str, tuple[int, float]] = {}
        # Problems already reported, so each is logged once rather than every scan
        self._reported: set[str] = set()

    def scan(self) -> list[str]:
        """Moves every entry that hasn't changed since the previous scan into the cache. Returns their UIDs."""
        self.inbox_dir.mkdir(parents=True, exist_ok=True)
        seen: dict[str, tuple[int, float]] = {}
        ingested = []
        for path in sorted(self.inbox_dir.iterdir()):
            name = path.name
            if name.startswith(".") or name.endswith(".part"):
                continue
            if not UID_PATTERN.match(name):
                self._report(name, f"'{name}' isn't named by a UID (such as f28c7dd48e02c8a6.xxh3_64); leaving it")
                continue
            try:
                snapshot = size_and_mtime(path)
            except FileNotFoundError:
                continue
            seen[name] = snapshot
            if self._last_seen.get(name) != snapshot:
                continue  # New since the last scan, or still being copied
            try:
                if self._ingest(name, path):
                    ingested.append(name)
                    print(f"[Engine Inbox] Moved '{name}' into the cache.")
            except Exception as e:
                self._report(f"{name}:{snapshot}", f"Could not take in '{name}': {e}; leaving it")
        self._last_seen = seen
        return ingested

    def _ingest(self, uid: str, path: Path) -> bool:
        if path.is_dir():
            computed = XXH3_64().from_directory(path)
            if computed != uid:
                raise ValueError(f"its contents hash to '{computed}'")
        with self.lock:
            if self._model_path(uid).exists():
                self._report(uid, f"'{uid}' is already in the model store; it can be deleted from the inbox")
                return False
            destination = self.data_root / path_from_uid(uid)
            destination.parent.mkdir(parents=True, exist_ok=True)
            remove_path(destination)
            os.replace(path, destination)
            # Copies often keep their original modified time, which would make the entry look long unused
            os.utime(destination, None)
        return True

    def _report(self, key: str, message: str) -> None:
        if key not in self._reported:
            self._reported.add(key)
            print(f"[Engine Inbox] {message}")

    def start_background_scans(self, interval_seconds: float) -> threading.Thread:
        def _loop() -> None:
            while True:
                try:
                    self.scan()
                except Exception as e:
                    import traceback
                    print(f"[Engine Inbox] Scan failed: {e}")
                    traceback.print_exc()
                time.sleep(interval_seconds)

        thread = threading.Thread(target=_loop, name="engine-inbox-scanner", daemon=True)
        thread.start()
        return thread
