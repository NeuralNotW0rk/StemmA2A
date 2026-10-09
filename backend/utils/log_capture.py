"""Mirrors console output (prints, tracebacks and server logs) into a persistent, timestamped log file."""

import os
import sys
import threading
from datetime import datetime
from pathlib import Path

# Rotated at startup once the log grows past this size, keeping this many older files (<name>.log.1, ...)
MAX_LOG_BYTES = 10 * 1024 * 1024
BACKUP_COUNT = 3


class _TeeStream:
    """Passes writes through to the original console stream and appends them, timestamped per line, to a log file."""

    def __init__(self, stream, log_file, lock: threading.Lock) -> None:
        self._stream = stream
        self._log_file = log_file
        self._lock = lock
        self._at_line_start = True

    def write(self, text: str) -> int:
        if self._stream is not None:
            try:
                self._stream.write(text)
            except Exception:
                pass

        if text:
            with self._lock:
                prefix = f"{datetime.now():%Y-%m-%d %H:%M:%S.%f}"[:-3] + f" [{os.getpid()}] "
                lines = text.split("\n")
                out = []
                for i, line in enumerate(lines):
                    if i > 0:
                        out.append("\n")
                        self._at_line_start = True
                    if line:
                        if self._at_line_start:
                            out.append(prefix)
                        out.append(line)
                        self._at_line_start = False
                try:
                    self._log_file.write("".join(out))
                    self._log_file.flush()
                except Exception:
                    pass
        return len(text)

    def flush(self) -> None:
        if self._stream is not None:
            try:
                self._stream.flush()
            except Exception:
                pass

    def __getattr__(self, name):
        # isatty, fileno, encoding, etc. come from the original console stream
        return getattr(self._stream, name)


def _rotate(log_path: Path) -> None:
    """Shifts <name>.log to <name>.log.1 (and older files up by one) once it exceeds MAX_LOG_BYTES."""
    if not log_path.exists() or log_path.stat().st_size < MAX_LOG_BYTES:
        return
    for i in range(BACKUP_COUNT - 1, 0, -1):
        older = log_path.with_name(f"{log_path.name}.{i}")
        if older.exists():
            os.replace(older, log_path.with_name(f"{log_path.name}.{i + 1}"))
    os.replace(log_path, log_path.with_name(f"{log_path.name}.1"))


def start_log_capture(log_dir: Path, name: str) -> Path:
    """
    Tees sys.stdout and sys.stderr into <log_dir>/<name>.log for the rest of the process, so output that
    would otherwise only reach a console window (including tracebacks from background threads) is kept.
    Returns the log file path.
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{name}.log"

    # Under Flask's debug reloader, the watcher process starts first and the server runs in a child
    # (WERKZEUG_RUN_MAIN=true) that appends to the same file, so only the watcher rotates
    if os.environ.get("WERKZEUG_RUN_MAIN") != "true":
        try:
            _rotate(log_path)
        except OSError as e:
            print(f"Warning: Could not rotate log file {log_path}: {e}")

    log_file = open(log_path, "a", encoding="utf-8", errors="replace", buffering=1)
    lock = threading.Lock()
    sys.stdout = _TeeStream(sys.stdout, log_file, lock)
    sys.stderr = _TeeStream(sys.stderr, log_file, lock)
    print(f"===== {name} started (pid {os.getpid()}); logging to {log_path} =====")
    return log_path
