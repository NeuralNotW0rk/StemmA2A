import os
import threading
from pathlib import Path
from .engine import Engine
from .local_engine import LocalEngine
from .remote_engine import RemoteEngine

class EngineProvider:
    _engine_instance: Engine = None
    # Every engine created since the last reset, keyed by remote URL (None for the local engine).
    # Engines that are switched away from are kept, so switching back doesn't reload models or encoders.
    _engines: dict[str | None, Engine] = {}
    _switch_lock = threading.Lock()

    def __init__(self, remote_url: str = None, data_root: str = None):
        if EngineProvider._engine_instance is None:
            EngineProvider._engines = {}
            EngineProvider._engine_instance = self._get_or_create(remote_url, data_root)

    @staticmethod
    def _get_or_create(remote_url: str | None, data_root: str | None) -> Engine:
        engine = EngineProvider._engines.get(remote_url)
        if engine is None:
            if remote_url:
                timeout = int(os.environ.get("REMOTE_ENGINE_TIMEOUT", 300))
                engine = RemoteEngine(remote_url, timeout=timeout, data_root=data_root)
            else:
                engine = LocalEngine(data_root=data_root)
            EngineProvider._engines[remote_url] = engine
        return engine

    def get_engine(self) -> Engine:
        return EngineProvider._engine_instance

    @property
    def remote_url(self) -> str | None:
        """URL of the active remote engine, or None when the local engine is active."""
        return getattr(EngineProvider._engine_instance, "remote_url", None)

    def set_data_root(self, data_root: str) -> None:
        """
        Points every engine at a new project's data root. Engines are reused across projects rather than
        recreated, since each local engine holds a worker thread, encoders and cached models.
        """
        with EngineProvider._switch_lock:
            for engine in EngineProvider._engines.values():
                engine.data_root = Path(data_root)
                engine.data_root.mkdir(parents=True, exist_ok=True)

    def switch(self, remote_url: str | None) -> Engine:
        """Makes the engine for remote_url (None for local) active, creating it on first use."""
        with EngineProvider._switch_lock:
            if remote_url != self.remote_url:
                data_root = str(EngineProvider._engine_instance.data_root)
                EngineProvider._engine_instance = self._get_or_create(remote_url, data_root)
            return EngineProvider._engine_instance
