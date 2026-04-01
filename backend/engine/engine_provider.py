import os
from .engine import Engine
from .local_engine import LocalEngine
from .remote_engine import RemoteEngine

class EngineProvider:
    _engine_instance: Engine = None

    def __init__(self, remote_url: str = None, data_root: str = None):
        if EngineProvider._engine_instance is None:
            if remote_url:
                timeout = int(os.environ.get("REMOTE_ENGINE_TIMEOUT", 300))
                EngineProvider._engine_instance = RemoteEngine(remote_url, timeout=timeout)
            else:
                EngineProvider._engine_instance = LocalEngine(data_root=data_root)
    
    def get_engine(self) -> Engine:
        return EngineProvider._engine_instance
