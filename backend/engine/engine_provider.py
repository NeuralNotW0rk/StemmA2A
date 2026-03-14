from .engine import Engine
from .local_engine import LocalEngine
from .remote_engine import RemoteEngine

class EngineProvider:
    _engine_instance: Engine = None

    def __init__(self, remote_url: str = None):
        if EngineProvider._engine_instance is None:
            if remote_url:
                EngineProvider._engine_instance = RemoteEngine(remote_url)
            else:
                EngineProvider._engine_instance = LocalEngine()
    
    def get_engine(self) -> Engine:
        return EngineProvider._engine_instance
