from .engine import LocalEngine, RemoteEngine
from param_graph.graph import ParameterGraph

class EngineProvider:
    def __init__(self, remote_url: str = None, graph: ParameterGraph = None):
        self.remote_url = remote_url
        self.graph = graph

    def get_engine(self):
        if self.remote_url:
            return RemoteEngine(self.remote_url, self.graph)
        else:
            return LocalEngine()
