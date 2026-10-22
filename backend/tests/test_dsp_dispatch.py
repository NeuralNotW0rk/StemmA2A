import asyncio
import shutil
import sys
import tempfile
import threading
import unittest
from pathlib import Path

import torch

backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

import operations.audio  # noqa: F401  registers the DSP operations
from operations.dispatcher import dispatch_operation
from param_graph.elements.artifacts.audio_element import Audio
from param_graph.elements.base_elements import Asset
from param_graph.elements.collections.group_element import Group
from param_graph.graph import ParameterGraph
from utils.audio import save_audio

SAMPLE_RATE = 48000


class TestImmediateDspDispatch(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.pg = ParameterGraph(str(self.tmp / "project"))
        self.pg.save()

        source_path = self.tmp / "source.wav"
        save_audio(torch.rand(2, SAMPLE_RATE * 2) * 0.5 - 0.25, source_path, SAMPLE_RATE, format="wav")
        self.source = Audio(
            id="source_audio", name="source", context={},
            file=Asset(path=str(source_path), uid="source_audio", extension=".wav"),
            sample_rate=SAMPLE_RATE, duration=2.0,
        )
        self.pg.add_element(self.source)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _dispatch(self, operation, **params):
        payload = {"operation": operation, "params": {"source_audio": self.source.id, **params}}
        return asyncio.run(dispatch_operation(
            payload, self.pg, engine_provider=None, graph_lock=threading.Lock(),
            sample_rate=SAMPLE_RATE, data_cache_root=self.tmp / "cache",
        ))

    def test_single_result_adds_linked_artifact(self):
        response, status = self._dispatch("gain", gain_db=6.0)

        self.assertEqual(status, 200, response)
        artifact_id = response["node_id"]
        self.assertEqual(response["artifact"]["id"], artifact_id)
        self.assertIsInstance(self.pg.get_element(artifact_id), Audio)
        self.assertTrue(self.pg.G.has_edge(self.source.id, artifact_id))

    def test_batch_result_is_grouped(self):
        response, status = self._dispatch("slice", chunk_duration=0.5, overlap=0.0)

        self.assertEqual(status, 200, response)
        self.assertEqual(len(response["artifacts"]), 4)
        group = self.pg.get_element(response["node_id"])
        self.assertIsInstance(group, Group)
        self.assertCountEqual(group.member_ids, [a["id"] for a in response["artifacts"]])


if __name__ == "__main__":
    unittest.main()
