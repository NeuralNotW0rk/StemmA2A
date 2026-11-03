import sys
import unittest
from pathlib import Path
from unittest import mock

backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from huggingface_hub import constants
from huggingface_hub.errors import LocalEntryNotFoundError

from utils.hf_hub import hub_online, is_cache_miss, load_cached_first


def _transformers_style_miss():
    # transformers wraps the hub error in an OSError, as in AutoTokenizer.from_pretrained
    try:
        raise LocalEntryNotFoundError("not in cache")
    except LocalEntryNotFoundError as e:
        raise OSError("couldn't connect to 'https://huggingface.co'") from e


class TestHfHub(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch.object(constants, "HF_HUB_OFFLINE", True)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_cache_miss_is_detected_through_exception_chain(self):
        with self.assertRaises(OSError) as ctx:
            _transformers_style_miss()
        self.assertTrue(is_cache_miss(ctx.exception))
        self.assertFalse(is_cache_miss(OSError("disk full")))

    def test_cache_hit_stays_offline(self):
        calls = []

        def load(local_files_only):
            calls.append((local_files_only, constants.HF_HUB_OFFLINE))
            return "model"

        self.assertEqual(load_cached_first(load, "test"), "model")
        self.assertEqual(calls, [(True, True)])

    def test_cache_miss_retries_online_then_restores_offline(self):
        calls = []

        def load(local_files_only):
            calls.append((local_files_only, constants.HF_HUB_OFFLINE))
            if constants.HF_HUB_OFFLINE:
                _transformers_style_miss()
            return "model"

        self.assertEqual(load_cached_first(load, "test"), "model")
        self.assertEqual(calls, [(True, True), (False, False)])
        self.assertIs(constants.HF_HUB_OFFLINE, True)

    def test_other_errors_are_not_retried(self):
        calls = []

        def load(local_files_only):
            calls.append(local_files_only)
            raise ValueError("bad config")

        with self.assertRaises(ValueError):
            load_cached_first(load, "test")
        self.assertEqual(calls, [True])

    def test_hub_online_restores_flag_on_error(self):
        with self.assertRaises(RuntimeError):
            with hub_online():
                self.assertIs(constants.HF_HUB_OFFLINE, False)
                raise RuntimeError
        self.assertIs(constants.HF_HUB_OFFLINE, True)


if __name__ == "__main__":
    unittest.main()
