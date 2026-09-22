import unittest
from operations.registry import SyncRegistry
from operations.audio.gain import GainOperation
from operations.audio.normalize import NormalizeOperation
from operations.audio.slice import SliceOperation
from operations.audio.onset_slice import LibrosaOnsetSliceOperation
from evolution.operations import get_evolution_operations

class TestOperationsConfig(unittest.TestCase):
    def setUp(self) -> None:
        self.registry = SyncRegistry()

    def test_sync_operations_load_json_form_configs(self) -> None:
        """Verify that audio sync operations auto-load form configs from JSON files."""
        gain_op = GainOperation()
        gain_form = gain_op.get_form_config()
        self.assertIsInstance(gain_form, list)
        self.assertTrue(len(gain_form) > 0)
        param_names = [field["name"] for field in gain_form]
        self.assertIn("source_audio", param_names)
        self.assertIn("gain_db", param_names)

        norm_op = NormalizeOperation()
        norm_form = norm_op.get_form_config()
        self.assertIsInstance(norm_form, list)
        self.assertTrue(len(norm_form) > 0)
        param_names = [field["name"] for field in norm_form]
        self.assertIn("source_audio", param_names)
        self.assertIn("target_peak", param_names)

        slice_op = SliceOperation()
        slice_form = slice_op.get_form_config()
        self.assertIsInstance(slice_form, list)
        self.assertTrue(len(slice_form) > 0)
        param_names = [field["name"] for field in slice_form]
        self.assertIn("chunk_duration", param_names)
        self.assertIn("overlap", param_names)

        onset_op = LibrosaOnsetSliceOperation()
        onset_form = onset_op.get_form_config()
        self.assertIsInstance(onset_form, list)
        self.assertTrue(len(onset_form) > 0)
        param_names = [field["name"] for field in onset_form]
        self.assertIn("backtrack", param_names)

    def test_evolution_operations_load_json(self) -> None:
        """Verify that evolution operations load properly from operations.json."""
        evo_ops = get_evolution_operations()
        self.assertIsInstance(evo_ops, list)
        self.assertEqual(len(evo_ops), 3)

        op_names = [op["name"] for op in evo_ops]
        self.assertIn("wrap_individual", op_names)
        self.assertIn("mutate", op_names)
        self.assertIn("recombine", op_names)

        for op in evo_ops:
            self.assertIn("name", op)
            self.assertIn("description", op)
            self.assertIn("category", op)
            self.assertIn("execution", op)
            self.assertIn("form_config", op)
            self.assertIsInstance(op["form_config"], list)
            self.assertTrue(len(op["form_config"]) > 0)

if __name__ == "__main__":
    unittest.main()
