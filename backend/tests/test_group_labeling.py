import unittest
import tempfile
import shutil
from pathlib import Path

from param_graph.graph import ParameterGraph
from param_graph.elements.artifacts.audio_element import Audio
from param_graph.elements.collections.group_element import Group
from param_graph.elements.base_elements import Asset
import app
from app import update_group_labels


class TestGroupDynamicLabeling(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.graph = ParameterGraph(self.test_dir)
        app.param_graph = self.graph

    def tearDown(self):
        app.param_graph = None
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_replicated_batch_varying_seed_only(self):
        """Verify that a sequence/batch varying only by seed only displays seed in member aliases."""
        ctx1 = {
            'prompt': 'techno heavy kick',
            'steps': 50,
            'cfg_scale': 7.0,
            'seed': 1001,
            'model_id': 'stable_audio_1',
            'init_audio': 'audio_parent_123',
            'noise_level': 0.7,
            'duration_padding_sec': 6.0,
        }
        ctx2 = {
            'prompt': 'techno heavy kick',
            'steps': 50,
            'cfg_scale': 7.0,
            'seed': 1002,
            'model_id': 'stable_audio_1',
            'init_audio': 'audio_parent_123',
            'noise_level': 0.7,
            'duration_padding_sec': 6.0,
        }
        ctx3 = {
            'prompt': 'techno heavy kick',
            'steps': 50,
            'cfg_scale': 7.0,
            'seed': 1003,
            'model_id': 'stable_audio_1',
            'init_audio': 'audio_parent_123',
            'noise_level': 0.7,
            'duration_padding_sec': 6.0,
        }
        ctx4 = {
            'prompt': 'techno heavy kick',
            'steps': 50,
            'cfg_scale': 7.0,
            'seed': 1004,
            'model_id': 'stable_audio_1',
            'init_audio': 'audio_parent_123',
            'noise_level': 0.7,
            'duration_padding_sec': 6.0,
        }

        m1 = Audio(id="m1", name="m1", file=Asset(path="1.wav", uid="m1"), context=ctx1)
        m2 = Audio(id="m2", name="m2", file=Asset(path="2.wav", uid="m2"), context=ctx2)
        m3 = Audio(id="m3", name="m3", file=Asset(path="3.wav", uid="m3"), context=ctx3)
        m4 = Audio(id="m4", name="m4", file=Asset(path="4.wav", uid="m4"), context=ctx4)
        grp = Group(id="grp_batch", member_ids=["m1", "m2", "m3", "m4"], member_type="audio")

        self.graph.add_element(m1)
        self.graph.add_element(m2)
        self.graph.add_element(m3)
        self.graph.add_element(m4)
        self.graph.add_element(grp)

        update_group_labels("grp_batch")

        self.assertEqual(self.graph.G.nodes["m1"].get("alias"), "seed: 1001")
        self.assertEqual(self.graph.G.nodes["m2"].get("alias"), "seed: 1002")
        self.assertEqual(self.graph.G.nodes["m3"].get("alias"), "seed: 1003")
        self.assertEqual(self.graph.G.nodes["m4"].get("alias"), "seed: 1004")

        grp_node = self.graph.G.nodes["grp_batch"]
        self.assertEqual(grp_node.get("alias"), "techno heavy kick")
        shared = grp_node.get("shared_context")
        self.assertEqual(shared.get("prompt"), "techno heavy kick")
        self.assertEqual(shared.get("steps"), 50)
        self.assertEqual(shared.get("cfg_scale"), 7.0)
        self.assertEqual(shared.get("init_audio"), "audio_parent_123")
        self.assertNotIn("seed", shared)

    def test_group_with_empty_or_partially_configured_member(self):
        """Verify that an empty/unconfigured context in one member does not pollute constant attributes onto other members."""
        ctx1 = {
            'prompt': 'ambient pad',
            'steps': 40,
            'cfg_scale': 6.5,
            'seed': 201,
            'model_id': 'model_pad'
        }
        ctx2 = {
            'prompt': 'ambient pad',
            'steps': 40,
            'cfg_scale': 6.5,
            'seed': 202,
            'model_id': 'model_pad'
        }
        # m3 has empty context (e.g. unexpressed or imported audio)
        ctx3 = {}

        m1 = Audio(id="a1", name="a1", file=Asset(path="1.wav", uid="a1"), context=ctx1)
        m2 = Audio(id="a2", name="a2", file=Asset(path="2.wav", uid="a2"), context=ctx2)
        m3 = Audio(id="a3", name="a3", file=Asset(path="3.wav", uid="a3"), context=ctx3)
        grp = Group(id="grp_mixed", member_ids=["a1", "a2", "a3"], member_type="audio")

        self.graph.add_element(m1)
        self.graph.add_element(m2)
        self.graph.add_element(m3)
        self.graph.add_element(grp)

        update_group_labels("grp_mixed")

        self.assertEqual(self.graph.G.nodes["a1"].get("alias"), "seed: 201")
        self.assertEqual(self.graph.G.nodes["a2"].get("alias"), "seed: 202")
        self.assertIsNone(self.graph.G.nodes["a3"].get("alias"))

        grp_node = self.graph.G.nodes["grp_mixed"]
        self.assertEqual(grp_node.get("alias"), "ambient pad")

    def test_identical_members_produce_none_alias(self):
        """Verify that when members have identical contexts, no diffs are shown."""
        ctx = {'prompt': 'identical sound', 'steps': 30, 'cfg_scale': 5.0}
        m1 = Audio(id="i1", name="i1", file=Asset(path="1.wav", uid="i1"), context=dict(ctx))
        m2 = Audio(id="i2", name="i2", file=Asset(path="2.wav", uid="i2"), context=dict(ctx))
        grp = Group(id="grp_ident", member_ids=["i1", "i2"], member_type="audio")

        self.graph.add_element(m1)
        self.graph.add_element(m2)
        self.graph.add_element(grp)

        update_group_labels("grp_ident")

        self.assertIsNone(self.graph.G.nodes["i1"].get("alias"))
        self.assertIsNone(self.graph.G.nodes["i2"].get("alias"))
        self.assertEqual(self.graph.G.nodes["grp_ident"].get("alias"), "identical sound")

    def test_blacklisted_keys_omitted_from_labels(self):
        """Verify that blacklisted internal metadata (job_id, operation, group_id, etc.) never appear on aliases."""
        ctx1 = {'prompt': 'test', 'seed': 1, 'job_id': 'job_123', 'operation': 'generate', 'group_id': 'grp_blk'}
        ctx2 = {'prompt': 'test', 'seed': 2, 'job_id': 'job_456', 'operation': 'generate', 'group_id': 'grp_blk'}

        m1 = Audio(id="b1", name="b1", file=Asset(path="1.wav", uid="b1"), context=ctx1)
        m2 = Audio(id="b2", name="b2", file=Asset(path="2.wav", uid="b2"), context=ctx2)
        grp = Group(id="grp_blk", member_ids=["b1", "b2"], member_type="audio")

        self.graph.add_element(m1)
        self.graph.add_element(m2)
        self.graph.add_element(grp)

        update_group_labels("grp_blk")

        self.assertEqual(self.graph.G.nodes["b1"].get("alias"), "seed: 1")
        self.assertEqual(self.graph.G.nodes["b2"].get("alias"), "seed: 2")

    def test_simplify_dynamic_label(self):
        """Verify dynamic label simplification into clean, safe filename stems with ultra-compact abbreviations."""
        from app import simplify_dynamic_label

        self.assertEqual(simplify_dynamic_label("seed: 1001"), "seed_1001")
        self.assertEqual(
            simplify_dynamic_label("seed: 1001\ncfg_scale: 7.0"),
            "seed_1001_cfg_scale_7.0"
        )
        self.assertEqual(
            simplify_dynamic_label("cluster: 1 (conv1.conv)"),
            "c1"
        )
        self.assertEqual(
            simplify_dynamic_label(
                "cluster: 1 (convs.2.conv)\nfactor: 1.5 (convs.2.conv)\ncluster: 3 (convs.4.conv)\nfactor: -0.5 (convs.4.conv)"
            ),
            "c1_f1.5_c3_f-0.5"
        )
        self.assertEqual(
            simplify_dynamic_label("prompt: ambient pad..."),
            "prompt_ambient_pad"
        )
        self.assertEqual(
            simplify_dynamic_label("strength: 0.8 (layer4)"),
            "s0.8"
        )
        self.assertEqual(
            simplify_dynamic_label("bias: 0.25 (layer4)"),
            "b0.25"
        )
        self.assertEqual(simplify_dynamic_label(None), "")
        self.assertEqual(simplify_dynamic_label(""), "")
        self.assertEqual(simplify_dynamic_label(":::"), "")

    def test_long_label_untruncated_full_alias(self):
        """Verify that UI alias is truncated with ellipsis while full_alias contains all info."""
        ctx1 = {
            'prompt': 'techno heavy kick',
            'steps': 50,
            'cfg_scale': 7.5,
            'seed': 1001,
            'noise_level': 0.65,
            'duration_padding_sec': 6.0,
        }
        ctx2 = {
            'prompt': 'techno heavy kick',
            'steps': 80,
            'cfg_scale': 9.0,
            'seed': 2002,
            'noise_level': 0.85,
            'duration_padding_sec': 8.0,
        }

        m1 = Audio(id="long1", name="long1", file=Asset(path="1.wav", uid="long1"), context=ctx1)
        m2 = Audio(id="long2", name="long2", file=Asset(path="2.wav", uid="long2"), context=ctx2)
        grp = Group(id="grp_long", member_ids=["long1", "long2"], member_type="audio")

        self.graph.add_element(m1)
        self.graph.add_element(m2)
        self.graph.add_element(grp)

        update_group_labels("grp_long")

        ui_alias = self.graph.G.nodes["long1"].get("alias")
        full_alias = self.graph.G.nodes["long1"].get("full_alias")

        self.assertTrue(ui_alias.endswith("..."))
        self.assertLessEqual(len(ui_alias), 40)
        self.assertFalse(full_alias.endswith("..."))
        self.assertIn("steps: 50", full_alias)
        self.assertIn("cfg_scale: 7.5", full_alias)
        self.assertIn("seed: 1001", full_alias)
        self.assertIn("noise_level: 0.65", full_alias)
        self.assertIn("duration_padding_sec: 6.0", full_alias)


if __name__ == "__main__":
    unittest.main()

