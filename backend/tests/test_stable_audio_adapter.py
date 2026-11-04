import json
import os
import struct
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add backend directory to sys.path
backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from engine.model_adapters.stable_audio_adapter import StableAudioAdapter
from engine.local_engine import LocalEngine
from param_graph.elements.base_elements import Asset
from utils.uid import XXH3_64, UIDMismatchError


class TestStableAudioAdapter(unittest.TestCase):
    def setUp(self):
        self.adapter = StableAudioAdapter()
        self.uid_gen = XXH3_64()

    def _create_dummy_safetensors(self, path: Path):
        header_json = b'{"__metadata__": {}}'
        header_len = struct.pack("<Q", len(header_json))
        with open(path, "wb") as f:
            f.write(header_len + header_json + b'{}')

    def test_register_model_with_directory_encoder(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            ckpt_path = tmp_path / "model.safetensors"
            self._create_dummy_safetensors(ckpt_path)

            config_path = tmp_path / "config.json"
            config_data = {
                "sample_rate": 44100,
                "sample_size": 2097152,
                "model": {
                    "conditioning": {
                        "configs": [{"id": "prompt", "type": "t5gemma"}]
                    }
                }
            }
            with open(config_path, "w") as f:
                json.dump(config_data, f)

            enc_dir = tmp_path / "encoder_dir"
            enc_dir.mkdir(parents=True, exist_ok=True)
            with open(enc_dir / "tokenizer.json", "w") as f:
                f.write('{"model": "t5"}')

            with patch("engine.model_adapters.stable_audio_adapter.load_ckpt_state_dict", return_value={}):
                model = self.adapter.register_model(
                    name="Test Stable Audio 3",
                    checkpoint_path=str(ckpt_path),
                    config_path=str(config_path),
                    encoder_path=str(enc_dir)
                )

            self.assertIsNotNone(model.encoder)
            self.assertEqual(model.encoder.uid, self.uid_gen.from_directory(enc_dir))
            expected_id = self.uid_gen.from_uids([model.checkpoint.uid, model.encoder.uid])
            self.assertEqual(model.id, expected_id)

    def test_register_model_with_archive_encoder(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            ckpt_path = tmp_path / "model.safetensors"
            self._create_dummy_safetensors(ckpt_path)

            config_path = tmp_path / "config.json"
            with open(config_path, "w") as f:
                json.dump({"sample_rate": 44100, "sample_size": 1000}, f)

            enc_archive = tmp_path / "encoder.zip"
            with open(enc_archive, "wb") as f:
                f.write(b"PK\x05\x06" + b"\x00" * 18)  # Empty zip archive bytes

            with patch("engine.model_adapters.stable_audio_adapter.load_ckpt_state_dict", return_value={}):
                model = self.adapter.register_model(
                    name="Test Archive Encoder",
                    checkpoint_path=str(ckpt_path),
                    config_path=str(config_path),
                    encoder_path=str(enc_archive),
                    encoder_uid="custom_encoder_uid.xxh3_64"
                )

            self.assertIsNotNone(model.encoder)
            self.assertEqual(model.encoder.uid, "custom_encoder_uid.xxh3_64")
            expected_id = self.uid_gen.from_uids([model.checkpoint.uid, "custom_encoder_uid.xxh3_64"])
            self.assertEqual(model.id, expected_id)

    def test_register_model_with_encoder_uid_only(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            ckpt_path = tmp_path / "model.safetensors"
            self._create_dummy_safetensors(ckpt_path)

            config_path = tmp_path / "config.json"
            with open(config_path, "w") as f:
                json.dump({"sample_rate": 44100, "sample_size": 1000}, f)

            with patch("engine.model_adapters.stable_audio_adapter.load_ckpt_state_dict", return_value={}):
                model = self.adapter.register_model(
                    name="Test UID Only",
                    checkpoint_path=str(ckpt_path),
                    config_path=str(config_path),
                    encoder_uid="remote_enc_123.xxh3_64",
                    encoder_size=5000
                )

            self.assertIsNotNone(model.encoder)
            self.assertEqual(model.encoder.uid, "remote_enc_123.xxh3_64")
            self.assertEqual(model.encoder.size, 5000)
            expected_id = self.uid_gen.from_uids([model.checkpoint.uid, "remote_enc_123.xxh3_64"])
            self.assertEqual(model.id, expected_id)

    def test_resolve_model_element_matches_by_checkpoint_and_encoder(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = LocalEngine(data_root=tmpdir)
            dummy_shared = MagicMock()
            dummy_shared.id = "shared_model_combined_id"
            dummy_shared.checkpoint = Asset(path="/models/ckpt.safetensors", uid="ckpt_uid_123.xxh3_64", size=100)
            dummy_shared.encoder = Asset(path="/models/enc", uid="enc_uid_456.xxh3_64", size=200)

            engine.shared_models = [dummy_shared]

            # Case A: Exact ID match
            query_exact = MagicMock()
            query_exact.id = "shared_model_combined_id"
            self.assertEqual(engine._resolve_model_element(query_exact), dummy_shared)

            # Case B: Diverged model ID from client project, but matching checkpoint + encoder UIDs
            query_diverged = MagicMock()
            query_diverged.id = "client_project_diff_id"
            query_diverged.checkpoint = Asset(path="", uid="ckpt_uid_123.xxh3_64", size=100)
            query_diverged.encoder = Asset(path="", uid="enc_uid_456.xxh3_64", size=200)

            resolved = engine._resolve_model_element(query_diverged)
            self.assertEqual(resolved, dummy_shared)



class TestStableAudioWeightLoading(unittest.TestCase):
    # Small enough to build in well under a second, with no Hub-backed conditioners
    CONFIG = {
        "model_type": "diffusion_cond",
        "sample_rate": 44100,
        "sample_size": 4096,
        "model": {
            "io_channels": 4,
            "diffusion": {
                "type": "dit",
                "cross_attention_cond_ids": ["seconds_start"],
                "global_cond_ids": ["seconds_total"],
                "config": {
                    "io_channels": 4, "embed_dim": 32, "depth": 2, "num_heads": 2,
                    "cond_token_dim": 16, "global_cond_dim": 16,
                    "transformer_type": "continuous_transformer",
                },
            },
            "conditioning": {
                "cond_dim": 16,
                "configs": [
                    {"id": "seconds_start", "type": "number", "config": {"min_val": 0, "max_val": 512}},
                    {"id": "seconds_total", "type": "number", "config": {"min_val": 0, "max_val": 512}},
                ],
            },
        },
    }

    def test_load_model_matches_a_copying_load(self) -> None:
        import torch
        from safetensors.torch import save_file
        from stable_audio_tools import create_model_from_config

        torch.manual_seed(0)
        reference = create_model_from_config(self.CONFIG)
        # Half precision on disk, as released checkpoints often are; the model must stay fp32
        weights = {k: v.clone().half().contiguous() for k, v in reference.state_dict().items()}
        expected = create_model_from_config(self.CONFIG)
        expected.load_state_dict(weights)

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "model.safetensors"
            save_file(weights, str(ckpt_path))
            adapter = StableAudioAdapter()
            info = adapter.register_model(config=self.CONFIG, checkpoint_path=str(ckpt_path), name="tiny")
            adapter.load_model(info)

        loaded = adapter.model.state_dict()
        self.assertEqual(list(loaded), list(expected.state_dict()))
        for key, tensor in expected.state_dict().items():
            self.assertEqual(loaded[key].dtype, tensor.dtype, key)
            self.assertTrue(torch.equal(loaded[key], tensor), key)
        for (name, ours), (_, theirs) in zip(adapter.model.named_parameters(), expected.named_parameters()):
            self.assertEqual(ours.requires_grad, theirs.requires_grad, name)

    def test_weight_init_is_restored_even_when_building_fails(self) -> None:
        import torch
        from engine.model_adapters.stable_audio_adapter import _skip_weight_init

        original = torch.nn.init.normal_
        with self.assertRaises(RuntimeError):
            with _skip_weight_init():
                self.assertIsNot(torch.nn.init.normal_, original)
                raise RuntimeError("build failed")
        self.assertIs(torch.nn.init.normal_, original)


class TestStableAudioLatentGeneration(unittest.TestCase):
    """
    Generating from an inverted latent hooks stable-audio-tools' sample_diffusion. Since 0.0.20
    every objective is routed through it, so the RF-only overrides must not touch v models.
    """

    def _generate_from_latent(
        self, diffusion_objective: str, inversion_meta: dict, latent: "torch.Tensor | None" = None,
        sampler_type: str = "dpmpp-2m", route_through_sample_k: bool = False,
    ) -> dict:
        import torch
        from types import SimpleNamespace
        import stable_audio_tools.inference.generation as sat_gen
        import stable_audio_tools.inference.sampling as sat_samp

        adapter = StableAudioAdapter()
        adapter.device = "cpu"
        model = MagicMock()
        model.to.return_value = model
        model.conditioner.return_value = {}
        model.mask_padding_attention = False
        model.use_effective_length_for_schedule = False
        model.pretransform.downsampling_ratio = 2048
        adapter.model = model
        adapter.model_info = SimpleNamespace(config={
            "sample_rate": 44100,
            "sample_size": 2048 * 8,
            "model": {"diffusion": {"diffusion_objective": diffusion_objective}},
        })

        if latent is None:
            latent = torch.randn(1, 64, 8)
        recorded: dict = {}

        def fake_original_sample_diffusion(*args: object, **kwargs: object) -> torch.Tensor:
            recorded.update(kwargs)
            if route_through_sample_k:
                # What 0.0.20's sample_diffusion does for v + k-diffusion samplers
                sat_samp.sample_k(
                    kwargs["model"], kwargs["noise"], init_data=None, steps=kwargs["steps"],
                    sampler_type=kwargs["sampler_type"], sigma_min=kwargs["sigma_min"],
                    sigma_max=kwargs["sigma_max"], rho=1.0, device="cpu", callback=None, cfg_scale=1.0,
                )
            return torch.randn(1, 2, 2048 * 8)

        def fake_k_sampler(denoiser: object, x: torch.Tensor, sigmas: torch.Tensor, **kwargs: object) -> torch.Tensor:
            recorded["k_x"] = x
            recorded["k_sigmas"] = sigmas
            recorded["k_extra_args"] = kwargs["extra_args"]
            return x

        def fake_generate_diffusion_cond(model_arg: object, **kwargs: object) -> torch.Tensor:
            # Mirrors 0.0.20: sigma_min/sigma_max travel to sample_diffusion as sampler kwargs
            return sat_gen.sample_diffusion(
                model=model_arg, noise=torch.randn_like(latent), cond_inputs={},
                diffusion_objective=diffusion_objective, steps=kwargs["steps"],
                sigma_min=kwargs["sigma_min"], sigma_max=kwargs["sigma_max"],
                sampler_type=kwargs["sampler_type"],
            )

        with tempfile.TemporaryDirectory() as tmp:
            latent_path = Path(tmp) / "latent.pt"
            torch.save(latent, latent_path)
            element = SimpleNamespace(
                id="latent-id", file=SimpleNamespace(path=str(latent_path)),
                context={"inversion_metadata": inversion_meta},
            )
            with patch.object(sat_samp, "sample_diffusion", fake_original_sample_diffusion), \
                    patch.object(sat_gen, "sample_diffusion", fake_original_sample_diffusion), \
                    patch.object(sat_samp.K.sampling, "sample_dpmpp_2m", fake_k_sampler), \
                    patch("engine.model_adapters.stable_audio_adapter.generate_diffusion_cond",
                          fake_generate_diffusion_cond):
                adapter.generate(
                    init_latent_element=element, steps=50, cfg_scale=1.0, sigma_min=0.3,
                    sigma_max=500.0, seed=0, sampler_category="k_diffusion",
                    k_sampler_type=sampler_type, rf_sampler_type="euler", seconds_total=0.3,
                )
        return recorded

    def test_v_model_keeps_its_sigma_range(self) -> None:
        recorded = self._generate_from_latent("v", {
            "algorithm": "euler_ode_inversion", "inversion_steps": 50, "stop_step_index": 50,
            "inversion_strength": 1.0, "sigma_min": 0.3, "sigma_max": 500.0,
        })
        self.assertEqual(recorded["sigma_max"], 500.0)
        self.assertNotIn("init_noise_level", recorded)

    def test_rf_model_starts_at_inversion_strength(self) -> None:
        recorded = self._generate_from_latent("rectified_flow", {
            "algorithm": "native_rf_inversion", "inversion_steps": 50, "stop_step_index": 50,
            "inversion_strength": 0.8, "sigma_min": 0.0, "sigma_max": 1.0,
        })
        self.assertEqual(recorded["sigma_max"], 0.8)
        self.assertEqual(recorded["init_noise_level"], 0.8)

    def test_partial_inversion_resumes_from_its_stop_sigma(self) -> None:
        import torch
        from k_diffusion.sampling import get_sigmas_karras

        latent = torch.randn(1, 64, 8)
        recorded = self._generate_from_latent("v", {
            "algorithm": "euler_ode_inversion", "inversion_steps": 50, "stop_step_index": 20,
            "inversion_strength": 0.4, "sigma_min": 0.3, "sigma_max": 500.0,
            "latent_mean": 0.5, "latent_std": 3.0,
        }, latent=latent, route_through_sample_k=True)

        inversion_grid = get_sigmas_karras(51, 0.3, 500.0)[:-1].flip(0)
        sigmas = recorded["k_sigmas"]
        self.assertAlmostEqual(sigmas[0].item(), inversion_grid[20].item(), places=4)
        self.assertEqual(len(sigmas), 22)
        self.assertTrue(torch.allclose(recorded["k_x"], latent * 3.0 + 0.5))
        # Conditioning kwargs still reach the denoiser
        self.assertEqual(recorded["k_extra_args"]["cfg_scale"], 1.0)

    def test_sampler_without_a_sigma_schedule_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "k-dpm-fast"):
            self._generate_from_latent("v", {
                "algorithm": "euler_ode_inversion", "inversion_steps": 50, "stop_step_index": 50,
                "inversion_strength": 1.0, "sigma_min": 0.3, "sigma_max": 500.0,
            }, sampler_type="k-dpm-fast", route_through_sample_k=True)


class TestInvertedLatentSchedule(unittest.TestCase):
    def test_schedule_retraces_the_inversion_grid(self) -> None:
        from k_diffusion.sampling import get_sigmas_karras
        from engine.model_adapters.stable_audio_adapter import _inverted_latent_sigmas

        ascending = get_sigmas_karras(51, 0.3, 500.0)[:-1].flip(0)
        for stop in (0, 1, 20, 50):
            sigmas = _inverted_latent_sigmas(50, stop, 0.3, 500.0, "cpu")
            self.assertEqual(len(sigmas), stop + 2)
            self.assertAlmostEqual(sigmas[0].item(), ascending[stop].item(), places=4)
            self.assertAlmostEqual(sigmas[-2].item(), 0.3, places=4)
            self.assertEqual(sigmas[-1].item(), 0.0)

    def test_out_of_range_stop_step_is_rejected(self) -> None:
        from engine.model_adapters.stable_audio_adapter import _inverted_latent_sigmas

        with self.assertRaises(ValueError):
            _inverted_latent_sigmas(50, 51, 0.3, 500.0, "cpu")

    def test_legacy_latents_restore_only_at_full_strength(self) -> None:
        import torch
        from engine.model_adapters.stable_audio_adapter import _restore_inverted_latent_scale

        latent = torch.randn(1, 4, 4)
        full = {"inversion_steps": 50, "stop_step_index": 50}
        self.assertTrue(torch.allclose(_restore_inverted_latent_scale(latent, full, 500.0), latent * 500.0))
        with self.assertRaisesRegex(ValueError, "Re-invert"):
            _restore_inverted_latent_scale(latent, {"inversion_steps": 50, "stop_step_index": 20}, 12.0)


if __name__ == "__main__":
    unittest.main()
