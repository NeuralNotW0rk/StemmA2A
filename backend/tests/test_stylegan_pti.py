import importlib.util
import math
import os
import sys
import tempfile
import unittest
from pathlib import Path

import torch
from torchvision.utils import save_image

backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from engine.model_adapters.stylegan_adapter import Generator, StyleGANAdapter
from engine.model_adapters.stylegan_pti import load_target_image, pivotal_tuning_inversion, render
from engine.results import OperationResult
from param_graph.elements.base_elements import Asset
from param_graph.elements.artifacts.image_element import Image
from diffracture.actant import Actant
from diffracture.topology.grating import Grating


class TestPivotalTuning(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.model = Generator(size=16, style_dim=64, n_mlp=2, channel_multiplier=1).eval()
        with torch.no_grad():
            self.w_avg = self.model.mean_latent(1000)
            # A target the generator can almost reach: its own render, with a perturbation no w explains
            target = render(self.model, self.model.style(torch.randn(1, 64)))
            self.target = (target + 0.3 * torch.randn_like(target)).clamp(-1, 1)

    def test_tuning_improves_on_projection_and_leaves_generator_untouched(self):
        before = {k: v.clone() for k, v in self.model.state_dict().items()}
        result = pivotal_tuning_inversion(
            self.model, self.target, self.w_avg,
            projection_steps=40, tuning_steps=40, rank=4, tuning_lr=3e-3, use_lpips=False
        )

        self.assertEqual(tuple(result["w_pivot"].shape), (1, 64))
        self.assertLess(result["tuning_loss"], result["projection_loss"])

        after = self.model.state_dict()
        self.assertEqual(before.keys(), after.keys())
        self.assertTrue(all(torch.equal(before[k], after[k]) for k in before))
        self.assertTrue(all(p.requires_grad for p in self.model.parameters()))

        # Applying the tuned grating at the pivot reproduces the reported tuning loss
        actant = Actant(self.model)
        actant.activate(result["grating"], injection_strategy="hook")
        with torch.no_grad():
            tuned = torch.nn.functional.mse_loss(render(self.model, result["w_pivot"]), self.target).item()
        actant.deactivate()
        self.assertLess(tuned, result["projection_loss"])

    def test_load_target_image_center_crops(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "wide.png")
            img = torch.zeros(3, 20, 40)
            img[:, :, 10:30] = 1.0  # white center square, black side bars
            save_image(img, path)
            target = load_target_image(path, 16, "cpu")
        self.assertEqual(tuple(target.shape), (1, 3, 16, 16))
        self.assertGreater(target.min().item(), 0.9)


class TestStyleGANAdapterInversion(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.adapter = StyleGANAdapter()
        info = self.adapter.register_model(name="PTI Test", checkpoint_path="", size=32, channel_multiplier=1)
        self.adapter.load_model(info)
        self.tmp = tempfile.TemporaryDirectory()
        path = os.path.join(self.tmp.name, "source.png")
        save_image(torch.rand(3, 32, 32), path)
        self.source = Image(id="img_source", name="source", context={}, file=Asset(path=path, uid="img_source", extension=".png"))

    def tearDown(self):
        self.tmp.cleanup()

    def invert(self, **kwargs) -> OperationResult:
        params = {"projection_steps": 3, "tuning_steps": 3, "rank": 2, "use_lpips": False, **kwargs}
        return self.adapter.invert(source_image_element=self.source, **params)

    def test_invert_returns_pivot_latent_and_linked_grating(self):
        result = self.invert()
        self.assertIsInstance(result, OperationResult)
        (latent, w_pivot), (grating, grating_data) = result.outputs

        self.assertEqual(latent.type, "latent")
        self.assertEqual(tuple(w_pivot.shape), (1, 512))
        self.assertTrue(torch.equal(torch.load(latent.file.path), w_pivot))
        self.assertEqual(latent.context["source_image_id"], "img_source")
        self.assertEqual(latent.context["operation"], "invert")

        self.assertIsNone(grating_data)
        self.assertEqual(grating.type, "grating")
        self.assertEqual(grating.base_model_id, self.adapter.model_info.id)
        self.assertEqual(grating.context["pivot_latent"], latent.id)
        self.assertEqual(grating.context["operation"], "invert")
        self.assertTrue(all(e["kernel_type"] == "lora" for e in grating.elements))
        self.assertEqual(len(Grating.load(grating.file.path).nodes), len(grating.elements))

        self.assertEqual(len(result.links), 1)
        link = result.links[0]
        self.assertEqual((link.source, link.target, link.relation), (latent.id, grating.id, "pivot"))

    def test_invert_records_reconstruction_loss(self):
        (latent, _), (grating, _) = self.invert(projection_steps=2, tuning_steps=2).outputs
        self.assertEqual(latent.context["reconstruction_loss"], "mse")
        self.assertEqual(grating.context["reconstruction_loss"], "mse")

    @unittest.skipUnless(importlib.util.find_spec("lpips"), "lpips is not installed")
    def test_invert_with_lpips(self):
        (_, _), (grating, _) = self.invert(projection_steps=2, tuning_steps=2, use_lpips=True).outputs
        self.assertEqual(grating.context["reconstruction_loss"], "lpips+mse")
        self.assertTrue(math.isfinite(grating.context["tuning_loss"]))

    def test_generate_from_pivot_with_grating_reproduces_tuned_render(self):
        (latent, w_pivot), (grating, _) = self.invert().outputs

        from_latent, img = self.adapter.generate(seed=1, init_latent_element=latent)
        _, other_seed_img = self.adapter.generate(seed=2, init_latent_element=latent)
        seeded, seeded_img = self.adapter.generate(seed=1)

        self.assertEqual(from_latent.context["init_latent_id"], latent.id)
        self.assertTrue(torch.equal(img, other_seed_img))  # the latent, not the seed, sets the image
        self.assertNotIn("init_latent_id", seeded.context)
        self.assertFalse(torch.equal(img, seeded_img))

        # Truncation is not applied to an explicit latent
        _, untruncated = self.adapter.generate(init_latent_element=latent, truncation=0.5)
        self.assertTrue(torch.equal(img, untruncated))

        # With the grating applied, the pivot renders through the tuned generator
        actant = Actant(self.adapter.model)
        actant.activate(Grating.load(grating.file.path).to(self.adapter.device), injection_strategy="hook")
        _, tuned_img = self.adapter.generate(init_latent_element=latent)
        actant.deactivate()
        self.assertFalse(torch.equal(img, tuned_img))

    def test_generate_rejects_latent_of_wrong_shape(self):
        path = os.path.join(self.tmp.name, "audio_latent.pt")
        torch.save(torch.randn(1, 64, 215), path)
        from param_graph.elements.artifacts.latent_element import Latent
        latent = Latent(id="lat_audio", name="audio", context={}, file=Asset(path=path, uid="lat_audio", extension=".pt"))
        with self.assertRaises(ValueError):
            self.adapter.generate(init_latent_element=latent)


if __name__ == "__main__":
    unittest.main()
