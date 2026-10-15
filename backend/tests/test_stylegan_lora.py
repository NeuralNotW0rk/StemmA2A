import sys
import unittest
from pathlib import Path

import torch

backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from engine.model_adapters.stylegan_adapter import Generator
from diffracture.actant import Actant
from diffracture.topology.grating import Grating
from diffracture.topology.lora import LoRAElement, find_lora_targets


class TestStyleGANWeightSpaceLoRA(unittest.TestCase):
    """LoRA on StyleGAN2's modulated convs goes through the hook injector's weight-space fallback."""

    def setUp(self):
        torch.manual_seed(0)
        self.model = Generator(size=32, style_dim=64, n_mlp=2, channel_multiplier=1).eval()
        self.w = torch.randn(1, 64)

    def render(self):
        img, _ = self.model([self.w], input_is_latent=True, randomize_noise=False)
        return img

    def synthesis_grating(self, up_std):
        grating = Grating()
        targets = find_lora_targets(self.model, include=r"(conv|to_rgb).*\.conv\.weight$")
        for address, param, weight in targets:
            element = LoRAElement.for_parameter(address, weight, rank=2, param=param)
            with torch.no_grad():
                element.params["lora_up"].normal_(0, up_std)
            grating.add_element(element)
        return grating, targets

    def test_targets_cover_synthesis_convs(self):
        _, targets = self.synthesis_grating(0.0)
        addresses = [address for address, _, _ in targets]
        self.assertIn("conv1.conv", addresses)
        self.assertIn("convs.0.conv", addresses)
        self.assertIn("to_rgbs.0.conv", addresses)
        self.assertFalse(any(a.startswith("style") for a in addresses))

    def test_lora_bends_and_restores_generator(self):
        base = self.render()
        grating, _ = self.synthesis_grating(0.05)

        actant = Actant(self.model)
        actant.activate(grating, injection_strategy="hook")
        bent = self.render()
        actant.deactivate()

        self.assertGreater((bent - base).abs().mean().item(), 1e-3)
        self.assertTrue(torch.equal(self.render(), base))

    def test_lora_trains_against_frozen_generator(self):
        target = torch.zeros(1, 3, 32, 32)
        grating, _ = self.synthesis_grating(0.0)
        params = [p for element in grating._nodes for p in element.params.values()]

        actant = Actant(self.model)
        actant.activate(grating, injection_strategy="hook")
        # The injector freezes the base weights it parametrizes, never the LoRA factors
        for address in grating.nodes:
            self.assertFalse(self.model.get_submodule(address).parametrizations.weight.original.requires_grad)
        self.assertTrue(all(p.requires_grad for p in params))

        opt = torch.optim.Adam(params, lr=3e-3)
        losses = []
        for _ in range(10):
            opt.zero_grad()
            loss = (self.render() - target).pow(2).mean()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        actant.deactivate()

        self.assertLess(losses[-1], losses[0])


if __name__ == "__main__":
    unittest.main()
