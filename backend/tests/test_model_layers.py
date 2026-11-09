import sys
import unittest
from pathlib import Path
from typing import Callable

import torch
from torch import nn

backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from engine.engine import Engine
from engine.model_adapters.stylegan_adapter import Generator


class AudioLikeModel(nn.Module):
    """Layer kinds found in audio models: linear layers on (B, T, C) and 1D convolutions on (B, C, T)."""

    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(8, 8)
        self.conv = nn.Conv1d(8, 8, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.proj(x).transpose(1, 2))


class TestModelLayers(unittest.TestCase):
    """spatial_output marks the layers whose output the spatial bending kernels can transform."""

    def assert_flags_match_outputs(self, model: nn.Module, run: Callable[[], object]) -> None:
        layers = Engine._extract_model_layers(Engine, model)
        self.assertTrue(layers)
        output_dims: dict[str, int] = {}
        handles = []
        for layer in layers:
            module = model.get_submodule(layer["address"])

            def record(_module: nn.Module, _inputs: tuple, output: torch.Tensor, address: str = layer["address"]) -> None:
                output_dims.setdefault(address, output.ndim)

            handles.append(module.register_forward_hook(record))
        try:
            with torch.no_grad():
                run()
        finally:
            for handle in handles:
                handle.remove()

        for layer in layers:
            if layer["address"] not in output_dims:
                continue  # Not reached by this forward pass
            with self.subTest(layer=layer["address"], type=layer["type"]):
                self.assertEqual(layer["spatial_output"], output_dims[layer["address"]] == 4)

    def test_stylegan_layers(self) -> None:
        generator = Generator(16, 32, 2, channel_multiplier=1).eval()
        self.assert_flags_match_outputs(generator, lambda: generator([torch.randn(1, 32)]))
        flags = {layer["type"]: layer["spatial_output"] for layer in Engine._extract_model_layers(Engine, generator)}
        self.assertTrue(flags["StyledConv"])
        self.assertTrue(flags["ModulatedConv2d"])
        self.assertFalse(flags["EqualLinear"])

    def test_audio_like_layers(self) -> None:
        model = AudioLikeModel().eval()
        self.assert_flags_match_outputs(model, lambda: model(torch.randn(1, 16, 8)))
        self.assertFalse(any(layer["spatial_output"] for layer in Engine._extract_model_layers(Engine, model)))


if __name__ == "__main__":
    unittest.main()
