from dataclasses import dataclass
import torch
import torchaudio
from stable_audio_tools import get_pretrained_model
from stable_audio_tools.inference.generation import generate_diffusion_cond

from engine import Engine

from ..elements.model import Model

@dataclass(kw_only=True)
class StableAudioModel(Model):

    def __post_init__(self):
        self.engine = 'StableAudioTools'


class StableAudioTools(Engine):
    def __init__(self) -> None:
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.model_config = None

    def load_default_model(self):
        # Download model
        self.model, self.model_config = get_pretrained_model("stabilityai/stable-audio-open-small")

    def load_model(self):
        pass

    def generate(self):

        model = model.to(self.device)

        # Set up text and timing conditioning
        conditioning = [{
            "prompt": "128 BPM tech house drum loop",
            "seconds_total": 11
        }]

        # Generate stereo audio
        output = generate_diffusion_cond(
            model,
            steps=8,
            cfg_scale=1.0,
            conditioning=conditioning,
            sample_size=self.model_config["sample_size"],
            sampler_type="pingpong",
            device=self.device
        )

        # Peak normalize, clip, convert to int16, and save to file
        output = output.to(torch.float32).div(torch.max(torch.abs(output))).clamp(-1, 1).mul(32767).to(torch.int16).cpu()
        torchaudio.save("output.wav", output, self.model_config["sample_rate"])
