import json
import os
import torch
import torchaudio
from einops import rearrange
from stable_audio_tools import get_pretrained_model, create_model_from_config
from stable_audio_tools.models.utils import load_ckpt_state_dict
from stable_audio_tools.inference.generation import generate_diffusion_cond

from param_graph.engine import Engine
from param_graph.uid_gen import UIDMismatchError
from param_graph.elements.artifacts.audio import Audio
from param_graph.elements.models.stable_audio import StableAudioModel

from coolname import generate_slug


class StableAudioTools(Engine):
    def __init__(self) -> None:
        super().__init__()
        self.name = 'stable_audio_tools'
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.model_info = None

    def register_model(self, **kwargs) -> StableAudioModel:
        config_path = kwargs.pop("config_path")
        with open(config_path, 'r') as cf:
            config = json.load(cf)
        
        # Create a temporary model object to generate a UID from
        model = create_model_from_config(config)
        model.load_state_dict(load_ckpt_state_dict(kwargs.get("checkpoint_path")))

        return StableAudioModel(
            **kwargs,
            config=config,
            id=self.uid_generator.from_module(model)
        )

    def load_model(self, info: StableAudioModel, verify: bool=True):
        # If a model is loaded, check if it's the same one
        if self.model and self.model_info.id == info.id:
            return # Same model, do nothing

        # Unload existing model if there is one
        if self.model:
            del self.model
            self.model = None

        # Load the new model
        self.model = create_model_from_config(info.config)
        self.model.load_state_dict(load_ckpt_state_dict(info.checkpoint_path))
        self.model_info = info

    def generate(self, output_dir: str, **kwargs):
        model = self.model.to(self.device)
        sample_rate = self.model_info.config["sample_rate"]
        sample_size = self.model_info.config["sample_size"]


        # Set up text and timing conditioning
        conditioning = [{
            "prompt": kwargs.get("prompt", ""),
            "seconds_total": kwargs.get("seconds_total", 11)
        }]

        print(f"Generating with conditioning:{str(conditioning)}")
        print(f"Sample Rate: {sample_rate}")
        print(f"Sample Size: {sample_size}")

        # Generate stereo audio
        output = generate_diffusion_cond(
            model,
            steps=kwargs.get("steps", 8),
            cfg_scale=kwargs.get("cfg_scale", 1.0),
            conditioning=conditioning,
            sample_size=sample_size,
            sampler_type=kwargs.get("sampler_type", "pingpong"),
            device=self.device,
            seed=kwargs.get("seed", 0)
        )
        
        print("Generation complete, rearranging...")

        # Rearrange audio batch to a single sequence
        output = rearrange(output, "b d n -> d (b n)")

        # Peak normalize, clip, convert to int16
        output = output.to(torch.float32).div(torch.max(torch.abs(output))).clamp(-1, 1).mul(32767).to(torch.int16).cpu()

        # Save to file
        id = self.uid_generator.from_tensor(output)
        filename = f"{id}.wav"
        output_path = os.path.join(output_dir, filename)
        torchaudio.save(output_path, output, sample_rate)

        # Create audio artifact
        return Audio(
            id=self.uid_generator.from_tensor(output),
            name=generate_slug(2),
            path=output_path,
        )
