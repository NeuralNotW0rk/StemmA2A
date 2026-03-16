import json
import os
import torch
import torchaudio
from pathlib import Path

from einops import rearrange
from stable_audio_tools import create_model_from_config
from stable_audio_tools.models.utils import load_ckpt_state_dict
from stable_audio_tools.inference.generation import generate_diffusion_cond

from .base import ModelAdapter
from utils.uid import UIDMismatchError, path_from_uid
from param_graph.elements.artifacts.audio import Audio
from param_graph.elements.models.stable_audio import StableAudioModel

from coolname import generate_slug


class StableAudioAdapter(ModelAdapter):
    def __init__(self) -> None:
        super().__init__()
        self.name = 'stable_audio_tools'
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.model_info: StableAudioModel | None = None

    def register_model(self, **kwargs) -> StableAudioModel:
        config_path = kwargs.get("config_path")
        with open(config_path, 'r') as cf:
            config_json = cf.read()
            config = json.loads(config_json)
        
        config_uid = self.uid_generator.from_dict(config)

        # Create a temporary model object to generate a UID from
        model = create_model_from_config(config)
        model.load_state_dict(load_ckpt_state_dict(kwargs.get("checkpoint_path")))

        checkpoint_uid = self.uid_generator.from_module(model)

        # Combine the UIDs to create a single ID
        model_id = self.uid_generator.from_uids([checkpoint_uid, config_uid])

        return StableAudioModel(
            **kwargs,
            config=config,
            config_uid=config_uid,
            id=model_id,
            checkpoint_uid=checkpoint_uid
        )
        
    def load_model(self, info: StableAudioModel, verify: bool = True):
        # If a model is loaded, check if it's the same one
        if self.model and self.model_info.id == info.id:
            return # Same model, do nothing

        # Unload existing model if there is one
        if self.model:
            del self.model
            self.model = None

        # Load the config
        with open(self.model_info.config_path, 'r') as cf:
            config_json = cf.read()
            config = json.loads(config_json)
        
        if verify and self.uid_generator.from_dict(config) != info.config_uid:
            raise UIDMismatchError("Config UID mismatch")

        # Load the new model
        self.model = create_model_from_config(info.config)
        self.model.load_state_dict(load_ckpt_state_dict(info.checkpoint_path))
        self.model_info = info

        if verify and self.uid_generator.from_module(self.model) != info.checkpoint_uid:
            raise UIDMismatchError("Checkpoint UID mismatch")
            

    def generate(self, output_dir: Path, **kwargs) -> Audio:
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
        path = output_dir / path_from_uid(id)
        torchaudio.save(path, output, sample_rate)

        # Create audio artifact
        return Audio(
            id=id,
            path=str(path),
            name=generate_slug(2),
        )
