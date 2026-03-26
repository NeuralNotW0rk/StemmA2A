import json
import struct
from pathlib import Path

import torch
import torchaudio
from einops import rearrange
from stable_audio_tools import create_model_from_config
from stable_audio_tools.inference.generation import generate_diffusion_cond
from safetensors.torch import load_file
from coolname import generate_slug

from .base import ModelAdapter
from param_graph.elements.base_elements import Asset
from param_graph.elements.artifacts.audio import Audio
from param_graph.elements.models.stable_audio import StableAudioModel
from utils.uid import UIDMismatchError


# Workaround for the file extension-based safetensors loading in stable audio tools
def load_ckpt_state_dict(ckpt_path):
    with open(ckpt_path, "rb") as f:
        # Check for Safetensors: 8-byte header length + starts with '{'
        header = f.read(9)
        is_safetensors = (len(header) == 9 and 
                          header[8:9] == b'{' and 
                          struct.unpack("<Q", header[:8])[0] < 100_000_000)
        
    if is_safetensors:
        state_dict = load_file(ckpt_path)
    else:
        state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)["state_dict"]
    
    return state_dict


class StableAudioAdapter(ModelAdapter):
    def __init__(self) -> None:
        super().__init__()
        self.name = 'stable_audio_tools'
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.model_info: StableAudioModel | None = None

    def register_model(self, **kwargs) -> StableAudioModel:
        config_path = kwargs.get("config_path")
        checkpoint_path = kwargs.get("checkpoint_path")

        # Load the config
        with open(config_path, 'r') as cf:
            config_json = cf.read()
            config = json.loads(config_json)

        # Create a temporary model object to generate a UID from
        model = create_model_from_config(config)
        model.load_state_dict(load_ckpt_state_dict(checkpoint_path))

        model_id = self.uid_generator.from_module(model)

        return StableAudioModel(
            id=model_id,
            name=kwargs.get("name"),
            checkpoint=Asset(path=checkpoint_path, uid=model_id),
            config=config,
            model_type=kwargs.get("model_type"),
            context={}
        )
        
    def load_model(self, info: StableAudioModel, verify: bool = True):
        # If a model is loaded, check if it's the same one
        if self.model and self.model_info.id == info.id:
            return # Same model, do nothing

        # Unload existing model if there is one
        if self.model:
            del self.model
            self.model = None

        # Load the new model
        self.model = create_model_from_config(info.config)
        self.model.load_state_dict(load_ckpt_state_dict(info.checkpoint.path))
        self.model_info = info

        if verify and self.uid_generator.from_module(self.model) != info.id:
            raise UIDMismatchError("Checkpoint UID mismatch")

    def generate(self, **kwargs) -> tuple[Audio, torch.Tensor]:
        model = self.model.to(self.device)
        sample_rate = self.model_info.config["sample_rate"]
        sample_size = self.model_info.config["sample_size"]
        seconds_start = kwargs.get("seconds_start", 0)
        seconds_total = kwargs.get("seconds_total", 11)

        # Set up text and timing conditioning
        conditioning = [{
            "prompt": kwargs.get("prompt", ""),
            "seconds_start": seconds_start,
            "seconds_total": seconds_total
        }]

        negative_prompt = kwargs.get("negative_prompt", "")
        negative_conditioning = None
        if negative_prompt:
            negative_conditioning = [{
                "prompt": negative_prompt,
                "seconds_start": seconds_start,
                "seconds_total": seconds_total
            }]

        print(f"Generating with conditioning:{str(conditioning)}")
        if negative_prompt:
            print(f"Generating with negative_conditioning:{str(negative_conditioning)}")
        print(f"Sample Rate: {sample_rate}")
        print(f"Sample Size: {sample_size}")

        sampler_type = "pingpong" # default
        if self.model_info.model_type == "k_diffusion":
            # The key from the form is 'k_sampler_type'
            sampler_type = kwargs.get("k_sampler_type")
        elif self.model_info.model_type == "rectified_flow":
            # The key from the form is 'rf_sampler_type'
            sampler_type = kwargs.get("rf_sampler_type")
        
        if not sampler_type:
            sampler_type = "pingpong"
            
        # Set up generation arguments
        args = {
            "steps": kwargs.get("steps", 8),
            "cfg_scale": kwargs.get("cfg_scale", 1.0),
            "conditioning": conditioning,
            "negative_conditioning": negative_conditioning,
            "batch_size": kwargs.get("batch_size", 1),
            "sample_size": sample_size,
            "sigma_min": kwargs.get("sigma_min", 0.3),
            "sigma_max": kwargs.get("sigma_max", 500),
            "sampler_type": sampler_type,
            "device": self.device,
            "seed": kwargs.get("seed", 0)
        }

        # Handle initial audio if provided
        init_audio_element = kwargs.get("init_audio_element")
        if init_audio_element:
            noise_level = kwargs.get("noise_level", 0.7)
            
            # Load audio
            audio_path = Path(init_audio_element.file.path)
            if audio_path and audio_path.exists():
                init_audio_tensor, init_audio_sample_rate = torchaudio.load(audio_path)
                init_audio_tensor = init_audio_tensor.to(self.device)

                # Resample if necessary
                if init_audio_sample_rate != sample_rate:
                    resampler = torchaudio.transforms.Resample(
                        orig_freq=init_audio_sample_rate, 
                        new_freq=sample_rate
                    ).to(self.device)
                    init_audio_tensor = resampler(init_audio_tensor)
                
                args["init_audio"] = (sample_rate, init_audio_tensor)
                args["init_noise_level"] = noise_level

        # Generate stereo audio
        output = generate_diffusion_cond(model, **args)

        # Trim silence
        output = output[:,:,:int(seconds_total*sample_rate)]

        # Rearrange audio batch to a single sequence
        print("Generation complete, rearranging...")
        output = rearrange(output, "b d n -> d (b n)")

        # Peak normalize, clip, convert to int16
        output = output.to(torch.float32).div(torch.max(torch.abs(output))).clamp(-1, 1).mul(32767).to(torch.int16).cpu()

        # Create audio artifact
        content_uid = self.uid_generator.from_tensor(output)
        artifact = Audio(
            id=content_uid,
            name=generate_slug(2),
            file=Asset(path=None, uid=content_uid, extension=".wav"),
            sample_rate=sample_rate,
            context=kwargs
        )

        return artifact, output
