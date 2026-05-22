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
from tqdm import tqdm
from k_diffusion.sampling import get_sigmas_karras

from .base_adapter import ModelAdapter
from param_graph.elements.base_elements import Asset
from param_graph.elements.artifacts.audio_element import Audio
from param_graph.elements.artifacts.latent_element import Latent
from param_graph.elements.models.stable_audio_element import StableAudioModel
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

    def cleanup(self):
        if self.model:
            del self.model
            self.model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

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

        sampler_type = None
        if self.model_info.model_type == "k_diffusion":
            # The key from the form is 'k_sampler_type'
            sampler_type = kwargs.get("k_sampler_type")
            if not sampler_type:
                raise ValueError("k_sampler_type is required for k_diffusion models.")
        elif self.model_info.model_type == "rectified_flow":
            # The key from the form is 'rf_sampler_type'
            sampler_type = kwargs.get("rf_sampler_type")
            if not sampler_type:
                raise ValueError("rf_sampler_type is required for rectified_flow models.")
        
        if not sampler_type:
            raise ValueError(f"Unable to determine sampler type for model_type: {self.model_info.model_type}")
            
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
                from utils.audio import load_audio
                init_audio_tensor = load_audio(self.device, audio_path, sample_rate)
                
                args["init_audio"] = (sample_rate, init_audio_tensor)
                args["init_noise_level"] = noise_level

        # Handle initial latent noise if provided
        init_latent_element = kwargs.get("init_latent_element")
        init_latent_tensor = None
        if init_latent_element:
            latent_path = Path(init_latent_element.file.path)
            if latent_path and latent_path.exists():
                init_latent_tensor = torch.load(latent_path, map_location=self.device, weights_only=True)

        # Generate stereo audio
        with torch.no_grad():
            if init_latent_tensor is not None:
                # stable-audio-tools doesn't expose a custom noise parameter, so we intercept the sampler hook
                import stable_audio_tools.inference.generation as sat_gen
                original_sample_k = sat_gen.sample_k
                
                def patched_sample_k(*inner_args, **inner_kwargs):
                    inner_args = list(inner_args)
                    noise = inner_args[1] if len(inner_args) > 1 else inner_kwargs.get("noise")
                    
                    if noise is not None:
                        custom_noise = init_latent_tensor.clone()
                        
                        # Expand to match batch size if necessary
                        if custom_noise.shape[0] == 1 and noise.shape[0] > 1:
                            custom_noise = custom_noise.repeat(noise.shape[0], 1, 1)
                            
                        # Align sequence lengths just in case of slight rounding differences
                        if custom_noise.shape[-1] != noise.shape[-1]:
                            target_len = noise.shape[-1]
                            if custom_noise.shape[-1] > target_len:
                                custom_noise = custom_noise[..., :target_len]
                            else:
                                custom_noise = torch.nn.functional.pad(custom_noise, (0, target_len - custom_noise.shape[-1]))

                            # Match target noise distribution to prevent high-frequency artifacts
                            target_std = noise.std().item()
                            current_std = custom_noise.std().item()
                            if current_std > 0:
                                custom_noise = custom_noise * (target_std / current_std)
                                
                            target_mean = noise.mean().item()
                            current_mean = custom_noise.mean().item()
                            custom_noise = custom_noise - current_mean + target_mean

                        if len(inner_args) > 1:
                            inner_args[1] = custom_noise
                        else:
                            inner_kwargs["noise"] = custom_noise

                    return original_sample_k(*inner_args, **inner_kwargs)

                sat_gen.sample_k = patched_sample_k
                try:
                    output = generate_diffusion_cond(model, **args)
                finally:
                    # Restore original hook immediately after generation completes
                    sat_gen.sample_k = original_sample_k
            else:
                output = generate_diffusion_cond(model, **args)

            # Trim silence
            output = output[:,:,:int(seconds_total*sample_rate)]

            # Rearrange audio batch to a single sequence
            print("Generation complete, rearranging...")
            output = rearrange(output, "b d n -> d (b n)")

            # Peak normalize, clip, convert to int16
            output = output.to(torch.float32).div(torch.max(torch.abs(output))).clamp(-1, 1).mul(32767).to(torch.int16).cpu()
            
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Create audio artifact
        content_uid = self.uid_generator.from_tensor(output)

        # Filter context
        context = {}
        for k, v in kwargs.items():
            if k.endswith('_element'):
                context[k.replace('_element', '_id')] = v.id
            elif k.endswith('_elements'):
                context[k.replace('_elements', '_ids')] = [el.id for el in v]
            else:
                context[k] = v

        artifact = Audio(
            id=content_uid,
            name=generate_slug(2),
            file=Asset(path=None, uid=content_uid, extension=".wav"),
            sample_rate=sample_rate,
            context=context
        )

        return artifact, output

    def invert(self, **kwargs) -> tuple[Latent, torch.Tensor]:
        model = self.model.to(self.device)
        sample_rate = self.model_info.config["sample_rate"]
        seconds_start = kwargs.get("seconds_start", 0)
        seconds_total = kwargs.get("seconds_total", 11)

        # Set up text and timing conditioning
        conditioning = [{
            "prompt": kwargs.get("prompt", ""),
            "seconds_start": seconds_start,
            "seconds_total": seconds_total
        }]

        steps = kwargs.get("steps", 50)
        inversion_strength = kwargs.get("inversion_strength", 0.5)
        
        print(f"Inverting with conditioning:{str(conditioning)}")
        print(f"Steps: {steps}")
        print(f"Inversion Strength: {inversion_strength}")
        print(f"Sample Rate: {sample_rate}")
        
        source_audio_element = kwargs.get("source_audio_element")
        if not source_audio_element:
            raise ValueError("source_audio_element is required for DDIM inversion.")
            
        audio_path = Path(source_audio_element.file.path)
        if audio_path and audio_path.exists():
            from utils.audio import load_audio
            source_audio_tensor = load_audio(self.device, audio_path, sample_rate)
        else:
            raise FileNotFoundError(f"Audio path not found: {audio_path}")

        source_audio_tensor = source_audio_tensor.unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # Encode to get the clean latent representation (x_0)
            init_latents = model.pretransform.encode(source_audio_tensor)
            if isinstance(init_latents, tuple):
                init_latents = init_latents[0]

            # Preprocess the text/timing conditioning
            raw_cond = model.conditioner(conditioning, self.device)
            
            # Explicitly create a new dictionary with only the embedded tensors the transformer expects
            valid_cond_keys = ["cross_attn_cond", "global_cond", "prepend_cond", "prepend_inputs", "cross_attn_cond_mask"]
            cond = {k: raw_cond[k] for k in valid_cond_keys if k in raw_cond}

        # 1. Build Continuous Sigmas Grid
        sigma_min = kwargs.get("sigma_min", 0.3)
        sigma_max = kwargs.get("sigma_max", 500.0)
        
        # Request steps+1 to get enough intervals, drop the 0.0 at the end, and flip to go min->max
        sigmas = get_sigmas_karras(steps + 1, sigma_min, sigma_max, device=self.device)[:-1].flip(0)

        stop_step = int(steps * inversion_strength)
        
        current_latents = init_latents

        # 2. Euler ODE Inversion Step Loop: Moving step-by-step from clean data to structured noise
        for i in tqdm(range(stop_step), desc="Euler ODE Inversion"):
            sigma_curr = sigmas[i]
            sigma_next = sigmas[i + 1]
            
            # Expand sigma_curr into a matching tensor shape
            t_tensor = torch.full((current_latents.shape[0],), sigma_curr.item(), device=self.device)
            
            with torch.no_grad():
                # Query the DiT model backbone using its native forward pass signature
                denoised_pred = model.model(current_latents, t_tensor, **cond)
                
            # Compute the local trajectory velocity vector
            v_velocity = (current_latents - denoised_pred) / sigma_curr
            
            # Integrate the new latent coordinate upstream using an explicit Euler stepping equation
            d_sigma = sigma_next - sigma_curr
            current_latents = current_latents + v_velocity * d_sigma

        # In stable-audio-tools, the generation sampler expects unit-variance initial noise.
        # Due to Euler integration discretization errors across large step sizes, the variance 
        # can drift significantly. We explicitly normalize the inverted latents to unit variance.
        latent_std = current_latents.std().item()
        if latent_std > 0:
            latent_tensor = (current_latents - current_latents.mean().item()) / latent_std
        else:
            latent_tensor = current_latents

        # Create latent artifact tracking IDs
        content_uid = self.uid_generator.from_tensor(latent_tensor)

        # Filter context metadata properties
        context = {}
        for k, v in kwargs.items():
            if k.endswith('_element'):
                context[k.replace('_element', '_id')] = v.id
            elif k.endswith('_elements'):
                context[k.replace('_elements', '_ids')] = [el.id for el in v]
            else:
                context[k] = v

        # Inject execution variables explicitly into our self-documenting context dictionary
        context["inversion_metadata"] = {
            "algorithm": "euler_ode_inversion",
            "sampler": "euler",
            "inversion_steps": steps,
            "stop_step_index": stop_step,
            "inversion_strength": inversion_strength,
            "sigma_min": sigma_min,
            "sigma_max": sigma_max
        }

        artifact = Latent(
            id=content_uid,
            name=generate_slug(2),
            file=Asset(path=None, uid=content_uid, extension=".pt"),
            context=context
        )
        
        return artifact, latent_tensor
