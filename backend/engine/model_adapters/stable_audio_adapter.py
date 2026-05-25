import json
import struct
import math
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
            sampler_type = kwargs.get("k_sampler_type")
            if not sampler_type:
                raise ValueError("k_sampler_type is required for k_diffusion models.")
        elif self.model_info.model_type == "rectified_flow":
            sampler_type = kwargs.get("rf_sampler_type")
            if not sampler_type:
                raise ValueError("rf_sampler_type is required for rectified_flow models.")
        
        if not sampler_type:
            raise ValueError(f"Unable to determine sampler type for model_type: {self.model_info.model_type}")
            
        is_euler = sampler_type == "euler"
        if is_euler:
            sampler_type = "k-heun"  # Bypass stable-audio-tools string validation
            
        # Extract metadata and tensor payload from initial latent if available
        init_latent_element = kwargs.get("init_latent_element")
        init_latent_tensor = None
        inversion_meta = None
        
        if init_latent_element:
            latent_path = Path(init_latent_element.file.path)
            if latent_path and latent_path.exists():
                init_latent_tensor = torch.load(latent_path, map_location=self.device, weights_only=True)
                inversion_meta = getattr(init_latent_element, "context", {}).get("inversion_metadata", None)

        # Establish execution steps and schedule boundaries
        steps = kwargs.get("steps", 8)
        sigma_min = kwargs.get("sigma_min", 0.3)
        sigma_max = kwargs.get("sigma_max", 500.0)

        # Synchronize parameters to the latent node if using an inverted blueprint
        if inversion_meta:
            steps = inversion_meta.get("inversion_steps", steps)
            sigma_min = inversion_meta.get("sigma_min", sigma_min)
            sigma_max = inversion_meta.get("sigma_max", sigma_max)

        # Set up generation arguments
        args = {
            "steps": steps,
            "cfg_scale": kwargs.get("cfg_scale", 1.0),
            "conditioning": conditioning,
            "negative_conditioning": negative_conditioning,
            "batch_size": kwargs.get("batch_size", 1),
            "sample_size": sample_size,
            "sigma_min": sigma_min,
            "sigma_max": sigma_max,
            "sampler_type": sampler_type,
            "device": self.device,
            "seed": kwargs.get("seed", 0)
        }

        # Handle traditional initial audio if provided
        init_audio_element = kwargs.get("init_audio_element")
        if init_audio_element:
            noise_level = kwargs.get("noise_level", 0.7)
            
            audio_path = Path(init_audio_element.file.path)
            if audio_path and audio_path.exists():
                from utils.audio import load_audio
                init_audio_tensor = load_audio(self.device, audio_path, sample_rate)
                
                args["init_audio"] = (sample_rate, init_audio_tensor)
                args["init_noise_level"] = noise_level

        # Generate stereo audio
        with torch.no_grad():
            import stable_audio_tools.inference.generation as sat_gen
            import k_diffusion.sampling as k_samp
            
            original_sample_k = sat_gen.sample_k
            original_heun = k_samp.sample_heun
            
            if is_euler:
                k_samp.sample_heun = k_samp.sample_euler
                
            # ALWAYS patch sample_k to conditionally inject latent noise if present
            def patched_sample_k(*inner_args, **inner_kwargs):
                inner_args = list(inner_args)
                inner_kwargs = dict(inner_kwargs)
                
                if init_latent_tensor is not None and inversion_meta is not None:
                    # 1. FORCE the initial noise argument to be a direct clone of your inverted latent
                    # In sample_k, noise is the second positional argument (index 1)
                    custom_coords = init_latent_tensor.clone()
                    
                    # Ensure batch sizes match up smoothly
                    target_shape = inner_args[1].shape if len(inner_args) > 1 else inner_kwargs.get("noise").shape
                    if custom_coords.shape[0] == 1 and target_shape[0] > 1:
                        custom_coords = custom_coords.repeat(target_shape[0], 1, 1)

                    # 2. Extract and patch the sampler_fn to use our sliced sigmas timeline
                    sampler_fn = None
                    sampler_arg_index = None
                    
                    if "sampler_fn" in inner_kwargs:
                        sampler_fn = inner_kwargs["sampler_fn"]
                    else:
                        if len(inner_args) > 4:
                            sampler_fn = inner_args[4]
                            sampler_arg_index = 4
                    
                    if sampler_fn is None:
                        for idx, arg in enumerate(inner_args):
                            if callable(arg) and not hasattr(arg, "forward") and arg.__name__ != "patched_sample_k":
                                sampler_fn = arg
                                sampler_arg_index = idx
                                break

                    if sampler_fn is not None:
                        # Re-create the exact global timeline grid
                        global_sigmas = get_sigmas_karras(steps + 1, sigma_min, sigma_max, device=self.device)
                        stop_index = inversion_meta["stop_step_index"]
                        
                        # Flip schedule to run downstream from max noise to clean audio floor
                        downstream_sigmas = global_sigmas[:-1].flip(0)
                        
                        # Symmetrical boundary-safe timeline slice
                        start_index = steps - stop_index
                        generation_sigmas = downstream_sigmas[start_index:]
                        
                        # Append mandatory terminal 0.0 value
                        generation_sigmas = torch.cat([generation_sigmas, global_sigmas[-1:]])

                        # Build the clean proxy interceptor using the unified tensor coordinates
                        def custom_sampler_wrapper(model_wrap, x, sigmas_arg, *s_args, **s_kwargs):
                            custom_coords_local = custom_coords.clone()
                            if custom_coords_local.shape[0] == 1 and x.shape[0] > 1:
                                custom_coords_local = custom_coords_local.repeat(x.shape[0], 1, 1)
                            
                            orig_mean = inversion_meta.get("original_mean", 0.0)
                            orig_std = inversion_meta.get("original_std", 1.0)
                            custom_coords_local = (custom_coords_local * orig_std) + orig_mean
                            
                            # Detect Hybrid Mode: Combine audio baseline with inverted latent
                            if "init_audio" in args:
                                alpha = kwargs.get("noise_level", 0.7)
                                custom_coords_local = custom_coords_local * alpha
                                x = (x * math.sqrt(1.0 - alpha)) + custom_coords_local
                            else:
                                x = custom_coords_local
                            
                            return sampler_fn(model_wrap, x, generation_sigmas, *s_args, **s_kwargs)

                        if "sampler_fn" in inner_kwargs or sampler_arg_index is None:
                            inner_kwargs["sampler_fn"] = custom_sampler_wrapper
                        else:
                            inner_args[sampler_arg_index] = custom_sampler_wrapper

                return original_sample_k(*inner_args, **inner_kwargs)

            sat_gen.sample_k = patched_sample_k
            
            try:
                output = generate_diffusion_cond(model, **args)
            finally:
                # Restore original hooks immediately after execution completes
                sat_gen.sample_k = original_sample_k
                if is_euler:
                    k_samp.sample_heun = original_heun

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
            raise ValueError("source_audio_element is required for inversion.")
            
        audio_path = Path(source_audio_element.file.path)
        if audio_path and audio_path.exists():
            from utils.audio import load_audio
            source_audio_tensor = load_audio(self.device, audio_path, sample_rate)
        else:
            raise FileNotFoundError(f"Audio path not found: {audio_path}")

        # Crop/pad to exact length
        target_length = int(seconds_total * sample_rate)
        if source_audio_tensor.shape[-1] > target_length:
            source_audio_tensor = source_audio_tensor[:, :target_length]
        elif source_audio_tensor.shape[-1] < target_length:
            source_audio_tensor = torch.nn.functional.pad(source_audio_tensor, (0, target_length - source_audio_tensor.shape[-1]))

        source_audio_tensor = source_audio_tensor.unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # Encode to get the clean latent representation (x_0)
            init_latents = model.pretransform.encode(source_audio_tensor)
            if isinstance(init_latents, tuple):
                init_latents = init_latents[0]

        sigma_min = kwargs.get("sigma_min", 0.3)
        sigma_max = kwargs.get("sigma_max", 500.0)
        
        # Capture the fully pre-conditioned denoiser from stable-audio-tools
        captured_denoiser = []
        captured_extra_args = []
        
        import stable_audio_tools.inference.generation as sat_gen
        import k_diffusion.sampling as k_samp
        
        original_sample_heun = k_samp.sample_heun
        
        def capture_denoiser(denoiser, x, sigmas, *args, **kwargs_inner):
            captured_denoiser.append(denoiser)
            captured_extra_args.append(kwargs_inner.get('extra_args', {}))
            raise InterruptedError("Captured denoiser")
            
        k_samp.sample_heun = capture_denoiser
        
        try:
            sat_gen.generate_diffusion_cond(
                model, 
                steps=steps,
                cfg_scale=1.0, # Pure inversion, no CFG
                conditioning=conditioning,
                sample_size=source_audio_tensor.shape[-1],
                sigma_min=sigma_min,
                sigma_max=sigma_max,
                sampler_type="k-heun",
                device=self.device
            )
        except InterruptedError:
            pass
        finally:
            k_samp.sample_heun = original_sample_heun
            
        if not captured_denoiser:
            raise RuntimeError("Failed to capture denoiser from generation pipeline.")
            
        denoiser = captured_denoiser[0]
        extra_args = captured_extra_args[0]

        # Build Continuous Sigmas Grid for Inversion (min to max)
        sigmas = get_sigmas_karras(steps + 1, sigma_min, sigma_max, device=self.device)[:-1].flip(0)

        stop_step = int(steps * inversion_strength)
        
        # Start at sigma_min by injecting the true noise floor
        torch.manual_seed(kwargs.get("seed", 0))
        current_latents = init_latents + torch.randn_like(init_latents) * sigma_min

        # Heun ODE Inversion Step Loop
        for i in tqdm(range(stop_step), desc="Heun ODE Inversion"):
            sigma_curr = sigmas[i]
            sigma_next = sigmas[i + 1]
            d_sigma = sigma_next - sigma_curr
            
            sigma_mid = (sigma_curr * sigma_next).sqrt()
            t_curr = torch.full((current_latents.shape[0],), sigma_mid.item(), device=self.device)
            t_next = torch.full((current_latents.shape[0],), sigma_next.item(), device=self.device)
            
            with torch.no_grad():
                # 1. Calculate velocity at current position
                denoised_curr = denoiser(current_latents, t_curr, **extra_args)
                v_curr = (current_latents - denoised_curr) / sigma_curr
                
                # 2. Predict the next position tentatively
                latents_next_pred = current_latents + v_curr * d_sigma
                
                # 3. Calculate velocity at the predicted next position
                denoised_next = denoiser(latents_next_pred, t_next, **extra_args)
                v_next = (latents_next_pred - denoised_next) / sigma_next
                
                # 4. Correct the trajectory by averaging the two velocities together
                v_corrected = 0.5 * (v_curr + v_next)
                
                # 5. Take the actual step forward into noise space
                current_latents = current_latents + v_corrected * d_sigma

        # Variance normalization to prepare latent for unit-variance native samplers
        original_mean = current_latents.mean().item()
        original_std = current_latents.std().item()
        
        latent_std = original_std
        if latent_std > 0:
            latent_tensor = (current_latents - original_mean) / latent_std
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

        # Inject execution config data explicitly for backend self-documentation
        context["inversion_metadata"] = {
            "algorithm": "heun_ode_inversion",
            "sampler": "heun",
            "inversion_steps": steps,
            "stop_step_index": stop_step,
            "inversion_strength": inversion_strength,
            "sigma_min": sigma_min,
            "sigma_max": sigma_max,
            "original_mean": original_mean,
            "original_std": original_std
        }

        artifact = Latent(
            id=content_uid,
            name=generate_slug(2),
            file=Asset(path=None, uid=content_uid, extension=".pt"),
            context=context
        )
        
        return artifact, latent_tensor