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

def load_ckpt_state_dict(ckpt_path):
    with open(ckpt_path, "rb") as f:
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

        with open(config_path, 'r') as cf:
            config_json = cf.read()
            config = json.loads(config_json)

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
        if self.model and self.model_info.id == info.id:
            return 

        if self.model:
            del self.model
            self.model = None

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
            sampler_type = "k-heun"  # Bypass stable-audio-tools sampler string validation
            
        # Extract metadata from initial latent if available
        init_latent_element = kwargs.get("init_latent_element")
        init_latent_tensor = None
        inversion_meta = None
        
        if init_latent_element:
            latent_path = Path(init_latent_element.file.path)
            if latent_path and latent_path.exists():
                init_latent_tensor = torch.load(latent_path, map_location=self.device, weights_only=True)
                inversion_meta = getattr(init_latent_element, "context", {}).get("inversion_metadata", None)

        # Base generation parameters
        steps = kwargs.get("steps", 8)
        sigma_min = kwargs.get("sigma_min", 0.3)
        sigma_max = kwargs.get("sigma_max", 500.0)

        # If a valid inverted latent map is passed, synchronize the generation steps to it
        if inversion_meta:
            steps = inversion_meta.get("inversion_steps", steps)
            sigma_min = inversion_meta.get("sigma_min", sigma_min)
            sigma_max = inversion_meta.get("sigma_max", sigma_max)

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
                
            # FIX: Patch sample_k to conditionally inject our custom inverted latent 
            # AND cleanly slice the continuous sigma tracking vector
            def patched_sample_k(*inner_args, **inner_kwargs):
                inner_args = list(inner_args)
                
                # If we have an inverted latent blueprint, intercept the execution grid
                if init_latent_tensor is not None and inversion_meta is not None:
                    # 1. Grab the current sampler function being called (e.g., sample_heun or sample_euler)
                    sampler_fn = inner_args[2] if len(inner_args) > 2 else inner_kwargs.get("sampler_fn")
                    
                    # 2. Re-create the exact global karras sigmas array
                    global_sigmas = get_sigmas_karras(steps + 1, sigma_min, sigma_max, device=self.device)
                    
                    # 3. CRITICAL: Slice the sigmas array so the sampler starts mid-way through denoising.
                    # Inversion climbs the ladder from index 0 to stop_step.
                    # The sampler needs to step down the ladder starting at the top of that noise height.
                    # We reverse the index order to read downstream towards clean data.
                    stop_index = inversion_meta["stop_step_index"]
                    generation_sigmas = global_sigmas[:-1].flip(0)[stop_index:]
                    
                    # Add back the final 0.0 sigma term that k-diffusion requires to terminate loops safely
                    generation_sigmas = torch.cat([generation_sigmas, global_sigmas[-1:]])

                    # 4. Wrap the underlying execution function to inject our custom latent coordinates
                    # rather than letting stable_audio_tools overwrite it with random noise
                    def custom_sampler_wrapper(model_wrap, x, sigmas_arg, *s_args, **s_kwargs):
                        # Ensure batch sizes match up
                        custom_coords = init_latent_tensor.clone()
                        if custom_coords.shape[0] == 1 and x.shape[0] > 1:
                            custom_coords = custom_coords.repeat(x.shape[0], 1, 1)
                        
                        # Use our custom sliced timeline sigmas instead of the full global timeline
                        return sampler_fn(model_wrap, custom_coords, generation_sigmas, *s_args, **s_kwargs)

                    # Swap the native sampler function argument out for our custom sliced wrapper
                    if len(inner_args) > 2:
                        inner_args[2] = custom_sampler_wrapper
                    else:
                        inner_kwargs["sampler_fn"] = custom_sampler_wrapper

                return original_sample_k(*inner_args, **inner_kwargs)

            sat_gen.sample_k = patched_sample_k
            
            try:
                output = generate_diffusion_cond(model, **args)
            finally:
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

        target_length = int(seconds_total * sample_rate)
        if source_audio_tensor.shape[-1] > target_length:
            source_audio_tensor = source_audio_tensor[:, :target_length]
        elif source_audio_tensor.shape[-1] < target_length:
            source_audio_tensor = torch.nn.functional.pad(source_audio_tensor, (0, target_length - source_audio_tensor.shape[-1]))

        source_audio_tensor = source_audio_tensor.unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            init_latents = model.pretransform.encode(source_audio_tensor)
            if isinstance(init_latents, tuple):
                init_latents = init_latents[0]

        sigma_min = kwargs.get("sigma_min", 0.3)
        sigma_max = kwargs.get("sigma_max", 500.0)
        
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
                cfg_scale=1.0, 
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

        sigmas = get_sigmas_karras(steps + 1, sigma_min, sigma_max, device=self.device)[:-1].flip(0)
        stop_step = int(steps * inversion_strength)
        
        torch.manual_seed(kwargs.get("seed", 0))
        current_latents = init_latents + torch.randn_like(init_latents) * sigma_min

        for i in tqdm(range(stop_step), desc="Euler ODE Inversion"):
            sigma_curr = sigmas[i]
            sigma_next = sigmas[i + 1]
            
            t_tensor = torch.full((current_latents.shape[0],), sigma_curr.item(), device=self.device)
            
            with torch.no_grad():
                denoised_pred = denoiser(current_latents, t_tensor, **extra_args)
                
            v_velocity = (current_latents - denoised_pred) / sigma_curr
            d_sigma = sigma_next - sigma_curr
            current_latents = current_latents + v_velocity * d_sigma

        # Variance normalization to prepare latent for unit-variance native samplers
        latent_std = current_latents.std().item()
        if latent_std > 0:
            latent_tensor = (current_latents - current_latents.mean().item()) / latent_std
        else:
            latent_tensor = current_latents

        content_uid = self.uid_generator.from_tensor(latent_tensor)

        context = {}
        for k, v in kwargs.items():
            if k.endswith('_element'):
                context[k.replace('_element', '_id')] = v.id
            elif k.endswith('_elements'):
                context[k.replace('_elements', '_ids')] = [el.id for el in v]
            else:
                context[k] = v

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