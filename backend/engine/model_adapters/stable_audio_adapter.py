import json
import struct
import copy
import os
import xxhash
from pathlib import Path

def _hash_directory(directory_path: Path) -> str:
    h = xxhash.xxh3_64()
    # Find all files recursively and sort them to be deterministic
    for file_path in sorted(directory_path.glob("**/*")):
        if file_path.is_file():
            # Update hash with relative path to ensure structure is identical
            h.update(str(file_path.relative_to(directory_path)).encode("utf-8"))
            # Update hash with file content in chunks
            with open(file_path, "rb") as f:
                while chunk := f.read(8192):
                    h.update(chunk)
    return f"{h.hexdigest()}.xxh3_64"

import torch
import torchaudio
from einops import rearrange
from stable_audio_tools import create_model_from_config
from stable_audio_tools.inference.generation import generate_diffusion_cond
from safetensors.torch import load_file
from coolname import generate_slug
from tqdm import tqdm
from k_diffusion.sampling import get_sigmas_karras

from .base_adapter import ModelAdapter, operation
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
        encoder_path = kwargs.get("encoder_path")

        # Load the config
        with open(config_path, 'r') as cf:
            config_json = cf.read()
            config = json.loads(config_json)

        # Load the state dict and generate the checkpoint ID directly from it
        state_dict = load_ckpt_state_dict(checkpoint_path)
        checkpoint_uid = self.uid_generator.from_state_dict(state_dict)

        encoder_asset = None
        if encoder_path:
            encoder_dir = Path(encoder_path)
            if encoder_dir.is_dir():
                encoder_uid = _hash_directory(encoder_dir)
                encoder_asset = Asset(path=str(encoder_dir).replace("\\", "/"), uid=encoder_uid)
                # Combine checkpoint UID and encoder UID to form the overall Model ID
                model_id = self.uid_generator.from_uids([checkpoint_uid, encoder_uid])
            else:
                model_id = checkpoint_uid
        else:
            model_id = checkpoint_uid

        return StableAudioModel(
            id=model_id,
            name=kwargs.get("name"),
            checkpoint=Asset(path=checkpoint_path, uid=checkpoint_uid),
            config=config,
            model_type=kwargs.get("model_type"),
            encoder=encoder_asset,
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

        # Load the state dict
        state_dict = load_ckpt_state_dict(info.checkpoint.path)

        if verify:
            checkpoint_uid = self.uid_generator.from_state_dict(state_dict)
            encoder_asset = getattr(info, "encoder", None)
            if encoder_asset:
                expected_id = self.uid_generator.from_uids([checkpoint_uid, encoder_asset.uid])
            else:
                expected_id = checkpoint_uid
            if expected_id != info.id:
                raise UIDMismatchError("Combined Model UID mismatch")

        # Load the new model
        # Override conditioner paths if local encoder path is supplied and exists
        config = copy.deepcopy(info.config)
        encoder_asset = getattr(info, "encoder", None)
        if encoder_asset and encoder_asset.path:
            actual_path = encoder_asset.path
            # If the path points to an archive file (uploaded via sync), extract it first
            if os.path.isfile(actual_path):
                extracted_dir = Path(actual_path + "_extracted")
                if not extracted_dir.exists():
                    import zipfile
                    print(f"Extracting encoder archive {actual_path} to {extracted_dir}...")
                    extracted_dir.mkdir(parents=True, exist_ok=True)
                    with zipfile.ZipFile(actual_path, 'r') as zip_ref:
                        zip_ref.extractall(extracted_dir)
                actual_path = str(extracted_dir)

            if os.path.exists(actual_path):
                clean_encoder_path = str(Path(actual_path)).replace("\\", "/")
                conditioning = config.get("model", {}).get("conditioning", {})
                for cond_config in conditioning.get("configs", []):
                    cond_type = cond_config.get("type")
                    if cond_type in ["t5", "t5gemma", "clap", "clap_audio"]:
                        cond_inner_config = cond_config.setdefault("config", {})
                        cond_inner_config["model_path"] = clean_encoder_path
                        cond_inner_config.pop("repo_id", None)
                        cond_inner_config.pop("subfolder", None)

        self.model = create_model_from_config(config)
        self.model.load_state_dict(state_dict)
        self.model_info = info

    def cleanup(self):
        if self.model:
            del self.model
            self.model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    @operation(
        name="generate",
        is_standard=True,
        description="Generate audio from a generative model",
        initiator_types=["model", "grating", "latent", "audio"],
        context_overrides={
            "audio": {
                "name": "audio-to-audio",
                "description": "Perform audio-guided generation (audio-to-audio) using this node as initialization"
            },
            "grating": {
                "name": "generate from grating",
                "description": "Generate audio guided by the selected grating structure"
            },
            "latent": {
                "name": "generate from latent",
                "description": "Generate audio guided by the selected latent structure"
            },
            "model": {
                "name": "Generate Audio",
                "description": "Generate audio from the selected generative model"
            }
        }
    )
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
                    
                    # Overwrite the random seed noise immediately at the entrance of the wrapper
                    if len(inner_args) > 1:
                        inner_args[1] = custom_coords
                    else:
                        inner_kwargs["noise"] = custom_coords

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
                            # Pass through x, which has now been safely overridden with your custom_coords 
                            # at step 1, ensuring its internal variance profiles line up with generation_sigmas
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

            # Trim silence to the remaining segment length
            trim_duration = max(0.0, seconds_total - seconds_start)
            output = output[:,:,:int(trim_duration*sample_rate)]

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
            duration=float(output.shape[-1]) / sample_rate,
            context=context
        )

        return artifact, output

    @operation(
        name="invert",
        is_standard=True,
        description="Invert audio to a latent representation",
        initiator_types=["audio"]
    )
    def invert(self, **kwargs) -> tuple[Latent, torch.Tensor]:
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

        # Crop/pad to exact segment length (total track minus start offset)
        trim_duration = max(0.0, seconds_total - seconds_start)
        target_length = int(trim_duration * sample_rate)
        if source_audio_tensor.shape[-1] > target_length:
            source_audio_tensor = source_audio_tensor[:, :target_length]
        elif source_audio_tensor.shape[-1] < target_length:
            source_audio_tensor = torch.nn.functional.pad(source_audio_tensor, (0, target_length - source_audio_tensor.shape[-1]))

        # Pad up to the model's native sample_size
        if source_audio_tensor.shape[-1] < sample_size:
            source_audio_tensor = torch.nn.functional.pad(source_audio_tensor, (0, sample_size - source_audio_tensor.shape[-1]))

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
                sample_size=sample_size,
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

        # Euler ODE Inversion Step Loop with Heun Correction
        for i in tqdm(range(stop_step), desc="Heun ODE Inversion"):
            sigma_curr = sigmas[i]
            sigma_next = sigmas[i + 1]
            d_sigma = sigma_next - sigma_curr
            
            t_curr = torch.full((current_latents.shape[0],), sigma_curr.item(), device=self.device)
            t_next = torch.full((current_latents.shape[0],), sigma_next.item(), device=self.device)
            
            with torch.no_grad():
                # 1. Calculate velocity at current position (Standard Euler Step)
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

        # Explicit unit-variance scaling to normalize data before handing to generation samplers
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

        # Inject execution config data explicitly for backend self-documentation
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