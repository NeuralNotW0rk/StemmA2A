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


def _get_cache_file_path() -> Path:
    is_container = os.environ.get("RUNNING_IN_CONTAINER") == "true"
    if is_container:
        data_path = os.environ.get("CONTAINER_DATA_PATH")
    else:
        data_path = os.environ.get("LOCAL_DATA_PATH")
        
    if data_path:
        cache_dir = Path(data_path).expanduser()
        try:
            cache_dir.mkdir(parents=True, exist_ok=True)
            return cache_dir / "model_uid_cache.json"
        except Exception:
            pass
            
    for p in ["/app/data", "./data", "."]:
        path = Path(p)
        if path.exists() and os.access(path, os.W_OK):
            return path / "model_uid_cache.json"
            
    return Path("model_uid_cache.json")

def _get_cached_uid(file_path: str, uid_generator) -> str:
    abs_path = os.path.abspath(file_path)
    if not os.path.exists(abs_path):
        raise FileNotFoundError(f"Checkpoint file not found: {abs_path}")
        
    stat = os.stat(abs_path)
    file_size = stat.st_size
    file_mtime = stat.st_mtime
    
    cache_file = _get_cache_file_path()
    cache = {}
    if cache_file.exists():
        try:
            with open(cache_file, "r") as f:
                cache = json.load(f)
        except Exception as e:
            print(f"Error reading UID cache: {e}")
            
    entry = cache.get(abs_path)
    if entry and entry.get("size") == file_size and entry.get("mtime") == file_mtime:
        return entry.get("uid")
        
    # Cache miss: compute uid by loading state dict
    print(f"Cache miss for {abs_path}. Computing state dict UID...")
    state_dict = load_ckpt_state_dict(abs_path)
    uid = uid_generator.from_state_dict(state_dict)
    
    # Save to cache
    cache[abs_path] = {
        "size": file_size,
        "mtime": file_mtime,
        "uid": uid
    }
    
    try:
        with open(cache_file, "w") as f:
            json.dump(cache, f, indent=2)
    except Exception as e:
        print(f"Error writing UID cache: {e}")
        
    return uid


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

        # Generate the checkpoint ID using the persistent cache or by computing it
        checkpoint_uid = _get_cached_uid(checkpoint_path, self.uid_generator)

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

        if verify:
            checkpoint_uid = _get_cached_uid(info.checkpoint.path, self.uid_generator)
            encoder_asset = getattr(info, "encoder", None)
            if encoder_asset:
                expected_id = self.uid_generator.from_uids([checkpoint_uid, encoder_asset.uid])
            else:
                expected_id = checkpoint_uid
            if expected_id != info.id:
                raise UIDMismatchError("Combined Model UID mismatch")

        # Load the state dict
        state_dict = load_ckpt_state_dict(info.checkpoint.path)

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

        # Determine sampler category (explicit toggle or model config fallback)
        model_config = self.model_info.config.get("model", {})
        diffusion_config = model_config.get("diffusion", {})
        diffusion_objective = diffusion_config.get("diffusion_objective", "v")
        is_rf_denoiser = diffusion_objective == "rf_denoiser"

        sampler_category = kwargs.get("sampler_category")
        if not sampler_category:
            sampler_category = "rectified_flow" if diffusion_objective in ["rectified_flow", "rf_denoiser"] else "k_diffusion"

        sampler_type = None
        if sampler_category == "k_diffusion":
            sampler_type = kwargs.get("k_sampler_type")
            if not sampler_type:
                raise ValueError("k_sampler_type is required for k_diffusion inference.")
        elif sampler_category == "rectified_flow":
            sampler_type = kwargs.get("rf_sampler_type")
            if not sampler_type:
                raise ValueError("rf_sampler_type is required for rectified_flow inference.")
        
        if not sampler_type:
            raise ValueError(f"Unable to determine sampler type for category: {sampler_category}")
            

            
        # Extract metadata and tensor payload from initial latent if available
        init_latent_element = kwargs.get("init_latent_element")
        init_latent_tensor = None
        inversion_meta = None
        
        if init_latent_element:
            print(f"DEBUG: Found init_latent_element with ID {init_latent_element.id}")
            latent_path = Path(init_latent_element.file.path)
            if latent_path and latent_path.exists():
                init_latent_tensor = torch.load(latent_path, map_location=self.device, weights_only=True)
                inversion_meta = getattr(init_latent_element, "context", {}).get("inversion_metadata", None)
                print(f"DEBUG: Loaded init_latent_tensor of shape {init_latent_tensor.shape}")
                print(f"DEBUG: inversion_metadata = {inversion_meta}")
            else:
                print(f"DEBUG: Latent path does not exist: {latent_path}")

        # Establish execution steps and schedule boundaries
        for required_param in ["steps", "cfg_scale", "sigma_min", "sigma_max", "seed"]:
            if required_param not in kwargs:
                raise ValueError(f"Parameter '{required_param}' is required.")

        steps = kwargs["steps"]
        sigma_min = kwargs["sigma_min"]
        sigma_max = kwargs["sigma_max"]

        # Synchronize parameters to the latent node if using an inverted blueprint
        if inversion_meta:
            steps = inversion_meta.get("inversion_steps", steps)
            sigma_min = inversion_meta.get("sigma_min", sigma_min)
            sigma_max = inversion_meta.get("sigma_max", sigma_max)

        # Encode conditioning
        conditioning_tensors = model.conditioner(conditioning, self.device)
        negative_conditioning_tensors = None
        if negative_conditioning is not None:
            negative_conditioning_tensors = model.conditioner(negative_conditioning, self.device)

        # Check model's expected conditioning IDs
        expected_keys = []
        for m in [model, getattr(model, "model", None)]:
            if m is not None:
                for attr in ["local_add_cond_ids", "input_concat_ids", "global_cond_ids", "prepend_cond_ids"]:
                    if hasattr(m, attr):
                        expected_keys.extend(getattr(m, attr))
        expected_keys = list(set(expected_keys))

        batch_size = kwargs.get("batch_size", 1)

        # Replicate duration adaptation logic from generate_diffusion_cond
        mask_padding_attention = getattr(model, 'mask_padding_attention', False)
        use_effective_length_for_schedule = getattr(model, 'use_effective_length_for_schedule', False)
        adapt_duration_to_conditioning = mask_padding_attention or use_effective_length_for_schedule

        audio_sample_size = sample_size
        if adapt_duration_to_conditioning and conditioning is not None:
            max_seconds = 0.0
            for cond_dict in conditioning:
                if "seconds_total" in cond_dict:
                    max_seconds = max(max_seconds, cond_dict["seconds_total"])

            if max_seconds > 0:
                duration_padding_sec = kwargs.get("duration_padding_sec", 6.0)
                target_audio_samples = int((max_seconds + duration_padding_sec) * getattr(model, "sample_rate", sample_rate))

                if model.pretransform is not None:
                    ds_ratio = model.pretransform.downsampling_ratio
                    latent_align = 1
                    if (hasattr(model.pretransform, 'model') and
                            hasattr(model.pretransform.model, 'encoder')):
                        encoder = model.pretransform.model.encoder
                        if hasattr(encoder, 'layers'):
                            first_chunked_index = next(
                                (i for i, l in enumerate(encoder.layers)
                                 if hasattr(l, 'chunk_size') and getattr(l, 'sliding_window_latents', True) is None), None
                            )
                            if first_chunked_index is not None:
                                first_chunked = encoder.layers[first_chunked_index]
                                stride = getattr(first_chunked, 'stride', None)
                                if stride is None and hasattr(encoder, 'strides') and len(encoder.strides) > first_chunked_index:
                                    stride = encoder.strides[first_chunked_index]
                                if stride and stride > 0:
                                    latent_align = max(1, first_chunked.chunk_size // stride)
                    align = ds_ratio * latent_align
                    target_audio_samples = ((target_audio_samples + align - 1) // align) * align
                audio_sample_size = min(target_audio_samples, sample_size)

        latent_sample_size = audio_sample_size
        if model.pretransform is not None:
            latent_sample_size = audio_sample_size // model.pretransform.downsampling_ratio

        # Inject dummy inpaint mask/input if needed
        needs_inpaint_mask = 'inpaint_mask' in expected_keys and 'inpaint_mask' not in conditioning_tensors
        needs_inpaint_input = 'inpaint_masked_input' in expected_keys and 'inpaint_masked_input' not in conditioning_tensors

        if needs_inpaint_mask or needs_inpaint_input:
            mask = torch.zeros((batch_size, 1, latent_sample_size), device=self.device)
            inpaint_input = torch.zeros((batch_size, model.io_channels, latent_sample_size), device=self.device)

            if needs_inpaint_mask:
                conditioning_tensors['inpaint_mask'] = [mask]
                if negative_conditioning_tensors is not None:
                    negative_conditioning_tensors['inpaint_mask'] = [mask]

            if needs_inpaint_input:
                conditioning_tensors['inpaint_masked_input'] = [inpaint_input]
                if negative_conditioning_tensors is not None:
                    negative_conditioning_tensors['inpaint_masked_input'] = [inpaint_input]

        # Set up generation arguments
        args = {
            "steps": steps,
            "cfg_scale": kwargs["cfg_scale"],
            "conditioning": conditioning,
            "negative_conditioning": negative_conditioning,
            "conditioning_tensors": conditioning_tensors,
            "negative_conditioning_tensors": negative_conditioning_tensors,
            "batch_size": batch_size,
            "sample_size": sample_size,
            "sigma_min": sigma_min,
            "sigma_max": sigma_max,
            "sampler_type": sampler_type,
            "device": self.device,
            "seed": kwargs["seed"],
            "apg_scale": kwargs.get("apg_scale", 1.0),
            "cfg_interval": (kwargs.get("cfg_interval_min", 0.0), kwargs.get("cfg_interval_max", 1.0)),
            "scale_phi": kwargs.get("cfg_rescale", 0.0),
            "cfg_norm_threshold": kwargs.get("cfg_norm_threshold", 0.0),
            "duration_padding_sec": kwargs.get("duration_padding_sec", 6.0),
        }

        if init_latent_tensor is not None:
            # Match sample_size exactly to the latent shape to prevent cross-attention and schedule length mismatches
            ds_ratio = model.pretransform.downsampling_ratio if model.pretransform is not None else 1
            args["sample_size"] = init_latent_tensor.shape[-1] * ds_ratio
            args["adapt_duration_to_conditioning"] = False
            args["mask_padding_attention"] = False
            args["use_effective_length_for_schedule"] = False

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
            import stable_audio_tools.inference.sampling as sat_samp
            import k_diffusion.sampling as k_samp
            
            original_sample_k = sat_samp.sample_k
            original_sample_diffusion = sat_samp.sample_diffusion
                
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

            # ALWAYS patch sample_diffusion to conditionally inject latent noise if present (for RF models)
            def patched_sample_diffusion(*inner_args, **inner_kwargs):
                inner_args = list(inner_args)
                inner_kwargs = dict(inner_kwargs)
                
                if init_latent_tensor is not None and inversion_meta is not None:
                    print("DEBUG: patched_sample_diffusion triggered! Injecting latent noise...")
                    # Overwrite noise with the inverted latent
                    custom_noise = init_latent_tensor.clone()
                    
                    # Ensure batch sizes match up
                    target_shape = inner_args[1].shape if len(inner_args) > 1 else inner_kwargs.get("noise").shape
                    if custom_noise.shape[0] == 1 and target_shape[0] > 1:
                        custom_noise = custom_noise.repeat(target_shape[0], 1, 1)
                        
                    if len(inner_args) > 1:
                        inner_args[1] = custom_noise
                    else:
                        inner_kwargs["noise"] = custom_noise
                        
                    inversion_strength = inversion_meta.get("inversion_strength", 1.0)
                    inner_kwargs["init_noise_level"] = inversion_strength
                    inner_kwargs["sigma_max"] = inversion_strength
                    
                    # Clear init_data to prevent any blending (start directly from inverted latent)
                    inner_kwargs["init_data"] = None
                    print(f"DEBUG: Injected custom noise of shape {custom_noise.shape}, sigma_max={inversion_strength}")
                    
                return original_sample_diffusion(*inner_args, **inner_kwargs)

            sat_samp.sample_k = patched_sample_k
            sat_samp.sample_diffusion = patched_sample_diffusion
            sat_gen.sample_diffusion = patched_sample_diffusion
            
            try:
                output = generate_diffusion_cond(model, **args)
            finally:
                # Restore original hooks immediately after execution completes
                sat_samp.sample_k = original_sample_k
                sat_samp.sample_diffusion = original_sample_diffusion
                sat_gen.sample_diffusion = original_sample_diffusion

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

        # Replicate duration adaptation logic from generate_diffusion_cond to match target shapes
        mask_padding_attention = getattr(model, 'mask_padding_attention', False)
        use_effective_length_for_schedule = getattr(model, 'use_effective_length_for_schedule', False)
        adapt_duration_to_conditioning = mask_padding_attention or use_effective_length_for_schedule
        
        trim_duration = max(0.0, seconds_total - seconds_start)
        target_audio_samples = int(trim_duration * sample_rate)
        
        if adapt_duration_to_conditioning:
            duration_padding_sec = kwargs.get("duration_padding_sec", 6.0)
            target_audio_samples = int((trim_duration + duration_padding_sec) * sample_rate)
            
            if model.pretransform is not None:
                ds_ratio = model.pretransform.downsampling_ratio
                latent_align = 1
                if (hasattr(model.pretransform, 'model') and
                        hasattr(model.pretransform.model, 'encoder')):
                    encoder = model.pretransform.model.encoder
                    if hasattr(encoder, 'layers'):
                        first_chunked_index = next(
                            (i for i, l in enumerate(encoder.layers)
                             if hasattr(l, 'chunk_size') and getattr(l, 'sliding_window_latents', True) is None), None
                        )
                        if first_chunked_index is not None:
                            first_chunked = encoder.layers[first_chunked_index]
                            stride = getattr(first_chunked, 'stride', None)
                            if stride is None and hasattr(encoder, 'strides') and len(encoder.strides) > first_chunked_index:
                                stride = encoder.strides[first_chunked_index]
                            if stride and stride > 0:
                                latent_align = max(1, first_chunked.chunk_size // stride)
                align = ds_ratio * latent_align
                target_audio_samples = ((target_audio_samples + align - 1) // align) * align
                
        audio_sample_size = min(target_audio_samples, sample_size)
        
        # Crop/pad to adapted audio sample size
        if source_audio_tensor.shape[-1] > audio_sample_size:
            source_audio_tensor = source_audio_tensor[:, :audio_sample_size]
        elif source_audio_tensor.shape[-1] < audio_sample_size:
            source_audio_tensor = torch.nn.functional.pad(source_audio_tensor, (0, audio_sample_size - source_audio_tensor.shape[-1]))

        source_audio_tensor = source_audio_tensor.unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # Encode to get the clean latent representation (x_0)
            init_latents = model.pretransform.encode(source_audio_tensor)
            if isinstance(init_latents, tuple):
                init_latents = init_latents[0]

        is_rf = (self.model_info.config.get("model", {}).get("diffusion", {}).get("diffusion_objective") in ["rectified_flow", "rf_denoiser"])
        sampler_category = kwargs.get("sampler_category", "rectified_flow" if is_rf else "k_diffusion")

        if sampler_category == "rectified_flow":
            import stable_audio_tools.inference.inversion as sat_inv
            import copy
            
            inversion_gamma = kwargs.get("inversion_gamma", 0.3)
            inversion_unconditional = kwargs.get("inversion_unconditional", True)
            inversion_cfg_scale = kwargs.get("inversion_cfg_scale", 1.0)
            
            inversion_params = {
                "inversion_steps": steps,
                "inversion_gamma": inversion_gamma,
                "inversion_unconditional": inversion_unconditional,
                "inversion_sigma_max": 1.0,
                "inversion_cfg_scale": inversion_cfg_scale,
            }
            
            with torch.no_grad():
                # Modify the conditioning for inversion
                inversion_conditioning = copy.deepcopy(conditioning)
                inversion_conditioning_tensors = model.conditioner(inversion_conditioning, self.device)
                
                # Check model's expected conditioning IDs
                expected_keys = []
                for m in [model, getattr(model, "model", None)]:
                    if m is not None:
                        for attr in ["local_add_cond_ids", "input_concat_ids", "global_cond_ids", "prepend_cond_ids"]:
                            if hasattr(m, attr):
                                expected_keys.extend(getattr(m, attr))
                expected_keys = list(set(expected_keys))
                
                needs_inpaint_mask = 'inpaint_mask' in expected_keys and 'inpaint_mask' not in inversion_conditioning_tensors
                needs_inpaint_input = 'inpaint_masked_input' in expected_keys and 'inpaint_masked_input' not in inversion_conditioning_tensors
                
                if needs_inpaint_mask or needs_inpaint_input:
                    b_size = init_latents.shape[0]
                    l_size = init_latents.shape[-1]
                    mask = torch.zeros((b_size, 1, l_size), device=self.device)
                    inpaint_input = torch.zeros((b_size, model.io_channels, l_size), device=self.device)
                    
                    if needs_inpaint_mask:
                        inversion_conditioning_tensors['inpaint_mask'] = [mask]
                    if needs_inpaint_input:
                        inversion_conditioning_tensors['inpaint_masked_input'] = [inpaint_input]

                inversion_conditioning_inputs = model.get_conditioning_inputs(inversion_conditioning_tensors)
                
                model_dtype = next(model.model.parameters()).dtype
                inversion_conditioning_inputs = {
                    k: v.type(model_dtype) if v is not None and hasattr(v, "type") else v
                    for k, v in inversion_conditioning_inputs.items()
                }
                
                if inversion_unconditional:
                    cfg_dropout_prob = 1.0
                else:
                    cfg_dropout_prob = 0.0
                    for x in inversion_conditioning:
                        if "prompt" in x:
                            x["prompt"] = ""
                    inversion_conditioning_tensors = model.conditioner(inversion_conditioning, self.device)
                    
                    if needs_inpaint_mask:
                        inversion_conditioning_tensors['inpaint_mask'] = [mask]
                    if needs_inpaint_input:
                        inversion_conditioning_tensors['inpaint_masked_input'] = [inpaint_input]

                    inversion_conditioning_inputs = model.get_conditioning_inputs(inversion_conditioning_tensors)
                    inversion_conditioning_inputs = {
                        k: v.type(model_dtype) if v is not None and hasattr(v, "type") else v
                        for k, v in inversion_conditioning_inputs.items()
                    }
                
                torch.manual_seed(kwargs.get("seed", 0))
                inversion_noise = torch.randn_like(init_latents)
                
                # Perform native RF Inversion using invert_audio
                current_latents = sat_inv.invert_audio(
                    model=model.model,
                    waveform_latent=init_latents,
                    noise=inversion_noise,
                    inversion_params=inversion_params,
                    device=self.device,
                    cfg_dropout_prob=cfg_dropout_prob,
                    **inversion_conditioning_inputs
                )
                
            latent_tensor = current_latents
            
            # Setup metadata properties
            algorithm = "native_rf_inversion"
            sampler = "invert_audio"
            stop_step = steps
            sigma_min = 0.0
            sigma_max = 1.0
            
        else:
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
                
            algorithm = "euler_ode_inversion"
            sampler = "euler"

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
            "algorithm": algorithm,
            "sampler": sampler,
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