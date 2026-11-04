import json
import struct
import copy
import os
import xxhash
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

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
from param_graph.elements.base_elements import Asset, GraphElement
from param_graph.elements.artifacts.audio_element import Audio
from param_graph.elements.artifacts.latent_element import Latent
from param_graph.elements.models.stable_audio_element import StableAudioModel
from utils.uid import UIDMismatchError
from utils.filesystem import get_path_size
from utils.hf_hub import load_cached_first

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


@contextmanager
def _skip_weight_init() -> Iterator[None]:
    """
    Turns torch.nn.init's in-place initialisers into no-ops while a model is built.

    The checkpoint overwrites every persistent weight straight after, so randomly initialising
    a multi-billion-parameter model first is wasted CPU time, and leaving the storage untouched
    means the OS never commits it. This is the trick transformers' from_pretrained uses (and it
    nests with it, so conditioners loaded with from_pretrained still get their own weights).
    Non-persistent buffers (windows, FSQ levels) are computed directly rather than through
    torch.nn.init, so they are unaffected.
    """
    originals = {
        name: fn for name, fn in vars(torch.nn.init).items()
        if callable(fn) and name.endswith("_") and not name.startswith("_")
    }

    def no_init(tensor: torch.Tensor, *args: object, **kwargs: object) -> torch.Tensor:
        return tensor

    for name in originals:
        setattr(torch.nn.init, name, no_init)
    try:
        yield
    finally:
        for name, fn in originals.items():
            setattr(torch.nn.init, name, fn)


def _adopt_state_dict(model: torch.nn.Module, state_dict: dict[str, torch.Tensor]) -> None:
    """
    Loads state_dict into model by taking over its tensors rather than copying them, so the
    weights are held in memory once instead of twice. Tensors are cast to the dtype the model
    was built with first, keeping the model's precision what a copying load would give.
    """
    expected = model.state_dict()
    for key, tensor in state_dict.items():
        target = expected.get(key)
        if target is not None and tensor.dtype != target.dtype:
            state_dict[key] = tensor.to(target.dtype)
    model.load_state_dict(state_dict, assign=True)


# k-diffusion samplers that accept an explicit sigma schedule, by stable-audio-tools sampler_type
_K_SIGMA_SAMPLERS = {
    "k-heun": "sample_heun",
    "k-lms": "sample_lms",
    "k-dpmpp-2s-ancestral": "sample_dpmpp_2s_ancestral",
    "k-dpm-2": "sample_dpm_2",
    "dpmpp-2m": "sample_dpmpp_2m",
    "dpmpp-2m-sde": "sample_dpmpp_2m_sde",
    "dpmpp-3m-sde": "sample_dpmpp_3m_sde",
}


def _inverted_latent_sigmas(
    steps: int, stop_step: int, sigma_min: float, sigma_max: float, device: torch.device | str
) -> torch.Tensor:
    """
    The descending sigma schedule that retraces a k-diffusion ODE inversion back to clean data.

    Inversion walks the ascending Karras grid get_sigmas_karras(steps + 1)[:-1].flip(0) from index 0
    (sigma_min) up to index stop_step. Index k of that ascending grid is index steps - k of the
    descending one, so generation starts at descending[steps - stop_step] and runs down to sigma_min,
    then takes the final step to 0.
    """
    if not 0 <= stop_step <= steps:
        raise ValueError(f"Inversion stop_step_index {stop_step} is outside 0..{steps}.")
    descending = get_sigmas_karras(steps + 1, sigma_min, sigma_max, device=device)
    return descending[steps - stop_step:]


def _restore_inverted_latent_scale(
    latent: torch.Tensor, inversion_meta: dict, start_sigma: float
) -> torch.Tensor:
    """
    Undoes the unit-variance normalisation invert() applies to k-diffusion latents, so the latent
    sits at the noise level its schedule starts from.

    Latents inverted before the stats were recorded can only be restored at full strength: there the
    latent is dominated by sigma_max noise, so scaling by sigma_max is what sample_k always did.
    """
    if "latent_std" in inversion_meta and "latent_mean" in inversion_meta:
        return latent * inversion_meta["latent_std"] + inversion_meta["latent_mean"]
    if inversion_meta["stop_step_index"] == inversion_meta["inversion_steps"]:
        return latent * start_sigma
    raise ValueError(
        "This latent was inverted at partial strength before its normalisation stats were recorded, "
        "so its noise scale can't be restored. Re-invert the source audio."
    )


def _compute_stable_audio_uid(file_path: str, uid_generator) -> str:
    abs_path = os.path.abspath(file_path)
    if not os.path.exists(abs_path):
        raise FileNotFoundError(f"Checkpoint file not found: {abs_path}")
    print(f"Computing state dict UID for {abs_path}...")
    state_dict = load_ckpt_state_dict(abs_path)
    return uid_generator.from_state_dict(state_dict)


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
        config = kwargs.get("config")

        if config is None:
            if not config_path:
                raise ValueError("Either 'config' dictionary or 'config_path' must be provided.")
            # Load the config
            with open(config_path, 'r') as cf:
                config_json = cf.read()
                config = json.loads(config_json)

        checkpoint_uid = kwargs.get("checkpoint_uid")
        checkpoint_size = kwargs.get("checkpoint_size")
        encoder_uid = kwargs.get("encoder_uid")
        encoder_size = kwargs.get("encoder_size")

        # Generate the checkpoint ID using the persistent cache or by computing it
        if not checkpoint_uid:
            checkpoint_uid = _compute_stable_audio_uid(checkpoint_path, self.uid_generator)
        if not checkpoint_size:
            checkpoint_size = get_path_size(Path(checkpoint_path))
        checkpoint_asset = Asset(
            path=checkpoint_path,
            uid=checkpoint_uid,
            size=checkpoint_size
        )

        encoder_asset = None
        if encoder_path or encoder_uid:
            if encoder_path:
                encoder_path_obj = Path(encoder_path)
                if encoder_path_obj.is_dir():
                    if not encoder_uid:
                        encoder_uid = self.uid_generator.from_directory(encoder_path_obj)
                    if not encoder_size:
                        encoder_size = get_path_size(encoder_path_obj)
                    encoder_asset = Asset(
                        path=str(encoder_path_obj).replace("\\", "/"),
                        uid=encoder_uid,
                        size=encoder_size
                    )
                elif encoder_path_obj.is_file():
                    if not encoder_uid:
                        encoder_uid = self.uid_generator.from_string(encoder_path)
                    if not encoder_size:
                        encoder_size = os.path.getsize(encoder_path_obj)
                    encoder_asset = Asset(
                        path=str(encoder_path_obj).replace("\\", "/"),
                        uid=encoder_uid,
                        size=encoder_size
                    )
            elif encoder_uid:
                encoder_asset = Asset(
                    path="",
                    uid=encoder_uid,
                    size=encoder_size
                )

        if encoder_asset and encoder_asset.uid:
            # Combine checkpoint UID and encoder UID to form the overall Model ID
            model_id = self.uid_generator.from_uids([checkpoint_uid, encoder_asset.uid])
        else:
            model_id = checkpoint_uid

        return StableAudioModel(
            id=model_id,
            name=kwargs.get("name"),
            checkpoint=checkpoint_asset,
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

        # A directory encoder is stored as the directory itself, wherever it lives
        encoder_asset = getattr(info, "encoder", None)
        actual_encoder_path: Path | None = None
        if encoder_asset and encoder_asset.path and os.path.exists(encoder_asset.path):
            actual_encoder_path = Path(encoder_asset.path)

        if verify:
            # 1. Verify Checkpoint Asset
            ckpt_path = info.checkpoint.path
            if ckpt_path and os.path.exists(ckpt_path) and os.path.isfile(ckpt_path):
                ckpt_size = get_path_size(Path(ckpt_path))
                if info.checkpoint.size == ckpt_size:
                    checkpoint_uid = info.checkpoint.uid
                else:
                    print(f"Size mismatch or missing for checkpoint {ckpt_path}. Hashing weights...")
                    checkpoint_uid = _compute_stable_audio_uid(ckpt_path, self.uid_generator)
                
                if checkpoint_uid != info.checkpoint.uid:
                    raise UIDMismatchError("Model checkpoint UID mismatch")
            else:
                checkpoint_uid = info.checkpoint.uid

            # 2. Verify Encoder Asset (if present)
            if encoder_asset:
                if actual_encoder_path and actual_encoder_path.exists():
                    enc_size = get_path_size(actual_encoder_path)
                    if encoder_asset.size == enc_size:
                        encoder_uid = encoder_asset.uid
                    else:
                        print(f"Size mismatch or missing for encoder {actual_encoder_path}. Hashing directory...")
                        encoder_uid = self.uid_generator.from_directory(actual_encoder_path)
                    
                    if encoder_uid != encoder_asset.uid:
                        raise UIDMismatchError("Model encoder UID mismatch")
                else:
                    encoder_uid = encoder_asset.uid
                
                expected_id = self.uid_generator.from_uids([checkpoint_uid, encoder_uid])
            else:
                expected_id = checkpoint_uid

            if expected_id != info.id:
                raise UIDMismatchError("Combined Model UID mismatch")

        # Load the new model
        # Override conditioner paths if local encoder path is supplied and exists
        config = copy.deepcopy(info.config)
        if actual_encoder_path and actual_encoder_path.exists():
            clean_encoder_path = str(actual_encoder_path).replace("\\", "/")
            conditioning = config.get("model", {}).get("conditioning", {})
            for cond_config in conditioning.get("configs", []):
                cond_type = cond_config.get("type")
                if cond_type in ["t5", "t5gemma", "clap", "clap_audio"]:
                    cond_inner_config = cond_config.setdefault("config", {})
                    cond_inner_config["model_path"] = clean_encoder_path
                    cond_inner_config.pop("repo_id", None)
                    cond_inner_config.pop("subfolder", None)

        # Conditioners (e.g. T5) fetch tokenizers/weights from the Hub by name on first use
        with _skip_weight_init():
            model = load_cached_first(lambda **_: create_model_from_config(config), "Stable Audio conditioners")
        # Read after building, so a conditioner download doesn't hold the weights in memory meanwhile
        _adopt_state_dict(model, load_ckpt_state_dict(info.checkpoint.path))
        self.model = model
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
        initiator_types=["model", "grating", "latent", "audio", "individual"],
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
            "individual": {
                "name": "generate exemplar",
                "description": "Generate an audio exemplar expressing the selected individual's genome"
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
        is_rf_objective = diffusion_objective in ["rectified_flow", "rf_denoiser"]

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
            # How the latent was inverted (conditioned or not) only shaped the inversion. Generation is
            # always conditioned on this prompt: CFG dropout here would zero it (and the timing) at cfg 1.

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
            import math
            
            original_sample_k = sat_samp.sample_k
            original_sample_diffusion = sat_samp.sample_diffusion
            original_build_schedule = sat_samp.build_schedule
                
            # Patch sample_k (v models) to sample from the inverted latent along the inversion's own
            # Karras schedule. sample_k builds its own polyexponential schedule and scales the noise by
            # its first sigma, so for an inverted latent the k-diffusion sampler is called directly.
            # Mirrors sample_k's signature in stable-audio-tools 0.0.20 (sample_diffusion passes these by
            # keyword); the inversion's own schedule is captured separately so the names don't collide.
            inversion_schedule = (steps, sigma_min, sigma_max)

            def patched_sample_k(
                model_fn, noise, init_data=None, steps=100, sampler_type="dpmpp-2m-sde",
                sigma_min=0.01, sigma_max=100, rho=1.0, device="cuda", callback=None,
                cond_fn=None, **extra_args
            ):
                if is_rf_objective or init_latent_tensor is None or inversion_meta is None:
                    return original_sample_k(
                        model_fn, noise, init_data=init_data, steps=steps, sampler_type=sampler_type,
                        sigma_min=sigma_min, sigma_max=sigma_max, rho=rho, device=device,
                        callback=callback, cond_fn=cond_fn, **extra_args
                    )

                inv_steps, inv_sigma_min, inv_sigma_max = inversion_schedule

                sampler_name = _K_SIGMA_SAMPLERS.get(sampler_type)
                if sampler_name is None:
                    raise ValueError(
                        f"Sampler '{sampler_type}' can't follow an inverted latent's schedule. "
                        f"Use one of: {', '.join(_K_SIGMA_SAMPLERS)}."
                    )

                generation_sigmas = _inverted_latent_sigmas(
                    inv_steps, inversion_meta["stop_step_index"], inv_sigma_min, inv_sigma_max, noise.device
                )
                x = _restore_inverted_latent_scale(
                    init_latent_tensor.to(noise.device), inversion_meta, generation_sigmas[0].item()
                )
                if x.shape[0] == 1 and noise.shape[0] > 1:
                    x = x.repeat(noise.shape[0], 1, 1)
                x = x.to(noise.dtype)

                denoiser = sat_samp.K.external.VDenoiser(model_fn)
                if cond_fn is not None:
                    denoiser = sat_samp.make_cond_model_fn(denoiser, cond_fn)
                sampler = getattr(sat_samp.K.sampling, sampler_name)
                print(f"DEBUG: Sampling inverted latent from sigma {generation_sigmas[0].item():.4f} "
                      f"over {len(generation_sigmas) - 1} steps")
                return sampler(denoiser, x, generation_sigmas, disable=False, callback=callback, extra_args=extra_args)

            # Patch sample_diffusion to inject the inverted latent and RF noise level (RF models only).
            # stable-audio-tools routes every objective through sample_diffusion, so without the
            # objective check this would also clamp a v-model's sigma_max (e.g. 500) to the RF range.
            def patched_sample_diffusion(*inner_args, **inner_kwargs):
                inner_args = list(inner_args)
                inner_kwargs = dict(inner_kwargs)

                if is_rf_objective and init_latent_tensor is not None and inversion_meta is not None:
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

            # Patch build_schedule to return the exact flipped schedule used during inversion (RF models only;
            # v-ddim also calls build_schedule, and an RF logSNR schedule is wrong for it)
            def patched_build_schedule(*inner_args, **inner_kwargs):
                if is_rf_objective and init_latent_tensor is not None and inversion_meta is not None:
                    print("DEBUG: patched_build_schedule triggered!")
                    
                    # Extract steps (can be positional at index 0, or keyword 'steps')
                    sched_steps = inner_args[0] if len(inner_args) > 0 else inner_kwargs.get("steps", steps)
                    
                    inversion_strength = inversion_meta.get("inversion_strength", 1.0)
                    inversion_steps = inversion_meta.get("inversion_steps", sched_steps)
                    
                    logsnr_max = math.log(((1 - inversion_strength) / inversion_strength) + 1e-6) if inversion_strength < 1.0 else -6.0
                    logsnr = torch.linspace(logsnr_max, 2.0, inversion_steps + 1, device=self.device)
                    t_inv = torch.sigmoid(-logsnr)
                    t_inv = t_inv.flip(0)
                    t_inv[0] = 0.0
                    t_inv[-1] = 1.0
                    
                    generation_sigmas = t_inv.flip(0)
                    print(f"DEBUG: Overrode sigmas schedule to start at {generation_sigmas[0].item()} and end at {generation_sigmas[-1].item()}")
                    return generation_sigmas
                    
                return original_build_schedule(*inner_args, **inner_kwargs)

            sat_samp.sample_k = patched_sample_k
            sat_samp.sample_diffusion = patched_sample_diffusion
            sat_gen.sample_diffusion = patched_sample_diffusion
            sat_samp.build_schedule = patched_build_schedule
            
            try:
                output = generate_diffusion_cond(model, **args)
            finally:
                # Restore original hooks immediately after execution completes
                sat_samp.sample_k = original_sample_k
                sat_samp.sample_diffusion = original_sample_diffusion
                sat_gen.sample_diffusion = original_sample_diffusion
                sat_samp.build_schedule = original_build_schedule

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
            if k in ("job_id", "execution_mode"):
                continue
            if k.endswith('_element'):
                context[k.replace('_element', '_id')] = v.id
            elif k.endswith('_elements'):
                context[k.replace('_elements', '_ids')] = [el.id for el in v]
            elif isinstance(v, GraphElement):
                # Other element arguments (e.g. baseline_grating) are referenced by ID, not serialized whole
                context[f"{k}_id"] = v.id
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
        description="Invert an audio or image artifact into the selected model's latent space",
        initiator_types=["model", "audio"],
        context_overrides={
            "audio": {
                "name": "Invert Audio",
                "description": "Invert the selected audio into a latent with a Stable Audio model"
            }
        }
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

        # Normalisation stats for k-diffusion latents; the RF path stores its latent unnormalised
        latent_stats: dict[str, float] = {}

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
                
                # Unconditional zeroes the prompt and timing embeddings (the null conditioning CFG
                # contrasts against), which differs from conditioning on an empty prompt. Conditional
                # inversion uses the prompt as given, so leaving it empty gives empty-prompt conditioning.
                # (Upstream's generate_diffusion_cond always blanks the prompt here instead.)
                cfg_dropout_prob = 1.0 if inversion_unconditional else 0.0

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
            # The ODE inversion below always runs the denoiser on the prompt's conditioning
            inversion_unconditional = False

            # Capture the fully pre-conditioned denoiser from stable-audio-tools
            captured_denoiser = []
            captured_extra_args = []
            
            import stable_audio_tools.inference.generation as sat_gen
            import k_diffusion.sampling as k_samp
            import stable_audio_tools.inference.k_diffusion.sampling as sat_k_samp
            
            original_sample_heun = k_samp.sample_heun
            original_sat_sample_heun = sat_k_samp.sample_heun
            
            def capture_denoiser(denoiser, x, sigmas, *args, **kwargs_inner):
                captured_denoiser.append(denoiser)
                captured_extra_args.append(kwargs_inner.get('extra_args', {}))
                raise InterruptedError("Captured denoiser")
                
            k_samp.sample_heun = capture_denoiser
            sat_k_samp.sample_heun = capture_denoiser
            
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
                sat_k_samp.sample_heun = original_sat_sample_heun
                
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

            # Unit-variance normalisation; the stats are recorded so generation can restore the latent
            # to its noise level (essential below full strength, where data still dominates the variance)
            latent_mean = current_latents.mean().item()
            latent_std = current_latents.std().item()
            if not latent_std > 0:
                raise ValueError(f"Inverted latent has degenerate standard deviation {latent_std}.")
            latent_tensor = (current_latents - latent_mean) / latent_std
            latent_stats = {"latent_mean": latent_mean, "latent_std": latent_std}

            algorithm = "euler_ode_inversion"
            sampler = "euler"

        # Create latent artifact tracking IDs
        content_uid = self.uid_generator.from_tensor(latent_tensor)

        # Filter context metadata properties
        context = {}
        for k, v in kwargs.items():
            if k in ("job_id", "execution_mode"):
                continue
            if k.endswith('_element'):
                context[k.replace('_element', '_id')] = v.id
            elif k.endswith('_elements'):
                context[k.replace('_elements', '_ids')] = [el.id for el in v]
            elif isinstance(v, GraphElement):
                # Other element arguments (e.g. baseline_grating) are referenced by ID, not serialized whole
                context[f"{k}_id"] = v.id
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
            "sigma_max": sigma_max,
            "inversion_unconditional": inversion_unconditional,
            "inversion_gamma": kwargs.get("inversion_gamma", 0.3),
            "inversion_cfg_scale": kwargs.get("inversion_cfg_scale", 1.0),
            **latent_stats,
        }

        artifact = Latent(
            id=content_uid,
            name=generate_slug(2),
            file=Asset(path=None, uid=content_uid, extension=".pt"),
            context=context
        )
        
        return artifact, latent_tensor