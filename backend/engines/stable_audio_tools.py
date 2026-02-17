from dataclasses import dataclass
import torch
import torchaudio
from stable_audio_tools import get_pretrained_model, create_model_from_config, load_ckpt_state_dict
from stable_audio_tools.inference.generation import generate_diffusion_cond

from ..param_graph.engine import Engine, Model
from ..param_graph.uid_gen import UIDGenerator, UIDMismatchError

HF_MODEL_DEFAULT = "stabilityai/stable-audio-open-small"

@dataclass(kw_only=True)
class StableAudioModel(Model):
    default: bool
    path: str
    config: dict

    def __post_init__(self):
        super().__post_init__()
        self.engine = 'stable_audio_tools'


class StableAudioTools(Engine):
    def __init__(self) -> None:
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_id = None
        self.model = None

    def register_model(path: str, config: dict) -> StableAudioModel:
        model = create_model_from_config(config)
        model.load_state_dict(load_ckpt_state_dict(path))

        return StableAudioModel(
            default=False,
            path=path,
            config=config,
            uid=UIDGenerator.from_module(model)
        )    

    def register_model_default() -> StableAudioModel:
        model, config = get_pretrained_model(HF_MODEL_DEFAULT)

        return StableAudioModel(
            default=True,
            path=HF_MODEL_DEFAULT,
            config=config,
            uid=UIDGenerator.from_module(model)
        )

    def load_model(self, ele: StableAudioModel, verify: bool=True):
        # Check if a model is currently loaded
        if self.model:
            # Same model... do nothing
            if self.model_id == ele.uid:
                return
            
            # Unload existing model
            del self.model

            if ele.default:
                # Handle file i/o via the internal huggingface implementation
                self.model, _ = get_pretrained_model()
            else:
                # Use stored config and load model from path
                self.model = create_model_from_config(ele.config)
                self.model.load_state_dict(load_ckpt_state_dict(ele.path))

            if verify:
                uid = UIDGenerator.from_module(self.model)
                if uid != self.uid:
                    raise UIDMismatchError()
                
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
