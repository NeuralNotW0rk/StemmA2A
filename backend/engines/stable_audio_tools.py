from dataclasses import dataclass
import json
import torch
import torchaudio
from stable_audio_tools import get_pretrained_model, create_model_from_config
from stable_audio_tools.models.utils import load_ckpt_state_dict
from stable_audio_tools.inference.generation import generate_diffusion_cond

from param_graph.engine import Engine, Model
from param_graph.uid_gen import UIDMismatchError

HF_MODEL_NAME = "Stable Audio Open Small"
HF_MODEL_PATH = "stabilityai/stable-audio-open-small"

@dataclass(kw_only=True)
class StableAudioModel(Model):
    checkpoint_path: str
    config: dict
    model_type: str
    engine: str = 'stable_audio_tools'


class StableAudioTools(Engine):
    def __init__(self) -> None:
        super().__init__()
        self.name = 'stable_audio_tools'
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_id = None
        self.model = None

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
            uid=self.uid_generator.from_module(model),
            uid_type=self.uid_generator.type,
            uid_version=self.uid_generator.version
        )

    def load_model(self, ele: StableAudioModel, verify: bool=True):
        # Check if a model is currently loaded
        if self.model:
            # Same model... do nothing
            if self.model_id == ele.uid:
                return
            
            # Unload existing model
            del self.model

            # Use stored config and load model from path
            self.model = create_model_from_config(ele.config)
            self.model.load_state_dict(load_ckpt_state_dict(ele.path))

            if verify and self.uid_generator:
                uid = self.uid_generator.from_module(self.model)
                if uid != self.uid:
                    raise UIDMismatchError()
                
    @staticmethod
    def get_form_config():
        return {
            "generate": [
                {
                    "name": "prompt",
                    "label": "Prompt",
                    "type": "textarea",
                    "defaultValue": "128 BPM tech house drum loop",
                    "placeholder": "Enter a prompt for the model..."
                },
                {
                    "name": "steps",
                    "label": "Steps",
                    "type": "number",
                    "defaultValue": 100,
                    "placeholder": "Enter number of steps"
                },
                {
                    "name": "cfg_scale",
                    "label": "CFG Scale",
                    "type": "number",
                    "defaultValue": 7.0,
                    "placeholder": "Enter CFG scale"
                },
                {
                    "name": "seconds_total",
                    "label": "Seconds",
                    "type": "number",
                    "defaultValue": 30,
                    "placeholder": "Enter duration in seconds"
                },
                {
                    "name": "k_sampler_type",
                    "label": "K-Sampler",
                    "type": "select",
                    "defaultValue": "dpmpp_2m",
                    "options": [
                        { "label": "DPM++ 2M", "value": "dpmpp_2m" },
                        { "label": "DPM++ SDE", "value": "dpmpp_sde" },
                        { "label": "k-Heun", "value": "k_heun" },
                        { "label": "k-LMS", "value": "k_lms" },
                        { "label": "k-Euler", "value": "k_euler" },
                        { "label": "k-DPM2", "value": "k_dpm_2" },
                        { "label": "k-DPM Adaptive", "value": "k_dpm_adaptive" }
                    ],
                    "show_if": {"model_type": "k_diffusion"}
                },
                {
                    "name": "rf_sampler_type",
                    "label": "RF-Sampler",
                    "type": "select",
                    "defaultValue": "euler",
                    "options": [
                        { "label": "Euler", "value": "euler" },
                        { "label": "RK4", "value": "rk4" },
                        { "label": "DPM++", "value": "dpmpp" }
                    ],
                    "show_if": {"model_type": "rectified_flow"}
                },
                {
                    "name": "seed",
                    "label": "Seed",
                    "type": "number",
                    "defaultValue": 42,
                    "placeholder": "Enter a seed"
                }
            ],
            "import": [
                {
                    "name": "name",
                    "label": "Model Name",
                    "type": "text",
                    "placeholder": "Enter a name for the model",
                    "required": True
                },
                {
                    "name": "model_type",
                    "label": "Model Type",
                    "type": "select",
                    "defaultValue": "k_diffusion",
                    "options": [
                        { "label": "K-Diffusion", "value": "k_diffusion" },
                        { "label": "Rectified Flow", "value": "rectified_flow" }
                    ]
                },
                {
                    "name": "checkpoint_path",
                    "label": "Checkpoint File",
                    "type": "file",
                    "placeholder": "Select a checkpoint file",
                    "required": True,
                    "filters": [
                        { "name": "Model Files", "extensions": ["ckpt", "safetensors", "pt", "pth", "bin"] },
                        { "name": "All Files", "extensions": ["*"] }
                    ]
                },
                {
                    "name": "config_path",
                    "label": "Config File",
                    "type": "file",
                    "placeholder": "Select a config file",
                    "required": True,
                    "filters": [{ "name": "JSON Files", "extensions": ["json"] }]
                }
            ]
        }

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
