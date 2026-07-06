import math
import random
import os
import json
import tempfile
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.utils import save_image

from .base_adapter import ModelAdapter, operation
from param_graph.elements.base_elements import Asset, GraphElement
from param_graph.elements.artifacts.image_element import Image
from param_graph.elements.models.stylegan_element import StyleGANModel
from utils.uid import XXH3_64

# ==============================================================================
# Pure PyTorch StyleGAN2 Generator Fallback Implementation (No C++ Compilation)
# ==============================================================================

def fused_leaky_relu(input, bias=None, negative_slope=0.2, scale=2 ** 0.5):
    """Pure PyTorch fallback for fused bias activation."""
    if bias is not None:
        if input.ndim == 4:
            bias = bias.view(1, -1, 1, 1)
        elif input.ndim == 2:
            bias = bias.view(1, -1)
        res = input + bias
    else:
        res = input
    return F.leaky_relu(res, negative_slope) * scale


def upfirdn2d_native(input, kernel, up_x, up_y, down_x, down_y, pad_x0, pad_x1, pad_y0, pad_y1):
    """Pure PyTorch native implementation of upfirdn2d."""
    _, in_h, in_w, minor = input.shape
    kernel_h, kernel_w = kernel.shape

    out = input.view(-1, in_h, 1, in_w, 1, minor)
    out = F.pad(out, [0, 0, 0, up_x - 1, 0, 0, 0, up_y - 1])
    out = out.view(-1, in_h * up_y, in_w * up_x, minor)

    out = F.pad(
        out, [0, 0, max(pad_x0, 0), max(pad_x1, 0), max(pad_y0, 0), max(pad_y1, 0)]
    )
    out = out[
        :,
        max(-pad_y0, 0) : out.shape[1] - max(-pad_y1, 0),
        max(-pad_x0, 0) : out.shape[2] - max(-pad_x1, 0),
        :,
    ]

    out = out.permute(0, 3, 1, 2)
    out = out.reshape(
        [-1, 1, in_h * up_y + pad_y0 + pad_y1, in_w * up_x + pad_x0 + pad_x1]
    )
    w = torch.flip(kernel, [0, 1]).view(1, 1, kernel_h, kernel_w)
    out = F.conv2d(out, w)
    out = out.reshape(
        -1,
        minor,
        in_h * up_y + pad_y0 + pad_y1 - kernel_h + 1,
        in_w * up_x + pad_x0 + pad_x1 - kernel_w + 1,
    )
    out = out.permute(0, 2, 3, 1)

    return out[:, ::down_y, ::down_x, :]


def upfirdn2d(input, kernel, up=1, down=1, pad=(0, 0)):
    """Convenience wrapper converting tensor layouts to upfirdn2d_native format."""
    x = input.permute(0, 2, 3, 1)
    out = upfirdn2d_native(
        x, kernel, up, up, down, down, pad[0], pad[1], pad[0], pad[1]
    )
    return out.permute(0, 3, 1, 2)


def make_kernel(k):
    k = torch.tensor(k, dtype=torch.float32)
    if k.ndim == 1:
        k = k[None, :] * k[:, None]
    k /= k.sum()
    return k


class PixelNorm(nn.Module):
    def forward(self, input):
        return input * torch.rsqrt(torch.mean(input ** 2, dim=1, keepdim=True) + 1e-8)


class Upsample(nn.Module):
    def __init__(self, kernel, factor=2):
        super().__init__()
        self.factor = factor
        kernel = make_kernel(kernel) * (factor ** 2)
        self.register_buffer('kernel', kernel)
        p = kernel.shape[0] - factor
        pad0 = (p + 1) // 2 + factor - 1
        pad1 = p // 2
        self.pad = (pad0, pad1)

    def forward(self, input):
        return upfirdn2d(input, self.kernel, up=self.factor, down=1, pad=self.pad)


class Blur(nn.Module):
    def __init__(self, kernel, pad, upsample_factor=1):
        super().__init__()
        kernel = make_kernel(kernel)
        if upsample_factor > 1:
            kernel = kernel * (upsample_factor ** 2)
        self.register_buffer('kernel', kernel)
        self.pad = pad

    def forward(self, input):
        return upfirdn2d(input, self.kernel, pad=self.pad)


class EqualLinear(nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, bias_init=0, lr_mul=1, activation=None):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_dim, in_dim).div_(lr_mul))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim).fill_(bias_init))
        else:
            self.bias = None
        self.activation = activation
        self.scale = (1 / math.sqrt(in_dim)) * lr_mul
        self.lr_mul = lr_mul

    def forward(self, input):
        if self.activation:
            out = F.linear(input, self.weight * self.scale)
            out = fused_leaky_relu(out, self.bias * self.lr_mul, negative_slope=0.2, scale=1.0)
        else:
            out = F.linear(input, self.weight * self.scale, bias=self.bias * self.lr_mul)
        return out


class FusedLeakyReLU(nn.Module):
    def __init__(self, channel, negative_slope=0.2, scale=2 ** 0.5):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(channel))
        self.negative_slope = negative_slope
        self.scale = scale

    def forward(self, input):
        return fused_leaky_relu(input, self.bias, self.negative_slope, self.scale)


class ModulatedConv2d(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        style_dim,
        demodulate=True,
        upsample=False,
        blur_kernel=[1, 3, 3, 1],
    ):
        super().__init__()
        self.eps = 1e-8
        self.kernel_size = kernel_size
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.upsample = upsample

        if upsample:
            factor = 2
            p = (len(blur_kernel) - factor) - (kernel_size - 1)
            pad0 = (p + 1) // 2 + factor - 1
            pad1 = p // 2 + 1
            self.blur = Blur(blur_kernel, pad=(pad0, pad1), upsample_factor=factor)

        fan_in = in_channel * kernel_size ** 2
        self.scale = 1 / math.sqrt(fan_in)
        self.padding = kernel_size // 2

        self.weight = nn.Parameter(
            torch.randn(1, out_channel, in_channel, kernel_size, kernel_size)
        )
        self.modulation = EqualLinear(style_dim, in_channel, bias_init=1)
        self.demodulate = demodulate

    def forward(self, input, style):
        batch, in_channel, height, width = input.shape
        style = self.modulation(style).view(batch, 1, in_channel, 1, 1)
        weight = self.scale * self.weight * style

        if self.demodulate:
            demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + 1e-8)
            weight = weight * demod.view(batch, self.out_channel, 1, 1, 1)

        weight = weight.view(
            batch * self.out_channel, in_channel, self.kernel_size, self.kernel_size
        )

        if self.upsample:
            input = input.view(1, batch * in_channel, height, width)
            weight = weight.view(
                batch, self.out_channel, in_channel, self.kernel_size, self.kernel_size
            )
            weight = weight.transpose(1, 2).reshape(
                batch * in_channel, self.out_channel, self.kernel_size, self.kernel_size
            )
            out = F.conv_transpose2d(input, weight, padding=0, stride=2, groups=batch)
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)
            out = self.blur(out)
        else:
            input = input.view(1, batch * in_channel, height, width)
            out = F.conv2d(input, weight, padding=self.padding, groups=batch)
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)

        return out


class NoiseInjection(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1))

    def forward(self, image, noise=None):
        if noise is None:
            batch, _, height, width = image.shape
            noise = image.new_empty(batch, 1, height, width).normal_()
        return image + self.weight * noise


class ConstantInput(nn.Module):
    def __init__(self, channel, size=4):
        super().__init__()
        self.input = nn.Parameter(torch.randn(1, channel, size, size))

    def forward(self, input):
        batch = input.shape[0]
        return self.input.repeat(batch, 1, 1, 1)


class StyledConv(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        style_dim,
        upsample=False,
        blur_kernel=[1, 3, 3, 1],
        demodulate=True,
    ):
        super().__init__()
        self.conv = ModulatedConv2d(
            in_channel,
            out_channel,
            kernel_size,
            style_dim,
            upsample=upsample,
            blur_kernel=blur_kernel,
            demodulate=demodulate,
        )
        self.noise = NoiseInjection()
        self.activate = FusedLeakyReLU(out_channel)

    def forward(self, input, style, noise=None):
        out = self.conv(input, style)
        out = self.noise(out, noise=noise)
        out = self.activate(out)
        return out


class ToRGB(nn.Module):
    def __init__(self, in_channel, style_dim, upsample=True, blur_kernel=[1, 3, 3, 1]):
        super().__init__()
        if upsample:
            self.upsample = Upsample(blur_kernel)
        else:
            self.upsample = None
        self.conv = ModulatedConv2d(in_channel, 3, 1, style_dim, demodulate=False)
        self.bias = nn.Parameter(torch.zeros(1, 3, 1, 1))

    def forward(self, input, style, skip=None):
        out = self.conv(input, style)
        out = out + self.bias
        if skip is not None and self.upsample is not None:
            skip = self.upsample(skip)
            out = out + skip
        return out


class Generator(nn.Module):
    """Pure PyTorch StyleGAN2 Generator (manipulation-free for hooks compatibility)."""
    def __init__(
        self,
        size,
        style_dim,
        n_mlp,
        channel_multiplier=2,
        blur_kernel=[1, 3, 3, 1],
        lr_mlp=0.01,
    ):
        super().__init__()
        self.size = size
        self.style_dim = style_dim

        layers = [PixelNorm()]
        for _ in range(n_mlp):
            layers.append(EqualLinear(style_dim, style_dim, lr_mul=lr_mlp, activation='fused_lrelu'))
        self.style = nn.Sequential(*layers)

        self.channels = {
            4: 512,
            8: 512,
            16: 512,
            32: 512,
            64: 256 * channel_multiplier,
            128: 128 * channel_multiplier,
            256: 64 * channel_multiplier,
            512: 32 * channel_multiplier,
            1024: 16 * channel_multiplier,
        }

        self.input = ConstantInput(self.channels[4])
        self.conv1 = StyledConv(self.channels[4], self.channels[4], 3, style_dim, blur_kernel=blur_kernel)
        self.to_rgb1 = ToRGB(self.channels[4], style_dim, upsample=False)

        self.log_size = int(math.log(size, 2))
        self.num_layers = (self.log_size - 2) * 2 + 1

        self.noises = nn.Module()
        for layer_idx in range(self.num_layers):
            res = (layer_idx + 5) // 2
            shape = [1, 1, 2 ** res, 2 ** res]
            self.noises.register_buffer(f'noise_{layer_idx}', torch.randn(*shape))

        self.convs = nn.ModuleList()
        self.to_rgbs = nn.ModuleList()

        in_channel = self.channels[4]
        for i in range(3, self.log_size + 1):
            out_channel = self.channels[2 ** i]
            self.convs.append(StyledConv(in_channel, out_channel, 3, style_dim, upsample=True, blur_kernel=blur_kernel))
            self.convs.append(StyledConv(out_channel, out_channel, 3, style_dim, blur_kernel=blur_kernel))
            self.to_rgbs.append(ToRGB(out_channel, style_dim))
            in_channel = out_channel

        self.n_latent = self.log_size * 2 - 2

    def mean_latent(self, n_latent):
        latent_in = torch.randn(n_latent, self.style_dim, device=self.input.input.device)
        latent = self.style(latent_in).mean(0, keepdim=True)
        return latent

    def forward(
        self,
        styles,
        truncation=1,
        truncation_latent=None,
        input_is_latent=False,
        noise=None,
        randomize_noise=True,
    ):
        if not input_is_latent:
            styles = [self.style(s) for s in styles]

        if noise is None:
            if randomize_noise:
                noise = [None] * self.num_layers
            else:
                noise = [getattr(self.noises, f'noise_{i}') for i in range(self.num_layers)]

        if truncation < 1:
            style_t = []
            for style in styles:
                style_t.append(truncation_latent + truncation * (style - truncation_latent))
            styles = style_t

        if len(styles) < 2:
            inject_index = self.n_latent
            if styles[0].ndim < 3:
                latent = styles[0].unsqueeze(1).repeat(1, inject_index, 1)
            else:
                latent = styles[0]
        else:
            inject_index = random.randint(1, self.n_latent - 1)
            latent = styles[0].unsqueeze(1).repeat(1, inject_index, 1)
            latent2 = styles[1].unsqueeze(1).repeat(1, self.n_latent - inject_index, 1)
            latent = torch.cat([latent, latent2], 1)

        out = self.input(latent)
        out = self.conv1(out, latent[:, 0], noise=noise[0])
        skip = self.to_rgb1(out, latent[:, 1])

        i = 1
        for conv1, conv2, noise1, noise2, to_rgb in zip(
            self.convs[::2], self.convs[1::2], noise[1::2], noise[2::2], self.to_rgbs
        ):
            out = conv1(out, latent[:, i], noise=noise1)
            out = conv2(out, latent[:, i + 1], noise=noise2)
            skip = to_rgb(out, latent[:, i + 2], skip)
            i += 2

        return skip, latent


# ==============================================================================
# Model UID Caching Helper (Copied design from stable_audio_adapter)
# ==============================================================================

def _get_cache_file_path() -> Path:
    data_path = os.environ.get("LOCAL_DATA_PATH") or os.environ.get("CONTAINER_DATA_PATH")
    if data_path:
        cache_dir = Path(data_path).expanduser()
        try:
            cache_dir.mkdir(parents=True, exist_ok=True)
            return cache_dir / "model_uid_cache.json"
        except Exception:
            pass
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
        except Exception:
            pass
            
    entry = cache.get(abs_path)
    if entry and entry.get("size") == file_size and entry.get("mtime") == file_mtime:
        return entry.get("uid")
        
    checkpoint = torch.load(abs_path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise ValueError("Loaded checkpoint is not a dictionary. Ensure it is a valid PyTorch model file.")
        
    state_dict = checkpoint.get('g_ema') or checkpoint.get('g') or checkpoint
    if not isinstance(state_dict, dict):
        raise ValueError("Checkpoint weights are not serialized as a dictionary.")
        
    filtered_state_dict = {k: v for k, v in state_dict.items() if 'manipulation' not in k}
    uid = uid_generator.from_state_dict(filtered_state_dict)
    print(f"Computed weight-based UID: {uid}")
    
    cache[abs_path] = {
        "size": file_size,
        "mtime": file_mtime,
        "uid": uid
    }
    
    try:
        with open(cache_file, "w") as f:
            json.dump(cache, f, indent=2)
    except Exception as e:
        print(f"Warning: Failed to save UID cache: {e}")
        
    return uid


# ==============================================================================
# StyleGAN2 Model Adapter Implementation
# ==============================================================================

class StyleGANAdapter(ModelAdapter):
    def __init__(self) -> None:
        super().__init__()
        self.name = 'stylegan2'
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.model_info: StyleGANModel | None = None
        self.mean_latent = None
        self.style_dim = 512

    def register_model(self, **kwargs) -> StyleGANModel:
        checkpoint_path = kwargs.get("checkpoint_path")
        if checkpoint_path and os.path.exists(checkpoint_path) and os.path.isfile(checkpoint_path):
            checkpoint_uid = _get_cached_uid(checkpoint_path, self.uid_generator)
            checkpoint_asset = Asset(path=checkpoint_path, uid=checkpoint_uid)
        else:
            import uuid
            checkpoint_uid = "stylegan2_rand_" + str(uuid.uuid4())[:8]
            checkpoint_asset = Asset(path="", uid=checkpoint_uid)

        size = int(kwargs.get("size", 256))
        channel_multiplier = int(kwargs.get("channel_multiplier", 2))

        return StyleGANModel(
            id=checkpoint_uid,
            name=kwargs.get("name") or "StyleGAN2",
            checkpoint=checkpoint_asset,
            config={
                "size": size,
                "channel_multiplier": channel_multiplier
            },
            context={}
        )

    def load_model(self, info: StyleGANModel, verify: bool = True):
        if self.model and self.model_info.id == info.id:
            return

        if self.model:
            del self.model
            self.model = None

        size = info.config.get("size", 256)
        channel_multiplier = info.config.get("channel_multiplier", 2)

        ckpt_path = info.checkpoint.path
        loaded_state_dict = None

        if ckpt_path and os.path.exists(ckpt_path) and os.path.isfile(ckpt_path):
            try:
                print(f"Inspecting checkpoint {ckpt_path} to auto-detect model shape...")
                checkpoint = torch.load(ckpt_path, map_location="cpu")
                if 'g_ema' in checkpoint:
                    state_dict = checkpoint['g_ema']
                elif 'g' in checkpoint:
                    state_dict = checkpoint['g']
                else:
                    state_dict = checkpoint
                
                loaded_state_dict = {k: v for k, v in state_dict.items() if 'manipulation' not in k}

                # Auto-detect size & channel_multiplier from weights keys
                conv_keys = [k for k in loaded_state_dict.keys() if k.startswith("convs.")]
                if conv_keys:
                    indices = [int(k.split(".")[1]) for k in conv_keys if k.split(".")[1].isdigit()]
                    if indices:
                        max_idx = max(indices)
                        log_size = (max_idx // 2) + 3
                        size = 2 ** log_size
                        
                        # Find channel_multiplier from high-res block (res >= 64)
                        for idx in sorted(list(set(indices))):
                            res = 2 ** ((idx // 2) + 3)
                            if res >= 64:
                                key = f"convs.{idx}.conv.weight"
                                if key in loaded_state_dict:
                                    out_channels = loaded_state_dict[key].shape[1]
                                    channel_multiplier = out_channels // (16384 // res)
                                    break
                        print(f"Auto-detected model architecture: size={size}, channel_multiplier={channel_multiplier}")
                        # Keep the model info config aligned
                        info.config["size"] = size
                        info.config["channel_multiplier"] = channel_multiplier
            except Exception as e:
                print(f"Failed to auto-detect model shape from checkpoint: {e}. Falling back to default/config values.")

        self.model = Generator(
            size=size,
            style_dim=self.style_dim,
            n_mlp=8,
            channel_multiplier=channel_multiplier
        )

        if loaded_state_dict is not None:
            try:
                print(f"Loading StyleGAN2 weights...")
                self.model.load_state_dict(loaded_state_dict, strict=False)
            except Exception as e:
                print(f"Failed to load weights: {e}. Running with random weights.")
        else:
            print("No valid checkpoint file. Initializing model with random weights.")

        self.model.to(self.device)
        self.model.eval()
        self.model_info = info

        with torch.no_grad():
            self.mean_latent = self.model.mean_latent(4096).to(self.device)

    def cleanup(self):
        if self.model:
            del self.model
            self.model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    @operation(
        name="generate",
        is_standard=True,
        description="Generate images using StyleGAN2 model",
        initiator_types=["model", "grating", "latent"],
        context_overrides={
            "model": {
                "name": "Generate Image",
                "description": "Generate image from the selected StyleGAN2 model"
            },
            "grating": {
                "name": "generate from grating",
                "description": "Generate image guided by the selected grating structure"
            }
        }
    )
    def generate(self, **kwargs) -> tuple[Image, torch.Tensor]:
        truncation = float(kwargs.get("truncation", 0.7))
        seed = kwargs.get("seed", None)

        if seed is not None:
            torch.manual_seed(seed)
            import random
            random.seed(seed)
        
        with torch.no_grad():
            z = torch.randn(1, self.style_dim, device=self.device)
            img, _ = self.model([z], truncation=truncation, truncation_latent=self.mean_latent)
            img = img.squeeze(0) # Shape: [3, H, W]

        # Generate unique IDs
        uid_gen = XXH3_64()
        img_id = uid_gen.from_tensor(img)
        
        # Save output temporarily (engine will move it to the project files)
        temp_dir = tempfile.mkdtemp()
        temp_path = os.path.join(temp_dir, f"{img_id}.png")
        save_image(img, temp_path, normalize=True, value_range=(-1, 1))

        asset = Asset(path=temp_path, uid=img_id, extension=".png")
        size = self.model_info.config.get("size", 256)
        
        image_artifact = Image(
            id=img_id,
            name=f"stylegan_gen_{seed if seed is not None else 'rand'}",
            context={
                "seed": seed,
                "truncation": truncation,
                "size": size
            },
            file=asset,
            width=size,
            height=size
        )
        image_artifact._temp_dir_ref = temp_dir

        return image_artifact, img

    @operation(
        name="invert",
        is_standard=True,
        description="StyleGAN2 latent space projection",
        initiator_types=["model", "image"]
    )
    def invert(self, **kwargs) -> tuple[GraphElement, torch.Tensor]:
        # Return a mock mapping or a w-latent for test purposes
        # Since StyleGAN2 inversion involves optimization, we can generate a random/empty w-latent mapping
        import uuid
        from param_graph.elements.artifacts.latent_element import Latent
        
        latent_id = "latent_" + str(uuid.uuid4())[:8]
        temp_dir = tempfile.mkdtemp()
        temp_path = os.path.join(temp_dir, f"{latent_id}.pt")
        
        # Mock projection vector of shape [1, 14, 512]
        latent_vector = torch.zeros(1, self.model.n_latent, self.style_dim)
        torch.save(latent_vector, temp_path)
        
        asset = Asset(path=temp_path, uid=latent_id, extension=".pt")
        latent_artifact = Latent(
            id=latent_id,
            name="stylegan_inversion_latent",
            context={"inversion_strength": 1.0},
            file=asset
        )
        latent_artifact._temp_dir_ref = temp_dir
        
        return latent_artifact, latent_vector
