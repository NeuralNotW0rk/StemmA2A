"""
Pivotal Tuning Inversion (Roich et al., 2021) for the StyleGAN2 adapter.

Two phases:
  1. Projection: optimize a single w (the pivot) so G(w) reconstructs the target image, following the
     StyleGAN2-ADA projector (start at w_avg, decaying exploration noise, cosine lr ramp).
  2. Pivotal tuning: freeze w and the generator, and train weight-space LoRA factors on the synthesis
     convs until G(w) matches the target. The rank limit keeps the tuning local to the pivot, which is
     what PTI's locality regularizer is for.
"""

import math
from typing import Callable, Optional

import torch
import torch.nn.functional as F
from PIL import Image as PILImage
from torchvision.transforms import functional as TF

from diffracture.actant import Actant
from diffracture.topology.grating import Grating
from diffracture.topology.lora import LoRAElement, find_lora_targets

# The synthesis network's modulated convs; the mapping network is left out because w is fixed
SYNTHESIS_WEIGHTS = r"^(conv1|convs\.\d+|to_rgb1|to_rgbs\.\d+)\.conv\.weight$"


def load_target_image(path: str, size: int, device) -> torch.Tensor:
    """Loads an image as a [1, 3, size, size] tensor in [-1, 1], center-cropped to square."""
    img = PILImage.open(path).convert("RGB")
    side = min(img.size)
    img = TF.center_crop(img, [side, side])
    img = TF.resize(img, [size, size], interpolation=TF.InterpolationMode.LANCZOS, antialias=True)
    return (TF.to_tensor(img) * 2.0 - 1.0).unsqueeze(0).to(device)


class ReconstructionLoss:
    """
    LPIPS (VGG) plus a weighted pixel MSE. LPIPS is optional: without the `lpips` package, or when its
    VGG16 weights can't be fetched (first use downloads them to TORCH_HOME), the loss falls back to MSE
    alone, which reconstructs color and layout but leaves fine detail softer.
    """
    def __init__(self, device, mse_weight: float = 1.0, use_lpips: bool = True):
        self.mse_weight = mse_weight
        self.lpips = None
        if use_lpips:
            try:
                import lpips
            except ImportError:
                print("PTI: 'lpips' is not installed; using pixel MSE only. `pip install lpips` for sharper inversions.")
            else:
                try:
                    self.lpips = lpips.LPIPS(net="vgg", verbose=False).to(device).eval()
                    self.lpips.requires_grad_(False)
                except Exception as e:
                    print(f"PTI: could not load LPIPS (VGG16 weights unavailable?): {e}. Using pixel MSE only.")

    @property
    def name(self) -> str:
        return "lpips+mse" if self.lpips is not None else "mse"

    def __call__(self, synth: torch.Tensor, target: torch.Tensor, mse_weight: Optional[float] = None) -> torch.Tensor:
        mse_weight = self.mse_weight if mse_weight is None else mse_weight
        loss = F.mse_loss(synth, target) * mse_weight
        if self.lpips is not None:
            # LPIPS was calibrated at 256px; larger images are compared downsampled, as in the projector
            if synth.shape[-1] > 256:
                synth = F.interpolate(synth, size=(256, 256), mode="area")
                target = F.interpolate(target, size=(256, 256), mode="area")
            loss = loss + self.lpips(synth, target).mean()
        return loss


def render(model, w: torch.Tensor) -> torch.Tensor:
    img, _ = model([w], input_is_latent=True, randomize_noise=False)
    return img


def project_w(
    model,
    target: torch.Tensor,
    loss_fn: ReconstructionLoss,
    w_avg: torch.Tensor,
    steps: int = 500,
    initial_lr: float = 0.1,
    initial_noise_factor: float = 0.05,
    noise_ramp_length: float = 0.75,
    lr_rampdown_length: float = 0.25,
    lr_rampup_length: float = 0.05,
    w_std_samples: int = 10000,
    progress: Optional[Callable[[str, int, int, float], None]] = None,
) -> torch.Tensor:
    """Finds the pivot: a single w in W (shape [1, style_dim]) whose rendering best matches target."""
    device = target.device
    with torch.no_grad():
        z = torch.randn(w_std_samples, model.style_dim, device=device)
        w_samples = model.style(z)
        w_std = ((w_samples - w_avg).pow(2).sum() / w_std_samples).sqrt().item()

    w_opt = w_avg.detach().clone().requires_grad_(True)
    optimizer = torch.optim.Adam([w_opt], betas=(0.9, 0.999), lr=initial_lr)

    for step in range(steps):
        t = step / max(steps, 1)
        w_noise_scale = w_std * initial_noise_factor * max(0.0, 1.0 - t / noise_ramp_length) ** 2
        lr_ramp = min(1.0, (1.0 - t) / lr_rampdown_length)
        lr_ramp = 0.5 - 0.5 * math.cos(lr_ramp * math.pi)
        lr_ramp = lr_ramp * min(1.0, t / lr_rampup_length)
        for group in optimizer.param_groups:
            group["lr"] = initial_lr * lr_ramp

        # Exploration noise on w keeps the early steps from settling in a poor basin
        w = w_opt + torch.randn_like(w_opt) * w_noise_scale
        loss = loss_fn(render(model, w), target, mse_weight=0.0 if loss_fn.lpips is not None else None)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if progress:
            progress("projection", step + 1, steps, loss.item())

    return w_opt.detach()


def build_tuning_grating(model, rank: int, alpha: Optional[float] = None) -> Grating:
    """A LoRA element on every synthesis conv weight, sized from the weight itself."""
    grating = Grating()
    for address, param, weight in find_lora_targets(model, include=SYNTHESIS_WEIGHTS):
        grating.add_element(LoRAElement.for_parameter(address, weight, rank=rank, alpha=alpha, param=param))
    if not grating.nodes:
        raise RuntimeError("PTI found no synthesis conv weights to tune.")
    return grating


def tune_pivot(
    model,
    w_pivot: torch.Tensor,
    target: torch.Tensor,
    loss_fn: ReconstructionLoss,
    rank: int = 4,
    steps: int = 350,
    lr: float = 1e-3,
    progress: Optional[Callable[[str, int, int, float], None]] = None,
) -> tuple[Grating, float]:
    """Trains LoRA factors on the frozen generator so G(w_pivot) reproduces target. Returns (grating, final loss)."""
    grating = build_tuning_grating(model, rank=rank)
    grating.to(target.device)
    params = [p for element in grating._nodes for p in element.params.values()]
    optimizer = torch.optim.Adam(params, lr=lr)

    actant = Actant(model)
    actant.activate(grating, injection_strategy="hook")
    loss_value = float("nan")
    try:
        for step in range(steps):
            loss = loss_fn(render(model, w_pivot), target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            loss_value = loss.item()
            if progress:
                progress("tuning", step + 1, steps, loss_value)
    finally:
        actant.deactivate()

    for p in params:
        p.requires_grad_(False)
    return grating, loss_value


def pivotal_tuning_inversion(
    model,
    target: torch.Tensor,
    w_avg: torch.Tensor,
    projection_steps: int = 500,
    tuning_steps: int = 350,
    rank: int = 4,
    projection_lr: float = 0.1,
    tuning_lr: float = 1e-3,
    use_lpips: bool = True,
    progress: Optional[Callable[[str, int, int, float], None]] = None,
) -> dict:
    """
    Runs both PTI phases on a generator, leaving it unmodified afterwards.
    Returns {"grating", "w_pivot", "projection_loss", "tuning_loss", "loss"}.
    """
    # Only the pivot and the LoRA factors are optimized; the generator itself never trains
    was_requiring_grad = {name: p.requires_grad for name, p in model.named_parameters()}
    model.requires_grad_(False)
    loss_fn = ReconstructionLoss(target.device, use_lpips=use_lpips)

    try:
        w_pivot = project_w(model, target, loss_fn, w_avg, steps=projection_steps, initial_lr=projection_lr, progress=progress)
        with torch.no_grad():
            projection_loss = loss_fn(render(model, w_pivot), target).item()

        grating, tuning_loss = tune_pivot(
            model, w_pivot, target, loss_fn, rank=rank, steps=tuning_steps, lr=tuning_lr, progress=progress
        )
    finally:
        for name, p in model.named_parameters():
            p.requires_grad_(was_requiring_grad.get(name, False))

    return {
        "grating": grating,
        "w_pivot": w_pivot,
        "projection_loss": projection_loss,
        "tuning_loss": tuning_loss,
        "loss": loss_fn.name,
    }

