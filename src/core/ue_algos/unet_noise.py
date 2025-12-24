# file: src/core/ue_algos/unet_noise.py
"""
UNet Noise Generator for 3D segmentation (e.g., BraTS19).

This algorithm replaces the PGD noise update in MinMin with a trainable U-Net
that generates noise. The U-Net takes the original image as input and outputs
perturbation noise directly.

Core differences from MinMin:
  - MinMin: PGD gradient descent to update noise directly
  - UNetNoise: Train a U-Net to predict noise, then store generated noise

Training loop:
  1. Surrogate-step: Same as MinMin, train surrogate on noisy images
  2. Noise-step:
     - Forward original image through noise U-Net to get noise prediction
     - Add noise to image, pass through surrogate, compute loss
     - Backprop to update noise U-Net parameters
     - Store generated noise to backend
"""
from __future__ import annotations
from typing import Dict, Iterable, List, Optional, Any

import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from monai.losses import DiceCELoss
from monai.networks.nets import UNet as MonaiUNet

from ...registry import register_plugin
from ...utils.config import get_config, require_config
from ...utils.logger import get_logger


def _build_noise_unet(cfg: DictConfig, in_channels: int, spatial_dims: int = 3) -> nn.Module:
    """
    Build a small U-Net for noise generation.
    Input: original image [B, C, D, H, W]
    Output: noise [B, C, D, H, W] (same shape as input)
    """
    channels = list(get_config(cfg, "channels", [16, 32, 64, 128]))
    strides = list(get_config(cfg, "strides", [2, 2, 2]))
    num_res_units = int(get_config(cfg, "num_res_units", 1))
    act = get_config(cfg, "act", "LEAKYRELU")
    norm = get_config(cfg, "norm", "INSTANCE")
    dropout = float(get_config(cfg, "dropout", 0.0))

    unet = MonaiUNet(
        spatial_dims=spatial_dims,
        in_channels=in_channels,
        out_channels=in_channels,  # output same channels as input
        channels=channels,
        strides=strides,
        num_res_units=num_res_units,
        act=act,
        norm=norm,
        dropout=dropout,
    )
    return unet


class NoiseUNetWrapper(nn.Module):
    """
    Wrapper for noise U-Net that applies tanh and scales output to [-eps, eps].
    """
    def __init__(self, unet: nn.Module, epsilon: float = 8/255):
        super().__init__()
        self.unet = unet
        self.epsilon = epsilon

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input image [B, C, D, H, W] in [0, 1]
        Returns:
            Noise [B, C, D, H, W] in [-eps, eps]
        """
        raw_noise = self.unet(x)
        # Apply tanh to bound output to [-1, 1], then scale to [-eps, eps]
        noise = torch.tanh(raw_noise) * self.epsilon
        return noise


@register_plugin("unet_noise")
class UNetNoiseUE:
    """
    UNet-based noise generation for 3D segmentation UE.

    Instead of using PGD to update noise iteratively, this method uses a
    trainable U-Net to generate noise. The U-Net is trained to minimize
    the segmentation loss on noisy images.

    Assumptions (same as MinMin):
      - Batch:
          batch["image"]: FloatTensor [B, C, ...]      (3D: [B,C,D,H,W])
          batch["label"]: LongTensor  [B,   ...]       (3D: [B,D,H,W])
          batch["key"]:   sample-wise key
      - Surrogate:
          s_model(x) -> logits: [B, C_seg, ...]
      - Noise backend:
          noise_backend.batch_noise(keys) -> [N, C_in, ...]

    Additional components:
      - noise_unet: U-Net that generates noise from input images
      - opt_noise_unet: Optimizer for the noise U-Net
    """

    def __init__(self):
        self._seg_loss: DiceCELoss | None = None
        self._noise_unet: NoiseUNetWrapper | None = None
        self._opt_noise_unet: torch.optim.Optimizer | None = None
        self._noise_unet_device: torch.device | None = None
        self._initialized: bool = False
        self.logger = get_logger()

    @staticmethod
    def _norm_inplace(x: torch.Tensor, mean, std):
        """
        In-place per-channel normalize for ND volume.
        x: [B, C, ...]
        """
        for c, (m, s) in enumerate(zip(mean, std)):
            x[:, c].sub_(float(m)).div_(float(s))
        return x

    def _get_seg_loss(self, trainer) -> DiceCELoss:
        """
        Build DiceCELoss consistent with SegTrainer configuration.
        """
        if self._seg_loss is not None:
            return self._seg_loss

        cfg = trainer.config
        crit_cfg = get_config(cfg, "training.criterion", DictConfig({}))
        include_background = bool(get_config(crit_cfg, "include_background", False))
        squared_pred = bool(get_config(crit_cfg, "squared_pred", False))
        jaccard = bool(get_config(crit_cfg, "jaccard", False))
        lambda_dice = float(get_config(crit_cfg, "lambda_dice", 1.0))
        lambda_ce = float(get_config(crit_cfg, "lambda_ce", 1.0))

        self._seg_loss = DiceCELoss(
            include_background=include_background,
            to_onehot_y=True,
            softmax=True,
            squared_pred=squared_pred,
            jaccard=jaccard,
            lambda_dice=lambda_dice,
            lambda_ce=lambda_ce,
            reduction="mean",
        )
        return self._seg_loss

    def _init_noise_unet(self, trainer, in_channels: int, spatial_dims: int = 3):
        """
        Initialize the noise U-Net and its optimizer (lazy, once).
        """
        if self._initialized:
            return

        cfg = trainer.config
        device = trainer.device

        # Get noise UNet config
        noise_unet_cfg = get_config(cfg, "ue.noise_unet", DictConfig({}))
        eps = float(get_config(cfg, "ue.algorithm.params.epsilon", 8/255))

        # Build noise UNet
        base_unet = _build_noise_unet(noise_unet_cfg, in_channels, spatial_dims)
        self._noise_unet = NoiseUNetWrapper(base_unet, epsilon=eps)
        self._noise_unet = self._noise_unet.to(device)
        self._noise_unet_device = device

        # Build optimizer
        opt_cfg = get_config(noise_unet_cfg, "optimizer", DictConfig({}))
        lr = float(get_config(opt_cfg, "lr", 1e-4))
        weight_decay = float(get_config(opt_cfg, "weight_decay", 1e-5))
        betas = tuple(get_config(opt_cfg, "betas", (0.9, 0.999)))

        self._opt_noise_unet = torch.optim.Adam(
            self._noise_unet.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=betas,
        )

        self._initialized = True
        self.logger.info(
            f"[UNetNoise] Initialized noise UNet: in_channels={in_channels}, "
            f"spatial_dims={spatial_dims}, eps={eps:.6f}, lr={lr}"
        )

    # ---------------- Surrogate-step: Update surrogate ---------------- #
    def surrogate_step_batch(self, trainer, batch) -> Dict[str, float]:
        """
        Update surrogate parameters only, do not update noise.
        Uses generated noise from noise U-Net.

        This is similar to MinMin, but noise comes from U-Net prediction
        instead of noise backend (for training, we still read from backend).
        """
        cfg = trainer.config
        device = trainer.device
        nb = trainer.noise_backend
        if nb is None:
            raise RuntimeError("[UE] noise_backend is required.")

        # data
        x = batch["image"].to(device).float()  # [B, C, ...]
        y = batch["label"]
        y = y.to(device).long() if torch.is_tensor(y) else torch.as_tensor(
            y, device=device, dtype=torch.long
        )
        keys: Iterable[int] = batch["key"]

        B, C_in = x.shape[:2]
        spatial_dims = len(x.shape) - 2  # 3 for 3D

        # Initialize noise UNet if not done
        self._init_noise_unet(trainer, C_in, spatial_dims)

        # normalization config
        mean = tuple(get_config(cfg, "training.data.transforms.mean", (0.0,) * C_in))
        std = tuple(get_config(cfg, "training.data.transforms.std", (1.0,) * C_in))

        # Get noise from backend (like MinMin for consistency during surrogate training)
        delta = nb.batch_noise(list(keys)).to(device).float()  # [B, C_in, ...]
        if delta.shape[:2] != x.shape[:2]:
            raise RuntimeError(
                f"[UE] noise shape mismatch: noise {tuple(delta.shape)} vs input {tuple(x.shape)}"
            )

        # select surrogate and optimizer
        if not trainer.surrogates:
            raise RuntimeError("[UE] No surrogate bound.")
        name, s_model = next(iter(trainer.surrogates.items()))
        opt = trainer.opt_surrogates.get(name, None)
        if opt is None:
            raise RuntimeError(f"[UE] No optimizer for surrogate '{name}'.")

        seg_loss_fn = self._get_seg_loss(trainer)

        s_model.train()
        for p in s_model.parameters():
            p.requires_grad = True

        # forward with noisy input
        noisy = (x + delta).clamp(0.0, 1.0)
        xn = noisy.clone()
        self._norm_inplace(xn, mean, std)

        out = s_model(xn)
        logits = out[0] if isinstance(out, (tuple, list)) else out

        loss = seg_loss_fn(logits, y.unsqueeze(1))

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        loss_val = float(loss.detach().cpu())
        return {
            "surrogate_loss": loss_val,
            "loss": loss_val,
        }

    # ---------------- N-step: Update noise via UNet ---------------- #
    def noise_step_batch(self, trainer, batch) -> Dict[str, float]:
        """
        Update noise using trainable U-Net instead of PGD.

        Process:
          1. Forward image through noise U-Net to get noise prediction
          2. Add noise to image, normalize, pass through frozen surrogate
          3. Compute segmentation loss
          4. Backprop to update noise U-Net parameters
          5. Store generated noise to backend
        """
        cfg = trainer.config
        device = trainer.device
        nb = trainer.noise_backend
        if nb is None:
            raise RuntimeError("[UE] noise_backend is required.")

        # -------- data & config --------
        x = batch["image"].to(device).float()  # [B, C_in, ...]
        y = batch["label"]
        y = y.to(device).long() if torch.is_tensor(y) else torch.as_tensor(
            y, device=device, dtype=torch.long
        )
        keys = batch["key"]
        keys_list: List[int] = list(keys)

        B, C_in = x.shape[:2]
        spatial_dims = len(x.shape) - 2

        # Initialize if needed
        self._init_noise_unet(trainer, C_in, spatial_dims)

        algo = require_config(cfg, "ue.algorithm")
        params = require_config(algo, "params")
        eps = float(get_config(params, "epsilon", 8 / 255.0))
        num_steps = int(get_config(params, "noise_step", 1))  # training iterations per batch

        # normalization config
        mean = tuple(get_config(cfg, "training.data.transforms.mean", (0.0,) * C_in))
        std = tuple(get_config(cfg, "training.data.transforms.std", (1.0,) * C_in))

        seg_loss_fn = self._get_seg_loss(trainer)

        # -------- freeze surrogate --------
        if not trainer.surrogates:
            raise RuntimeError("[UE] No surrogate bound.")
        _, s_model = next(iter(trainer.surrogates.items()))
        s_model.eval()
        for p in s_model.parameters():
            p.requires_grad = False

        # -------- train noise UNet --------
        self._noise_unet.train()
        last_loss = torch.tensor(0.0, device=device)

        for _ in range(max(1, num_steps)):
            # Generate noise from UNet
            delta = self._noise_unet(x)  # [B, C_in, ...], already clamped to [-eps, eps]

            # Create perturbed image
            perturb_img = (x + delta).clamp(0.0, 1.0)
            xn = perturb_img.clone()
            self._norm_inplace(xn, mean, std)

            # Forward through surrogate
            out = s_model(xn)
            logits = out[0] if isinstance(out, (tuple, list)) else out

            # Compute loss (min-min: minimize loss)
            loss = seg_loss_fn(logits, y.unsqueeze(1))
            last_loss = loss.detach()

            # Backprop to update noise UNet
            self._opt_noise_unet.zero_grad(set_to_none=True)
            loss.backward()
            self._opt_noise_unet.step()

        # -------- Store generated noise to backend --------
        # Generate final noise and commit
        self._noise_unet.eval()
        with torch.no_grad():
            final_delta = self._noise_unet(x)  # [B, C_in, ...]
            # Extra clamp for safety
            final_delta = final_delta.clamp(-eps, eps)

        nb.commit_batch(keys_list, final_delta.detach().cpu())

        delta_linf = float(final_delta.detach().abs().max().cpu())
        return {
            "noise_loss": float(last_loss.cpu()),
            "delta_linf": delta_linf,
        }