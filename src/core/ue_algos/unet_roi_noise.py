# file: src/core/ue_algos/unet_roi_noise.py
"""
UNet ROI Noise Generator for 3D segmentation (e.g., BraTS19).

This algorithm is similar to UNet Noise, but only generates noise within
the ROI (Region of Interest) area. ROI is defined as label > 0.

Key difference from UNet Noise:
  - UNet Noise: generates noise for the entire image
  - UNet ROI Noise: generates noise only within ROI (label > 0), background = 0
"""
from __future__ import annotations
from typing import Dict, Iterable, List

import torch
import torch.nn as nn
from omegaconf import DictConfig
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
        out_channels=in_channels,
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
        noise = torch.tanh(raw_noise) * self.epsilon
        return noise


@register_plugin("unet_roi_noise")
class UNetROINoiseUE:
    """
    UNet-based ROI noise generation for 3D segmentation UE.

    Similar to UNetNoiseUE, but only generates noise within ROI regions.
    ROI is defined as: label > 0 (foreground), background (label == 0) has no noise.

    Assumptions (same as MinMin/UNetNoise):
      - Batch:
          batch["image"]: FloatTensor [B, C, ...]      (3D: [B,C,D,H,W])
          batch["label"]: LongTensor  [B,   ...]       (3D: [B,D,H,W])
          batch["key"]:   sample-wise key
      - Surrogate:
          s_model(x) -> logits: [B, C_seg, ...]
      - Noise backend:
          noise_backend.batch_noise(keys) -> [N, C_in, ...]
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

    @staticmethod
    def _create_roi_mask(label: torch.Tensor, spatial_shape: tuple) -> torch.Tensor:
        """
        Create ROI mask from label. ROI = label > 0.

        Args:
            label: [B, D, H, W] or [B, 1, D, H, W]
            spatial_shape: target spatial shape
        Returns:
            mask: [B, 1, D, H, W] with 1 for ROI, 0 for background
        """
        # Ensure label is [B, D, H, W]
        if label.dim() == 5:
            label = label.squeeze(1)

        # Create binary mask: ROI = label > 0
        mask = (label > 0).float()  # [B, D, H, W]

        # Add channel dimension: [B, 1, D, H, W]
        mask = mask.unsqueeze(1)

        return mask

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

        noise_unet_cfg = get_config(cfg, "ue.noise_unet", DictConfig({}))
        eps = float(get_config(cfg, "ue.algorithm.params.epsilon", 8/255))

        base_unet = _build_noise_unet(noise_unet_cfg, in_channels, spatial_dims)
        self._noise_unet = NoiseUNetWrapper(base_unet, epsilon=eps)
        self._noise_unet = self._noise_unet.to(device)
        self._noise_unet_device = device

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
            f"[UNetROINoise] Initialized noise UNet: in_channels={in_channels}, "
            f"spatial_dims={spatial_dims}, eps={eps:.6f}, lr={lr}"
        )

    # ---------------- Surrogate-step: Update surrogate ---------------- #
    def surrogate_step_batch(self, trainer, batch) -> Dict[str, float]:
        """
        Update surrogate parameters only, do not update noise.
        Uses noise from backend (with ROI mask applied).
        """
        cfg = trainer.config
        device = trainer.device
        nb = trainer.noise_backend
        if nb is None:
            raise RuntimeError("[UE] noise_backend is required.")

        x = batch["image"].to(device).float()
        y = batch["label"]
        y = y.to(device).long() if torch.is_tensor(y) else torch.as_tensor(
            y, device=device, dtype=torch.long
        )
        keys: Iterable[int] = batch["key"]

        B, C_in = x.shape[:2]
        spatial_dims = len(x.shape) - 2

        self._init_noise_unet(trainer, C_in, spatial_dims)

        mean = tuple(get_config(cfg, "training.data.transforms.mean", (0.0,) * C_in))
        std = tuple(get_config(cfg, "training.data.transforms.std", (1.0,) * C_in))

        delta = nb.batch_noise(list(keys)).to(device).float()
        if delta.shape[:2] != x.shape[:2]:
            raise RuntimeError(
                f"[UE] noise shape mismatch: noise {tuple(delta.shape)} vs input {tuple(x.shape)}"
            )

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

    # ---------------- N-step: Update noise via UNet (ROI only) ---------------- #
    def noise_step_batch(self, trainer, batch) -> Dict[str, float]:
        """
        Update noise using trainable U-Net, but only within ROI regions.

        Process:
          1. Forward image through noise U-Net to get noise prediction
          2. Create ROI mask from label (ROI = label > 0)
          3. Apply ROI mask to noise (background noise = 0)
          4. Add masked noise to image, pass through frozen surrogate
          5. Compute segmentation loss
          6. Backprop to update noise U-Net parameters
          7. Store masked noise to backend
        """
        cfg = trainer.config
        device = trainer.device
        nb = trainer.noise_backend
        if nb is None:
            raise RuntimeError("[UE] noise_backend is required.")

        x = batch["image"].to(device).float()
        y = batch["label"]
        y = y.to(device).long() if torch.is_tensor(y) else torch.as_tensor(
            y, device=device, dtype=torch.long
        )
        keys = batch["key"]
        keys_list: List[int] = list(keys)

        B, C_in = x.shape[:2]
        spatial_dims = len(x.shape) - 2

        self._init_noise_unet(trainer, C_in, spatial_dims)

        algo = require_config(cfg, "ue.algorithm")
        params = require_config(algo, "params")
        eps = float(get_config(params, "epsilon", 8 / 255.0))
        num_steps = int(get_config(params, "noise_step", 1))

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

        # -------- create ROI mask --------
        # ROI = label > 0, mask shape: [B, 1, D, H, W]
        roi_mask = self._create_roi_mask(y, x.shape[2:])
        # Expand to match input channels: [B, C_in, D, H, W]
        roi_mask = roi_mask.expand(-1, C_in, -1, -1, -1).to(device)

        # -------- train noise UNet --------
        self._noise_unet.train()
        last_loss = torch.tensor(0.0, device=device)

        for _ in range(max(1, num_steps)):
            # Generate noise from UNet
            delta_raw = self._noise_unet(x)  # [B, C_in, ...]

            # Apply ROI mask: only keep noise in ROI regions
            delta = delta_raw * roi_mask  # background noise = 0

            # Create perturbed image
            perturb_img = (x + delta).clamp(0.0, 1.0)
            xn = perturb_img.clone()
            self._norm_inplace(xn, mean, std)

            # Forward through surrogate
            out = s_model(xn)
            logits = out[0] if isinstance(out, (tuple, list)) else out

            # Compute loss
            loss = seg_loss_fn(logits, y.unsqueeze(1))
            last_loss = loss.detach()

            # Backprop to update noise UNet
            self._opt_noise_unet.zero_grad(set_to_none=True)
            loss.backward()
            self._opt_noise_unet.step()

        # -------- Store masked noise to backend --------
        self._noise_unet.eval()
        with torch.no_grad():
            final_delta_raw = self._noise_unet(x)
            final_delta = final_delta_raw * roi_mask  # Apply ROI mask
            final_delta = final_delta.clamp(-eps, eps)

        nb.commit_batch(keys_list, final_delta.detach().cpu())

        delta_linf = float(final_delta.detach().abs().max().cpu())
        return {
            "noise_loss": float(last_loss.cpu()),
            "delta_linf": delta_linf,
        }