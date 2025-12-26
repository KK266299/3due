# file: src/core/ue_algos/unet_boundary_noise.py
"""
UNet Boundary-Aware Noise Generator for 3D segmentation (e.g., BraTS19).

This algorithm generates noise that is weighted by boundary regions. The key insight
is that boundary regions are more critical for segmentation accuracy, so focusing
noise on these regions can more effectively disrupt model learning.

Core idea:
  - Extract boundary regions from GT labels using morphological operations
  - Generate noise using U-Net
  - Weight the noise based on boundary proximity: higher weight near boundaries
  - This creates "unlearnable examples" that disrupt boundary learning

Boundary extraction method:
  - Use 3D morphological dilation - erosion to extract boundary
  - Create a distance-based weight map that smoothly transitions from boundary to interior
  - Boundary weight is higher than interior weight

Key difference from UNet ROI Noise:
  - UNet ROI Noise: uniform noise within ROI (label > 0)
  - UNet Boundary Noise: higher noise weight at boundaries, lower weight at interior
"""
from __future__ import annotations
from typing import Dict, Iterable, List

import torch
import torch.nn as nn
import torch.nn.functional as F
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


def _create_3d_sphere_kernel(radius: int, device: torch.device) -> torch.Tensor:
    """
    Create a 3D spherical structuring element for morphological operations.

    Args:
        radius: Radius of the sphere
        device: Target device

    Returns:
        kernel: [1, 1, D, H, W] binary kernel
    """
    size = 2 * radius + 1
    center = radius

    # Create coordinate grids
    z, y, x = torch.meshgrid(
        torch.arange(size, device=device),
        torch.arange(size, device=device),
        torch.arange(size, device=device),
        indexing='ij'
    )

    # Compute distance from center
    dist = torch.sqrt((z - center).float()**2 + (y - center).float()**2 + (x - center).float()**2)

    # Create spherical kernel
    kernel = (dist <= radius).float()

    return kernel.view(1, 1, size, size, size)


def _morphological_dilate_3d(mask: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """
    3D morphological dilation using max pooling approximation.

    Args:
        mask: [B, 1, D, H, W] binary mask
        kernel: [1, 1, kD, kH, kW] structuring element

    Returns:
        dilated: [B, 1, D, H, W] dilated mask
    """
    ksize = kernel.shape[2]
    pad = ksize // 2

    # Pad input
    padded = F.pad(mask, (pad, pad, pad, pad, pad, pad), mode='constant', value=0)

    # Use max pooling with kernel size
    dilated = F.max_pool3d(padded, kernel_size=ksize, stride=1, padding=0)

    return dilated


def _morphological_erode_3d(mask: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """
    3D morphological erosion using min pooling approximation.

    Args:
        mask: [B, 1, D, H, W] binary mask
        kernel: [1, 1, kD, kH, kW] structuring element

    Returns:
        eroded: [B, 1, D, H, W] eroded mask
    """
    ksize = kernel.shape[2]
    pad = ksize // 2

    # Pad input with 1 (for erosion, we want min)
    padded = F.pad(mask, (pad, pad, pad, pad, pad, pad), mode='constant', value=1)

    # Use -max_pool(-x) = min_pool(x)
    eroded = -F.max_pool3d(-padded, kernel_size=ksize, stride=1, padding=0)

    return eroded


@register_plugin("unet_boundary_noise")
class UNetBoundaryNoiseUE:
    """
    UNet-based Boundary-Aware noise generation for 3D segmentation UE.

    This method generates noise that is weighted based on boundary proximity.
    Boundary regions receive higher noise weight, making it harder for models
    to learn accurate boundary delineation.

    Boundary extraction:
      - From GT labels, extract boundary using morphological operations
      - boundary = dilate(mask) - erode(mask)
      - Create weight map: boundary region has weight 1.0, interior has lower weight

    Noise application:
      - Generate noise using U-Net
      - Apply boundary weight map to the noise
      - Higher weight at boundaries, lower weight at interior (but still non-zero)

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
        self._boundary_kernel: torch.Tensor | None = None
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

    def _create_boundary_weight_map(
        self,
        label: torch.Tensor,
        device: torch.device,
        boundary_width: int = 3,
        interior_weight: float = 0.3,
        boundary_weight: float = 1.0
    ) -> torch.Tensor:
        """
        Create boundary-aware weight map from label.

        The weight map assigns:
          - boundary_weight (default 1.0) to boundary regions
          - interior_weight (default 0.3) to interior regions
          - 0.0 to background regions

        Args:
            label: [B, D, H, W] segmentation labels
            device: Target device
            boundary_width: Width of boundary band (morphological kernel radius)
            interior_weight: Weight for interior (non-boundary) foreground regions
            boundary_weight: Weight for boundary regions

        Returns:
            weight_map: [B, 1, D, H, W] weight map
        """
        # Ensure label is [B, D, H, W]
        if label.dim() == 5:
            label = label.squeeze(1)

        # Create binary foreground mask: ROI = label > 0
        fg_mask = (label > 0).float().unsqueeze(1)  # [B, 1, D, H, W]
        fg_mask = fg_mask.to(device)

        # Create/cache morphological kernel
        if self._boundary_kernel is None or self._boundary_kernel.device != device:
            self._boundary_kernel = _create_3d_sphere_kernel(boundary_width, device)

        # Compute boundary using morphological operations
        # boundary = dilate(mask) - erode(mask)
        dilated = _morphological_dilate_3d(fg_mask, self._boundary_kernel)
        eroded = _morphological_erode_3d(fg_mask, self._boundary_kernel)

        # Boundary band
        boundary_mask = (dilated - eroded).clamp(0, 1)

        # Interior = foreground - boundary
        interior_mask = (fg_mask - boundary_mask).clamp(0, 1)

        # Create weight map
        # boundary regions: boundary_weight
        # interior regions: interior_weight
        # background: 0
        weight_map = boundary_mask * boundary_weight + interior_mask * interior_weight

        return weight_map

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
            f"[UNetBoundaryNoise] Initialized noise UNet: in_channels={in_channels}, "
            f"spatial_dims={spatial_dims}, eps={eps:.6f}, lr={lr}"
        )

    # ---------------- Surrogate-step: Update surrogate ---------------- #
    def surrogate_step_batch(self, trainer, batch) -> Dict[str, float]:
        """
        Update surrogate parameters only, do not update noise.
        Uses noise from backend (with boundary weight applied).
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

    # ---------------- N-step: Update noise via UNet (Boundary-aware) ---------------- #
    def noise_step_batch(self, trainer, batch) -> Dict[str, float]:
        """
        Update noise using trainable U-Net with boundary-aware weighting.

        Process:
          1. Create boundary weight map from GT labels
          2. Forward image through noise U-Net to get noise prediction
          3. Apply boundary weight map to noise (higher weight at boundaries)
          4. Add weighted noise to image, pass through frozen surrogate
          5. Compute segmentation loss
          6. Backprop to update noise U-Net parameters
          7. Store weighted noise to backend
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

        # Boundary-specific params
        boundary_width = int(get_config(params, "boundary_width", 3))
        interior_weight = float(get_config(params, "interior_weight", 0.3))
        boundary_weight_val = float(get_config(params, "boundary_weight", 1.0))

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

        # -------- create boundary weight map --------
        boundary_weight_map = self._create_boundary_weight_map(
            y, device,
            boundary_width=boundary_width,
            interior_weight=interior_weight,
            boundary_weight=boundary_weight_val
        )
        # Expand to match input channels: [B, C_in, D, H, W]
        boundary_weight_map = boundary_weight_map.expand(-1, C_in, -1, -1, -1)

        # -------- train noise UNet --------
        self._noise_unet.train()
        last_loss = torch.tensor(0.0, device=device)

        for _ in range(max(1, num_steps)):
            # Generate noise from UNet
            delta_raw = self._noise_unet(x)  # [B, C_in, ...]

            # Apply boundary weight map
            delta = delta_raw * boundary_weight_map

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

        # -------- Store weighted noise to backend --------
        self._noise_unet.eval()
        with torch.no_grad():
            final_delta_raw = self._noise_unet(x)
            final_delta = final_delta_raw * boundary_weight_map
            final_delta = final_delta.clamp(-eps, eps)

        nb.commit_batch(keys_list, final_delta.detach().cpu())

        delta_linf = float(final_delta.detach().abs().max().cpu())
        return {
            "noise_loss": float(last_loss.cpu()),
            "delta_linf": delta_linf,
        }