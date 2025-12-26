# file: src/core/ue_algos/unet_noise_slice.py
"""
Inter-Slice Consistency Attack for 3D Medical Image Segmentation.

This algorithm extends UNet Noise with inter-slice consistency constraints,
specifically designed for 3D medical imaging data (e.g., CT, MRI).

Key Innovation:
  - Multiple slice consistency modes to control noise behavior along depth axis
  - ROI-Aware: Optional focusing on regions of interest.
  - Gradient-Aware: Adapts noise smoothness based on image gradients.

Supported consistency modes:
  - "smooth": Encourage adjacent slices to have similar noise (L2 minimization)
  - "disrupt": Encourage adjacent slices to have different noise (L2 maximization)
  - "periodic": Apply periodic modulation to noise along depth axis

Core differences from UNetNoise:
  - UNetNoise: Generates noise without explicit slice constraints
  - UNetSliceConsistency: Adds explicit constraints along depth axis

Training loop:
  1. Surrogate-step: Same as UNetNoise, train surrogate on noisy images
  2. Noise-step:
     - Forward original image through noise U-Net to get noise prediction
     - Compute inter-slice consistency loss based on selected mode
     - Add noise to image, pass through surrogate, compute seg loss
     - Total loss depends on mode:
       * smooth:   seg_loss + lambda * consistency_loss
       * disrupt:  seg_loss - lambda * consistency_loss
       * periodic: seg_loss + lambda * periodic_loss
     - Backprop to update noise U-Net parameters
     - Store generated noise to backend

Maximum noise bound: 8/255 ≈ 0.0313725 (L∞ constraint)
"""
from __future__ import annotations
from typing import Dict, Iterable, List, Optional

import numpy as np
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


class SliceConsistencyLoss(nn.Module):
    """
    Inter-Slice Consistency Loss for 3D data.

    Computes consistency between adjacent slices along the depth dimension.
    Supports multiple consistency modes:
      - 'smooth': Minimize L2 distance between adjacent slices (encourage similarity)
      - 'disrupt': Maximize L2 distance between adjacent slices (encourage difference)
      - 'periodic': Encourage periodic patterns along depth axis

    Additional options:
      - 'gradient_aware': Weight by inverse image gradient
      - 'roi_aware': Only compute in ROI regions
    """
    def __init__(
        self,
        consistency_mode: str = "smooth",  # "smooth", "disrupt", "periodic"
        consistency_type: str = "l2",       # "l2", "gradient_aware"
        roi_aware: bool = True,
        bidirectional: bool = True,
        depth_dim: int = 2,  # Assuming [B, C, D, H, W]
        periodic_wavelength: int = 4,  # For periodic mode: wavelength in slices
    ):
        super().__init__()
        self.consistency_mode = consistency_mode
        self.consistency_type = consistency_type
        self.roi_aware = roi_aware
        self.bidirectional = bidirectional
        self.depth_dim = depth_dim
        self.periodic_wavelength = periodic_wavelength

    def forward(
        self,
        delta: torch.Tensor,
        image: Optional[torch.Tensor] = None,
        label: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute inter-slice consistency loss.

        Args:
            delta: Noise tensor [B, C, D, H, W]
            image: Original image (for gradient-aware mode) [B, C, D, H, W]
            label: Segmentation label (for ROI-aware mode) [B, D, H, W]

        Returns:
            Scalar consistency loss (always positive)
        """
        if self.consistency_mode == "periodic":
            return self._periodic_loss(delta, label)
        elif self.consistency_type == "gradient_aware":
            return self._gradient_aware_consistency(delta, image, label)
        else:
            return self._l2_consistency(delta, label)

    def _l2_consistency(
        self,
        delta: torch.Tensor,
        label: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Basic L2 consistency loss between adjacent slices.
        Returns the mean squared difference (always positive).
        """
        d = self.depth_dim

        # Forward difference: delta[d+1] - delta[d]
        delta_fwd = torch.diff(delta, n=1, dim=d)  # [B, C, D-1, H, W]
        diff_sq = delta_fwd ** 2

        if self.bidirectional:
            # Bidirectional: weight factor 2.0
            diff_sq = diff_sq * 2.0

        if self.roi_aware and label is not None:
            roi_mask = self._create_roi_mask(label, diff_sq.shape)
            diff_sq = diff_sq * roi_mask

        return diff_sq.mean()

    def _gradient_aware_consistency(
        self,
        delta: torch.Tensor,
        image: Optional[torch.Tensor] = None,
        label: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Gradient-aware consistency loss.
        Allows larger noise differences where image has larger gradients.
        """
        d = self.depth_dim
        delta_diff = torch.diff(delta, n=1, dim=d)

        if image is not None:
            img_grad = torch.diff(image, n=1, dim=d).abs()
            weight = 1.0 / (img_grad + 1e-4)
            weight = weight / (weight.mean() + 1e-8)
            diff_sq = weight * (delta_diff ** 2)
        else:
            diff_sq = delta_diff ** 2

        if self.roi_aware and label is not None:
            roi_mask = self._create_roi_mask(label, diff_sq.shape)
            diff_sq = diff_sq * roi_mask

        return diff_sq.mean()

    def _periodic_loss(
        self,
        delta: torch.Tensor,
        label: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Periodic consistency loss.
        Encourages noise to follow a periodic pattern along depth axis.

        Target pattern: sin(2*pi*d / wavelength)
        Loss = MSE between actual slice correlation and target periodic pattern
        """
        d = self.depth_dim
        D = delta.shape[d]  # Depth dimension size
        device = delta.device

        # Generate target periodic pattern for slice differences
        # We want: diff[i] should correlate with sin pattern
        z = torch.arange(D - 1, device=device, dtype=delta.dtype)
        target_pattern = torch.sin(2 * np.pi * z / self.periodic_wavelength)
        # Reshape for broadcasting: [1, 1, D-1, 1, 1]
        target_pattern = target_pattern.view(1, 1, -1, 1, 1)

        # Compute actual slice differences
        delta_diff = torch.diff(delta, n=1, dim=d)  # [B, C, D-1, H, W]

        # Compute mean difference per slice (reduce H, W)
        mean_diff = delta_diff.mean(dim=[-2, -1], keepdim=True)  # [B, C, D-1, 1, 1]

        # Normalize to [-1, 1] range for comparison
        mean_diff_norm = mean_diff / (mean_diff.abs().max() + 1e-8)

        # MSE between actual pattern and target periodic pattern
        periodic_loss = ((mean_diff_norm - target_pattern) ** 2).mean()

        return periodic_loss

    def _create_roi_mask(
        self,
        label: torch.Tensor,
        target_shape: tuple,
    ) -> torch.Tensor:
        """
        Create ROI mask from label tensor.
        ROI = label > 0 (foreground regions)
        """
        if label.dim() == 5:
            label = label.squeeze(1)

        mask = (label > 0).float()
        mask = mask.unsqueeze(1)

        # Adjust depth dimension for diff operation (D -> D-1)
        if mask.shape[2] > target_shape[2]:
            mask = mask[:, :, :-1, :, :]

        mask = mask.expand(-1, target_shape[1], -1, -1, -1)
        return mask.to(label.device)


@register_plugin("unet_noise_slice")
class UNetSliceConsistencyUE:
    """
    Inter-Slice Consistency Attack for 3D Medical Image Segmentation.

    This method generates noise with inter-slice consistency constraints,
    specifically designed to exploit the 3D structure of medical volumes.

    Assumptions:
      - Batch:
          batch["image"]: FloatTensor [B, C, D, H, W] (3D volume)
          batch["label"]: LongTensor  [B, D, H, W]
          batch["key"]:   sample-wise key
      - Surrogate:
          s_model(x) -> logits: [B, C_seg, D, H, W]
      - Noise backend:
          noise_backend.batch_noise(keys) -> [N, C_in, D, H, W]

    Additional components:
      - noise_unet: U-Net that generates noise from input images
      - opt_noise_unet: Optimizer for the noise U-Net
      - consistency_loss: Inter-slice consistency loss module
    """

    def __init__(self):
        self._seg_loss: DiceCELoss | None = None
        self._noise_unet: NoiseUNetWrapper | None = None
        self._opt_noise_unet: torch.optim.Optimizer | None = None
        self._consistency_loss: SliceConsistencyLoss | None = None
        self._noise_unet_device: torch.device | None = None
        self._initialized: bool = False
        self._consistency_mode: str = "smooth"  # "smooth", "disrupt", "periodic"
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
        if label.dim() == 5:
            label = label.squeeze(1)
        mask = (label > 0).float()
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
        Initialize the noise U-Net, its optimizer, and consistency loss (lazy, once).
        """
        if self._initialized:
            return

        cfg = trainer.config
        device = trainer.device

        # Get noise UNet config
        noise_unet_cfg = get_config(cfg, "ue.noise_unet", DictConfig({}))
        params = get_config(cfg, "ue.algorithm.params", DictConfig({}))
        eps = float(get_config(params, "epsilon", 8/255))

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

        # Build consistency loss
        consistency_mode = str(get_config(params, "consistency_mode", "smooth"))
        consistency_type = str(get_config(params, "consistency_type", "l2"))
        roi_aware = bool(get_config(params, "roi_aware", True))
        bidirectional = bool(get_config(params, "bidirectional", True))
        periodic_wavelength = int(get_config(params, "periodic_wavelength", 4))

        self._consistency_mode = consistency_mode
        self._consistency_loss = SliceConsistencyLoss(
            consistency_mode=consistency_mode,
            consistency_type=consistency_type,
            roi_aware=roi_aware,
            bidirectional=bidirectional,
            depth_dim=2,  # [B, C, D, H, W]
            periodic_wavelength=periodic_wavelength,
        )

        self._initialized = True
        self.logger.info(
            f"[UNetSliceConsistency] Initialized noise UNet: in_channels={in_channels}, "
            f"spatial_dims={spatial_dims}, eps={eps:.6f}, lr={lr}, "
            f"consistency_mode={consistency_mode}, consistency_type={consistency_type}, "
            f"roi_aware={roi_aware}"
        )

    # ---------------- Surrogate-step: Update surrogate ---------------- #
    def surrogate_step_batch(self, trainer, batch) -> Dict[str, float]:
        """
        Update surrogate parameters only, do not update noise.
        Uses generated noise from noise U-Net.
        """
        cfg = trainer.config
        device = trainer.device
        nb = trainer.noise_backend
        if nb is None:
            raise RuntimeError("[UE] noise_backend is required.")

        # data
        x = batch["image"].to(device).float()  # [B, C, D, H, W]
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

        # Get noise from backend
        delta = nb.batch_noise(list(keys)).to(device).float()
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

    # ---------------- N-step: Update noise via UNet with slice consistency ---------------- #
    def noise_step_batch(self, trainer, batch) -> Dict[str, float]:
        """
        Update noise using trainable U-Net with inter-slice consistency constraints.

        Process:
          1. Forward image through noise U-Net to get noise prediction
          2. Compute inter-slice consistency loss along depth dimension
          3. Optionally apply ROI mask (only noise in ROI regions)
          4. Add noise to image, normalize, pass through frozen surrogate
          5. Compute segmentation loss
          6. Total loss = seg_loss + lambda_consistency * consistency_loss
          7. Backprop to update noise U-Net parameters
          8. Store generated noise to backend
        """
        cfg = trainer.config
        device = trainer.device
        nb = trainer.noise_backend
        if nb is None:
            raise RuntimeError("[UE] noise_backend is required.")

        # -------- data & config --------
        x = batch["image"].to(device).float()  # [B, C_in, D, H, W]
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
        num_steps = int(get_config(params, "noise_step", 1))
        lambda_consistency = float(get_config(params, "lambda_consistency", 0.5))
        roi_aware = bool(get_config(params, "roi_aware", True))

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

        # -------- create ROI mask if enabled --------
        roi_mask = None
        if roi_aware:
            roi_mask = self._create_roi_mask(y, x.shape[2:])
            roi_mask = roi_mask.expand(-1, C_in, -1, -1, -1).to(device)

        # -------- train noise UNet --------
        self._noise_unet.train()
        last_seg_loss = torch.tensor(0.0, device=device)
        last_consistency_loss = torch.tensor(0.0, device=device)

        for _ in range(max(1, num_steps)):
            # Generate noise from UNet
            delta_raw = self._noise_unet(x)  # [B, C_in, D, H, W], in [-eps, eps]

            # Compute inter-slice consistency loss (before ROI masking)
            consistency_loss = self._consistency_loss(
                delta_raw,
                image=x,
                label=y if roi_aware else None,
            )
            last_consistency_loss = consistency_loss.detach()

            # Apply ROI mask if enabled
            if roi_aware and roi_mask is not None:
                delta = delta_raw * roi_mask
            else:
                delta = delta_raw

            # Create perturbed image
            perturb_img = (x + delta).clamp(0.0, 1.0)
            xn = perturb_img.clone()
            self._norm_inplace(xn, mean, std)

            # Forward through surrogate
            out = s_model(xn)
            logits = out[0] if isinstance(out, (tuple, list)) else out

            # Compute segmentation loss (min-min: minimize loss)
            seg_loss = seg_loss_fn(logits, y.unsqueeze(1))
            last_seg_loss = seg_loss.detach()

            # Total loss depends on consistency mode:
            #   - smooth:   seg_loss + lambda * consistency_loss (minimize difference)
            #   - disrupt:  seg_loss - lambda * consistency_loss (maximize difference)
            #   - periodic: seg_loss + lambda * consistency_loss (match periodic pattern)
            if self._consistency_mode == "disrupt":
                total_loss = seg_loss - lambda_consistency * consistency_loss
            else:  # "smooth" or "periodic"
                total_loss = seg_loss + lambda_consistency * consistency_loss

            # Backprop to update noise UNet
            self._opt_noise_unet.zero_grad(set_to_none=True)
            total_loss.backward()
            self._opt_noise_unet.step()

        # -------- Store generated noise to backend --------
        self._noise_unet.eval()
        with torch.no_grad():
            final_delta_raw = self._noise_unet(x)

            # Apply ROI mask if enabled
            if roi_aware and roi_mask is not None:
                final_delta = final_delta_raw * roi_mask
            else:
                final_delta = final_delta_raw

            # Extra clamp for safety
            final_delta = final_delta.clamp(-eps, eps)

        nb.commit_batch(keys_list, final_delta.detach().cpu())

        delta_linf = float(final_delta.detach().abs().max().cpu())
        return {
            "noise_loss": float(last_seg_loss.cpu()),
            "consistency_loss": float(last_consistency_loss.cpu()),
            "delta_linf": delta_linf,
        }