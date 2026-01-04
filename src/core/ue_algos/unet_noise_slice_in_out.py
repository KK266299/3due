# file: src/core/ue_algos/unet_noise_slice_in_out.py
"""
Anisotropy-Aware UNet Noise Generator for 3D Medical Image Segmentation.

This algorithm exploits the anisotropic nature of medical imaging:
  - Intra-slice (x-y): High resolution → Noise should be SMOOTH within each slice
  - Inter-slice (z):   Low resolution  → Noise can be DIFFERENT between slices

Key Innovation:
  - Intra-Slice Smoothness Loss: TV/L2 loss to encourage smooth noise within each slice
  - Inter-Slice Diversity Loss: Maximize L2 difference between adjacent slices
  - ROI-Aware: Optional focusing on regions of interest (foreground)

Core differences from UNetNoise:
  - UNetNoise: Only segmentation loss
  - This method: seg_loss + λ_intra * intra_smooth - λ_inter * inter_diff

Training loop:
  1. Surrogate-step: Same as MinMin, train surrogate on noisy images
  2. Noise-step:
     - Forward original image through noise U-Net to get noise prediction
     - Compute intra-slice smoothness loss (minimize: smooth within slice)
     - Compute inter-slice diversity loss (maximize: different between slices)
     - Add noise to image, pass through surrogate, compute seg loss
     - Total loss = seg_loss + λ_intra * intra_loss - λ_inter * inter_loss
     - Backprop to update noise U-Net parameters
     - Store generated noise to backend

Maximum noise bound: 8/255 ≈ 0.0313725 (L∞ constraint)
"""
from __future__ import annotations
from typing import Dict, Iterable, List, Optional

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


class IntraSliceSmoothLoss(nn.Module):
    """
    Intra-Slice Smoothness Loss.

    Encourages noise to be smooth WITHIN each slice (x-y plane),
    exploiting the high intra-slice resolution of medical images.

    Supports:
      - 'tv': Total Variation (L1 of gradients)
      - 'l2': L2 smoothness (squared gradients)
    """
    def __init__(
        self,
        smooth_type: str = "tv",
        roi_aware: bool = True,
    ):
        super().__init__()
        self.smooth_type = smooth_type
        self.roi_aware = roi_aware

    def forward(
        self,
        delta: torch.Tensor,
        label: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            delta: Noise tensor [B, C, D, H, W]
            label: Segmentation label [B, D, H, W] (for ROI mask)
        Returns:
            Scalar smoothness loss (to be minimized)
        """
        # Compute gradients in H and W directions (within each slice)
        # delta shape: [B, C, D, H, W]
        diff_h = delta[:, :, :, 1:, :] - delta[:, :, :, :-1, :]  # [B, C, D, H-1, W]
        diff_w = delta[:, :, :, :, 1:] - delta[:, :, :, :, :-1]  # [B, C, D, H, W-1]

        if self.smooth_type == "tv":
            # Total Variation: L1 norm
            loss_h = torch.abs(diff_h)
            loss_w = torch.abs(diff_w)
        else:  # "l2"
            # L2 smoothness
            loss_h = diff_h ** 2
            loss_w = diff_w ** 2

        # Apply ROI mask if enabled - only average over ROI pixels
        if self.roi_aware and label is not None:
            roi_mask_h = self._create_roi_mask_h(label, loss_h.shape)
            roi_mask_w = self._create_roi_mask_w(label, loss_w.shape)

            # Sum over ROI only, divide by ROI pixel count (avoid div by zero)
            sum_h = (loss_h * roi_mask_h).sum()
            sum_w = (loss_w * roi_mask_w).sum()
            count_h = roi_mask_h.sum().clamp(min=1.0)
            count_w = roi_mask_w.sum().clamp(min=1.0)

            return sum_h / count_h + sum_w / count_w
        else:
            return loss_h.mean() + loss_w.mean()

    def _create_roi_mask_h(self, label: torch.Tensor, target_shape: tuple) -> torch.Tensor:
        """Create ROI mask for H-direction differences."""
        if label.dim() == 5:
            label = label.squeeze(1)
        mask = (label > 0).float()
        # Adjust for diff: use minimum of adjacent pixels
        mask = torch.minimum(mask[:, :, 1:, :], mask[:, :, :-1, :])
        mask = mask.unsqueeze(1).expand(-1, target_shape[1], -1, -1, -1)
        return mask.to(label.device)

    def _create_roi_mask_w(self, label: torch.Tensor, target_shape: tuple) -> torch.Tensor:
        """Create ROI mask for W-direction differences."""
        if label.dim() == 5:
            label = label.squeeze(1)
        mask = (label > 0).float()
        mask = torch.minimum(mask[:, :, :, 1:], mask[:, :, :, :-1])
        mask = mask.unsqueeze(1).expand(-1, target_shape[1], -1, -1, -1)
        return mask.to(label.device)


class InterSliceDiversityLoss(nn.Module):
    """
    Inter-Slice Diversity Loss.

    Encourages noise to be DIFFERENT between adjacent slices,
    exploiting the low inter-slice resolution of medical images.

    The loss computes L2 difference between adjacent slices.
    This loss is MAXIMIZED (used with negative weight).
    """
    def __init__(
        self,
        roi_aware: bool = True,
        depth_dim: int = 2,  # [B, C, D, H, W]
    ):
        super().__init__()
        self.roi_aware = roi_aware
        self.depth_dim = depth_dim

    def forward(
        self,
        delta: torch.Tensor,
        label: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            delta: Noise tensor [B, C, D, H, W]
            label: Segmentation label [B, D, H, W] (for ROI mask)
        Returns:
            Scalar diversity loss (to be MAXIMIZED, use with negative weight)
        """
        d = self.depth_dim
        # Compute difference between adjacent slices
        delta_diff = torch.diff(delta, n=1, dim=d)  # [B, C, D-1, H, W]
        diff_sq = delta_diff ** 2

        # Apply ROI mask if enabled - only average over ROI pixels
        if self.roi_aware and label is not None:
            roi_mask = self._create_roi_mask(label, diff_sq.shape)

            # Sum over ROI only, divide by ROI pixel count (avoid div by zero)
            masked_sum = (diff_sq * roi_mask).sum()
            count = roi_mask.sum().clamp(min=1.0)

            return masked_sum / count
        else:
            return diff_sq.mean()

    def _create_roi_mask(self, label: torch.Tensor, target_shape: tuple) -> torch.Tensor:
        """Create ROI mask from label tensor."""
        if label.dim() == 5:
            label = label.squeeze(1)
        mask = (label > 0).float()
        mask = mask.unsqueeze(1)
        # Adjust depth dimension for diff operation (D -> D-1)
        if mask.shape[2] > target_shape[2]:
            # Use minimum of adjacent slices
            mask = torch.minimum(mask[:, :, 1:, :, :], mask[:, :, :-1, :, :])
        mask = mask.expand(-1, target_shape[1], -1, -1, -1)
        return mask.to(label.device)


@register_plugin("unet_noise_slice_in_out")
class UNetNoiseSliceInOutUE:
    """
    Anisotropy-Aware UNet Noise Generator.

    Exploits medical image anisotropy:
      - Intra-slice: High resolution → SMOOTH noise (minimize TV/L2)
      - Inter-slice: Low resolution  → DIVERSE noise (maximize L2 diff)

    Assumptions (same as MinMin):
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
      - intra_smooth_loss: Intra-slice smoothness loss
      - inter_diversity_loss: Inter-slice diversity loss
    """

    def __init__(self):
        self._seg_loss: DiceCELoss | None = None
        self._noise_unet: NoiseUNetWrapper | None = None
        self._opt_noise_unet: torch.optim.Optimizer | None = None
        self._intra_smooth_loss: IntraSliceSmoothLoss | None = None
        self._inter_diversity_loss: InterSliceDiversityLoss | None = None
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
        Initialize the noise U-Net, optimizer, and anisotropy losses (lazy, once).
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

        # Build anisotropy losses
        intra_smooth_type = str(get_config(params, "intra_smooth_type", "tv"))
        roi_aware = bool(get_config(params, "roi_aware", True))

        self._intra_smooth_loss = IntraSliceSmoothLoss(
            smooth_type=intra_smooth_type,
            roi_aware=roi_aware,
        )

        self._inter_diversity_loss = InterSliceDiversityLoss(
            roi_aware=roi_aware,
            depth_dim=2,  # [B, C, D, H, W]
        )

        self._initialized = True
        self.logger.info(
            f"[UNetSliceInOut] Initialized: in_ch={in_channels}, spatial_dims={spatial_dims}, "
            f"eps={eps:.6f}, lr={lr}, intra_type={intra_smooth_type}, roi_aware={roi_aware}"
        )

    # ---------------- Surrogate-step: Update surrogate ---------------- #
    def surrogate_step_batch(self, trainer, batch) -> Dict[str, float]:
        """
        Update surrogate parameters only, do not update noise.
        Uses stored noise from noise backend.
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
        spatial_dims = len(x.shape) - 2

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

    # ---------------- N-step: Update noise via UNet with anisotropy constraints ---------------- #
    def noise_step_batch(self, trainer, batch) -> Dict[str, float]:
        """
        Update noise using trainable U-Net with anisotropy constraints.

        Loss = seg_loss + λ_intra * intra_smooth_loss - λ_inter * inter_diversity_loss

        Process:
          1. Forward image through noise U-Net to get noise prediction
          2. Compute intra-slice smoothness loss (minimize)
          3. Compute inter-slice diversity loss (maximize via negative weight)
          4. Optionally apply ROI mask
          5. Add noise to image, normalize, pass through frozen surrogate
          6. Compute segmentation loss
          7. Combine losses and backprop to update noise U-Net
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
        lambda_intra = float(get_config(params, "lambda_intra", 0.3))
        lambda_inter = float(get_config(params, "lambda_inter", 0.5))
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
        last_intra_loss = torch.tensor(0.0, device=device)
        last_inter_loss = torch.tensor(0.0, device=device)

        for _ in range(max(1, num_steps)):
            # Generate noise from UNet
            delta_raw = self._noise_unet(x)  # [B, C_in, D, H, W], in [-eps, eps]

            # Compute intra-slice smoothness loss (to be minimized)
            intra_loss = self._intra_smooth_loss(delta_raw, label=y if roi_aware else None)
            last_intra_loss = intra_loss.detach()

            # Compute inter-slice diversity loss (to be maximized)
            inter_loss = self._inter_diversity_loss(delta_raw, label=y if roi_aware else None)
            last_inter_loss = inter_loss.detach()

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

            # Total loss:
            #   - seg_loss: minimize (make surrogate fit noisy data)
            #   - intra_loss: minimize (smooth within slices)
            #   - inter_loss: maximize (different between slices) → use negative weight
            total_loss = seg_loss + lambda_intra * intra_loss - lambda_inter * inter_loss

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
            "intra_loss": float(last_intra_loss.cpu()),
            "inter_loss": float(last_inter_loss.cpu()),
            "delta_linf": delta_linf,
        }