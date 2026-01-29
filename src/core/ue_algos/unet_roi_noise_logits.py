# file: src/core/ue_algos/unet_roi_noise_logits.py
"""
UNet ROI Noise Generator with Logits Divergence Loss.

Extension of unet_roi_noise that adds a logits divergence loss to
maximize the difference between predictions on clean vs noisy images.

Key difference from unet_roi_noise:
  - unet_roi_noise: only seg_loss on noisy image
  - unet_roi_noise_logits: seg_loss + logits_divergence_loss

Logits Divergence Modes:
  - 'l1': Direct L1 norm of logits difference
  - 'l2': Direct L2 norm of logits difference
  - 'fft_l1': FFT of logits difference, then L1 norm (ortho-normalized)
  - 'fft_l2': FFT of logits difference, then L2 norm (ortho-normalized)
  - 'kl_div': KL divergence between softmax distributions
"""
from __future__ import annotations
from typing import Dict, Iterable, List, Tuple

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


class LogitsDivergenceLoss(nn.Module):
    """
    Compute logits divergence loss between clean and noisy predictions.

    Supports multiple divergence computation modes:
      - 'l1': Direct L1 norm of logits difference
      - 'l2': Direct L2 norm of logits difference
      - 'fft_l1': FFT of logits difference, then L1 norm (ortho-normalized)
      - 'fft_l2': FFT of logits difference, then L2 norm (ortho-normalized)
      - 'kl_div': KL divergence between softmax distributions

    The loss is negated to maximize divergence (since we minimize loss).
    """

    def __init__(
        self,
        mode: str = 'fft_l1',
        weight: float = 1.0,
        temperature: float = 1.0,
        fft_dims: Tuple[int, ...] = (-3, -2, -1),
    ):
        super().__init__()
        self.mode = mode.lower()
        self.weight = weight
        self.temperature = temperature
        self.fft_dims = fft_dims

        valid_modes = {'l1', 'l2', 'fft_l1', 'fft_l2', 'kl_div'}
        if self.mode not in valid_modes:
            raise ValueError(f"Invalid mode '{mode}'. Must be one of {valid_modes}")

    def forward(
        self,
        logits_clean: torch.Tensor,
        logits_noisy: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute divergence loss.

        Args:
            logits_clean: [B, C, D, H, W] predictions on clean images
            logits_noisy: [B, C, D, H, W] predictions on noisy images

        Returns:
            loss: Scalar tensor (negative divergence for maximization)
        """
        diff = logits_noisy - logits_clean

        if self.mode == 'l1':
            divergence = diff.abs().mean()

        elif self.mode == 'l2':
            divergence = (diff ** 2).mean().sqrt()

        elif self.mode == 'fft_l1':
            # ortho normalization: magnitude stays comparable to input
            diff_fft = torch.fft.fftn(diff, dim=self.fft_dims, norm="ortho")
            divergence = diff_fft.abs().mean()

        elif self.mode == 'fft_l2':
            diff_fft = torch.fft.fftn(diff, dim=self.fft_dims, norm="ortho")
            divergence = (diff_fft.abs() ** 2).mean().sqrt()

        elif self.mode == 'kl_div':
            logits_clean_scaled = logits_clean / self.temperature
            logits_noisy_scaled = logits_noisy / self.temperature

            prob_clean = F.softmax(logits_clean_scaled, dim=1)
            log_prob_noisy = F.log_softmax(logits_noisy_scaled, dim=1)

            kl_loss = F.kl_div(
                log_prob_noisy,
                prob_clean,
                reduction='batchmean',
                log_target=False
            )
            divergence = kl_loss

        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        # Negative: maximize divergence by minimizing loss
        return -self.weight * divergence


@register_plugin("unet_roi_noise_logits")
class UNetROINoiseLogitsUE:
    """
    UNet-based ROI noise generation with Logits Divergence Loss.

    Extension of UNetROINoiseUE:
      - Generates noise only within ROI (label > 0)
      - Adds logits divergence loss to maximize prediction difference

    Configuration (ue.algorithm.params):
      epsilon: L_inf bound (default: 8/255)
      noise_step: UNet training iterations per batch (default: 1)

      # Logits Divergence Loss
      logits_div_enabled: Enable logits divergence loss (default: True)
      logits_div_mode: l1, l2, fft_l1, fft_l2, kl_div (default: fft_l1)
      logits_div_weight: Weight for divergence loss (default: 1.0)
      logits_div_temperature: Temperature for KL div mode (default: 1.0)
    """

    def __init__(self):
        self._seg_loss: DiceCELoss | None = None
        self._noise_unet: NoiseUNetWrapper | None = None
        self._opt_noise_unet: torch.optim.Optimizer | None = None
        self._logits_div_loss: LogitsDivergenceLoss | None = None
        self._logits_div_enabled: bool = True
        self._noise_unet_device: torch.device | None = None
        self._initialized: bool = False
        self.logger = get_logger()

    @staticmethod
    def _norm_inplace(x: torch.Tensor, mean, std):
        """In-place per-channel normalize for ND volume."""
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
        """Build DiceCELoss consistent with SegTrainer configuration."""
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

    def _init_components(self, trainer, in_channels: int, spatial_dims: int = 3):
        """Initialize all components (lazy, once)."""
        if self._initialized:
            return

        cfg = trainer.config
        device = trainer.device

        # Noise UNet
        noise_unet_cfg = get_config(cfg, "ue.noise_unet", DictConfig({}))
        eps = float(get_config(cfg, "ue.algorithm.params.epsilon", 8/255))

        base_unet = _build_noise_unet(noise_unet_cfg, in_channels, spatial_dims)
        self._noise_unet = NoiseUNetWrapper(base_unet, epsilon=eps)
        self._noise_unet = self._noise_unet.to(device)
        self._noise_unet_device = device

        # Optimizer
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

        # Logits divergence loss
        params = get_config(cfg, "ue.algorithm.params", DictConfig({}))
        self._logits_div_enabled = bool(get_config(params, "logits_div_enabled", True))
        if self._logits_div_enabled:
            logits_div_mode = str(get_config(params, "logits_div_mode", "fft_l1"))
            logits_div_weight = float(get_config(params, "logits_div_weight", 1.0))
            logits_div_temperature = float(get_config(params, "logits_div_temperature", 1.0))

            self._logits_div_loss = LogitsDivergenceLoss(
                mode=logits_div_mode,
                weight=logits_div_weight,
                temperature=logits_div_temperature,
            )
        else:
            self._logits_div_loss = None
            logits_div_mode = "disabled"
            logits_div_weight = 0.0

        self._initialized = True
        self.logger.info(
            f"[UNetROINoiseLogits] Initialized: in_channels={in_channels}, "
            f"spatial_dims={spatial_dims}, eps={eps:.6f}, lr={lr}, "
            f"logits_div_enabled={self._logits_div_enabled}, "
            f"logits_div_mode={logits_div_mode}, logits_div_weight={logits_div_weight}"
        )

    # ---------------- Surrogate-step: Update surrogate ---------------- #
    def surrogate_step_batch(self, trainer, batch) -> Dict[str, float]:
        """Update surrogate parameters only, using noise from backend."""
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

        self._init_components(trainer, C_in, spatial_dims)

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

    # ---------------- N-step: Update noise via UNet (ROI only) + Logits Divergence ---------------- #
    def noise_step_batch(self, trainer, batch) -> Dict[str, float]:
        """
        Update noise using trainable U-Net within ROI + logits divergence.

        Process:
          1. Forward image through noise U-Net to get noise prediction
          2. Create ROI mask from label (ROI = label > 0)
          3. Apply ROI mask to noise (background noise = 0)
          4. Get clean image predictions (for logits divergence)
          5. Add masked noise to image, pass through frozen surrogate
          6. Compute combined loss: seg_loss + logits_divergence_loss
          7. Backprop to update noise U-Net parameters
          8. Store masked noise to backend
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

        self._init_components(trainer, C_in, spatial_dims)

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
        roi_mask = self._create_roi_mask(y, x.shape[2:])
        roi_mask = roi_mask.expand(-1, C_in, -1, -1, -1).to(device)

        # -------- get clean predictions (for logits divergence) --------
        logits_clean = None
        if self._logits_div_enabled and self._logits_div_loss is not None:
            with torch.no_grad():
                x_clean_norm = x.clone()
                self._norm_inplace(x_clean_norm, mean, std)
                out_clean = s_model(x_clean_norm)
                logits_clean = out_clean[0] if isinstance(out_clean, (tuple, list)) else out_clean
                logits_clean = logits_clean.detach()

        # -------- train noise UNet --------
        self._noise_unet.train()
        last_loss = torch.tensor(0.0, device=device)
        last_seg_loss = torch.tensor(0.0, device=device)
        last_div_loss = torch.tensor(0.0, device=device)

        for _ in range(max(1, num_steps)):
            # Generate noise from UNet
            delta_raw = self._noise_unet(x)

            # Apply ROI mask: only keep noise in ROI regions
            delta = delta_raw * roi_mask

            # Create perturbed image
            perturb_img = (x + delta).clamp(0.0, 1.0)
            xn = perturb_img.clone()
            self._norm_inplace(xn, mean, std)

            # Forward through surrogate (noisy)
            out = s_model(xn)
            logits_noisy = out[0] if isinstance(out, (tuple, list)) else out

            # Compute combined loss
            seg_loss = seg_loss_fn(logits_noisy, y.unsqueeze(1))
            last_seg_loss = seg_loss.detach()

            # Logits divergence loss
            if self._logits_div_enabled and self._logits_div_loss is not None and logits_clean is not None:
                div_loss = self._logits_div_loss(logits_clean, logits_noisy)
                last_div_loss = div_loss.detach()
                loss = seg_loss + div_loss
            else:
                loss = seg_loss

            last_loss = loss.detach()

            # Backprop to update noise UNet
            self._opt_noise_unet.zero_grad(set_to_none=True)
            loss.backward()
            self._opt_noise_unet.step()

        # -------- Store masked noise to backend --------
        self._noise_unet.eval()
        with torch.no_grad():
            final_delta_raw = self._noise_unet(x)
            final_delta = final_delta_raw * roi_mask
            final_delta = final_delta.clamp(-eps, eps)

        nb.commit_batch(keys_list, final_delta.detach().cpu())

        delta_linf = float(final_delta.detach().abs().max().cpu())

        # Compute final logits divergence for logging
        logits_diff_l1 = 0.0
        if logits_clean is not None:
            with torch.no_grad():
                perturb_final = (x + final_delta).clamp(0.0, 1.0)
                xn_final = perturb_final.clone()
                self._norm_inplace(xn_final, mean, std)
                out_final = s_model(xn_final)
                logits_final = out_final[0] if isinstance(out_final, (tuple, list)) else out_final
                logits_diff_l1 = (logits_final - logits_clean).abs().mean().cpu().item()

        return {
            "noise_loss": float(last_loss.cpu()),
            "seg_loss": float(last_seg_loss.cpu()),
            "div_loss": float(last_div_loss.cpu()),
            "logits_diff_l1": logits_diff_l1,
            "delta_linf": delta_linf,
        }
