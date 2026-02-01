# file: src/core/ue_algos/noise_slice_frequence_learnable.py
"""
Frequency-Domain Constrained Noise with Global Learnable Cutoff Frequencies.

Key difference from noise_slice_frequence_h_l_pass:
  - Cutoff frequencies (z_cutoff, xy_cutoff) are TWO global nn.Parameter scalars
    shared across the entire dataset, NOT per-sample predictions.
  - Updated via gradient descent alongside the NoiseUNet.
  - Each epoch logs the current cutoff values.

Architecture:
  Image x ──→ NoiseUNet ──→ noise δ
                               │
  global (z_cutoff, xy_cutoff) ─┤
                               ▼
                  FreqConstraint(δ, z_c, xy_c) ──→ filtered δ
                                                       │
                                                  DiceCE Loss
                                                  ╱         ╲
                                          opt_unet.step()  opt_cutoff.step()
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


# ────────────────────────── Helper modules ────────────────────────── #

def _build_noise_unet(cfg: DictConfig, in_channels: int, spatial_dims: int = 3) -> nn.Module:
    channels = list(get_config(cfg, "channels", [16, 32, 64, 128]))
    strides = list(get_config(cfg, "strides", [2, 2, 2]))
    num_res_units = int(get_config(cfg, "num_res_units", 1))
    act = get_config(cfg, "act", "LEAKYRELU")
    norm = get_config(cfg, "norm", "INSTANCE")
    dropout = float(get_config(cfg, "dropout", 0.0))

    return MonaiUNet(
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


class NoiseUNetWrapper(nn.Module):
    def __init__(self, unet: nn.Module, epsilon: float = 8 / 255):
        super().__init__()
        self.unet = unet
        self.epsilon = epsilon

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.unet(x)) * self.epsilon


class GlobalLearnableCutoff(nn.Module):
    """
    Two global scalar parameters: z_cutoff and xy_cutoff.

    Stored as logits (unconstrained), mapped via sigmoid → range to get
    actual cutoff values in [range_low, range_high].
    """

    def __init__(
        self,
        z_cutoff_init: float = 0.1,
        xy_cutoff_init: float = 0.3,
        z_range: Tuple[float, float] = (0.01, 0.45),
        xy_range: Tuple[float, float] = (0.05, 0.45),
    ):
        super().__init__()
        self.z_range = z_range
        self.xy_range = xy_range

        # Convert init values to logit space
        z_t = (z_cutoff_init - z_range[0]) / (z_range[1] - z_range[0])
        xy_t = (xy_cutoff_init - xy_range[0]) / (xy_range[1] - xy_range[0])
        self.z_logit = nn.Parameter(torch.tensor(
            float(torch.logit(torch.tensor(z_t).clamp(0.01, 0.99)))
        ))
        self.xy_logit = nn.Parameter(torch.tensor(
            float(torch.logit(torch.tensor(xy_t).clamp(0.01, 0.99)))
        ))

    def forward(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (z_cutoff, xy_cutoff) as differentiable scalars."""
        z = torch.sigmoid(self.z_logit) * (self.z_range[1] - self.z_range[0]) + self.z_range[0]
        xy = torch.sigmoid(self.xy_logit) * (self.xy_range[1] - self.xy_range[0]) + self.xy_range[0]
        return z, xy


class FrequencyDomainConstraint(nn.Module):
    """
    Apply frequency domain constraints using differentiable sigmoid masks.

    Supports:
      - Static mode: fixed cutoffs (torch.where, cached)
      - Learnable mode: scalar cutoffs with sigmoid masks (differentiable)
    """

    def __init__(
        self,
        z_cutoff_low: float = 0.1,
        z_sigma: float = 0.05,
        xy_cutoff_high: float = 0.3,
        xy_sigma: float = 0.1,
    ):
        super().__init__()
        self.z_cutoff_low = z_cutoff_low
        self.z_sigma = z_sigma
        self.xy_cutoff_high = xy_cutoff_high
        self.xy_sigma = xy_sigma
        self._cached_mask = None
        self._cached_shape = None

    def _get_freq_grids(
        self, D: int, H: int, W: int, device: torch.device, dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        freq_z = torch.fft.fftfreq(D, device=device, dtype=dtype)
        freq_y = torch.fft.fftfreq(H, device=device, dtype=dtype)
        freq_x = torch.fft.fftfreq(W, device=device, dtype=dtype)
        abs_freq_z = freq_z.abs()  # [D]
        r_xy = torch.sqrt(freq_y.unsqueeze(1) ** 2 + freq_x.unsqueeze(0) ** 2)  # [H, W]
        return abs_freq_z, r_xy

    def _build_static_mask(
        self, D: int, H: int, W: int, device: torch.device, dtype: torch.dtype,
    ) -> torch.Tensor:
        abs_freq_z, r_xy = self._get_freq_grids(D, H, W, device, dtype)
        k_z, k_y, k_x = torch.meshgrid(
            torch.fft.fftfreq(D, device=device, dtype=dtype),
            torch.fft.fftfreq(H, device=device, dtype=dtype),
            torch.fft.fftfreq(W, device=device, dtype=dtype),
            indexing='ij',
        )
        abs_k_z = k_z.abs()
        r = torch.sqrt(k_y ** 2 + k_x ** 2)

        M_z = torch.where(
            abs_k_z >= self.z_cutoff_low,
            torch.ones_like(abs_k_z),
            torch.exp(-((self.z_cutoff_low - abs_k_z) ** 2) / (2 * self.z_sigma ** 2)),
        )
        M_xy = torch.where(
            r <= self.xy_cutoff_high,
            torch.ones_like(r),
            torch.exp(-((r - self.xy_cutoff_high) ** 2) / (2 * self.xy_sigma ** 2)),
        )
        M = M_z * M_xy
        M[0, 0, 0] = 0.1
        return M.unsqueeze(0).unsqueeze(0)  # [1,1,D,H,W]

    def forward(
        self,
        noise: torch.Tensor,
        z_cutoff: torch.Tensor | None = None,
        xy_cutoff: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, C, D, H, W = noise.shape
        device, dtype = noise.device, noise.dtype

        noise_fft = torch.fft.fftn(noise, dim=(-3, -2, -1))

        if z_cutoff is not None and xy_cutoff is not None:
            # ---- Learnable path: sigmoid masks (fully differentiable) ----
            abs_freq_z, r_xy = self._get_freq_grids(D, H, W, device, dtype)

            # M_z: [1, 1, D, 1, 1]  (scalar cutoff → broadcast to batch)
            freq_z_5d = abs_freq_z.view(1, 1, D, 1, 1)
            M_z = torch.sigmoid((freq_z_5d - z_cutoff.view(1, 1, 1, 1, 1)) / self.z_sigma)

            # M_xy: [1, 1, 1, H, W]
            r_xy_5d = r_xy.view(1, 1, 1, H, W)
            M_xy = torch.sigmoid((xy_cutoff.view(1, 1, 1, 1, 1) - r_xy_5d) / self.xy_sigma)

            # DC correction via torch.where (no inplace)
            is_dc_z = (abs_freq_z < 1e-7).view(1, 1, D, 1, 1)
            dc_mz = torch.sigmoid(-z_cutoff / self.z_sigma)
            dc_mxy = torch.sigmoid(xy_cutoff / self.xy_sigma)
            dc_product = (dc_mz * dc_mxy).clamp_min(1e-6)
            corr = (0.1 / dc_product).view(1, 1, 1, 1, 1)
            M_z = torch.where(is_dc_z, M_z * corr, M_z)

            noise_fft = noise_fft * M_z  # [B,C,D,H,W] * [1,1,D,1,1]
            noise_fft = noise_fft * M_xy  # [B,C,D,H,W] * [1,1,1,H,W]
        else:
            # ---- Static cached path ----
            if self._cached_mask is None or self._cached_shape != (D, H, W):
                self._cached_mask = self._build_static_mask(D, H, W, device, dtype)
                self._cached_shape = (D, H, W)
            else:
                self._cached_mask = self._cached_mask.to(device=device, dtype=dtype)
            noise_fft = noise_fft * self._cached_mask.expand(B, C, -1, -1, -1)

        return torch.fft.ifftn(noise_fft, dim=(-3, -2, -1)).real


class SoftROIMask(nn.Module):
    def __init__(
        self,
        soft_edge: bool = True,
        dilate_iterations: int = 2,
        dilate_kernel_size: int = 3,
        gaussian_sigma: float = 2.0,
    ):
        super().__init__()
        self.soft_edge = soft_edge
        self.dilate_iterations = dilate_iterations
        self.dilate_kernel_size = dilate_kernel_size
        self.gaussian_sigma = gaussian_sigma
        self._gaussian_kernel = None

    def _build_gaussian_kernel(self, device, dtype):
        sigma = self.gaussian_sigma
        ks = int(6 * sigma + 1)
        if ks % 2 == 0:
            ks += 1
        coords = torch.arange(ks, device=device, dtype=dtype) - ks // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = g / g.sum()
        k3d = g.view(-1, 1, 1) * g.view(1, -1, 1) * g.view(1, 1, -1)
        k3d = k3d / k3d.sum()
        return k3d.view(1, 1, ks, ks, ks)

    def forward(self, label: torch.Tensor, num_channels: int) -> torch.Tensor:
        if label.dim() == 5:
            label = label.squeeze(1)
        mask = (label > 0).float().unsqueeze(1)

        if self.soft_edge:
            if self.dilate_iterations > 0 and self.dilate_kernel_size > 0:
                k = self.dilate_kernel_size
                p = k // 2
                for _ in range(self.dilate_iterations):
                    mask = F.max_pool3d(mask, kernel_size=k, stride=1, padding=p)
            if self.gaussian_sigma > 0:
                if self._gaussian_kernel is None:
                    self._gaussian_kernel = self._build_gaussian_kernel(label.device, torch.float32)
                kernel = self._gaussian_kernel.to(device=label.device, dtype=torch.float32)
                p = kernel.shape[-1] // 2
                mask = F.pad(mask, (p, p, p, p, p, p), mode='replicate')
                mask = F.conv3d(mask, kernel)
            mask = mask / mask.max().clamp_min(1e-6)

        return mask.expand(-1, num_channels, -1, -1, -1)


class LogitsDivergenceLoss(nn.Module):
    """
    Compute logits divergence loss between clean and noisy predictions.

    Supports multiple divergence computation modes:
      - 'l1': Direct L1 norm of logits difference
      - 'l2': Direct L2 norm of logits difference
      - 'fft_l1': FFT of logits difference, then L1 norm
      - 'fft_l2': FFT of logits difference, then L2 norm
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
            diff_fft = torch.fft.fftn(diff, dim=self.fft_dims)
            divergence = diff_fft.abs().mean()
        elif self.mode == 'fft_l2':
            diff_fft = torch.fft.fftn(diff, dim=self.fft_dims)
            divergence = (diff_fft.abs() ** 2).mean().sqrt()
        elif self.mode == 'kl_div':
            logits_clean_scaled = logits_clean / self.temperature
            logits_noisy_scaled = logits_noisy / self.temperature
            prob_clean = F.softmax(logits_clean_scaled, dim=1)
            log_prob_noisy = F.log_softmax(logits_noisy_scaled, dim=1)
            divergence = F.kl_div(log_prob_noisy, prob_clean, reduction='batchmean', log_target=False)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        return -self.weight * divergence


# ────────────────────────── Main Plugin ────────────────────────── #

@register_plugin("noise_slice_frequence_learnable")
class NoiseSliceFrequenceLearnable:
    """
    Frequency-Domain Constrained Noise with Global Learnable Cutoffs.

    Two global nn.Parameter scalars (z_cutoff_logit, xy_cutoff_logit) are
    shared across the entire dataset and optimized via gradient descent.
    Cutoff values are logged every epoch.
    """

    def __init__(self):
        self._seg_loss: DiceCELoss | None = None
        self._noise_unet: NoiseUNetWrapper | None = None
        self._global_cutoff: GlobalLearnableCutoff | None = None
        self._opt_unet: torch.optim.Optimizer | None = None
        self._opt_cutoff: torch.optim.Optimizer | None = None
        self._freq_constraint: FrequencyDomainConstraint | None = None
        self._roi_mask_builder: SoftROIMask | None = None
        self._initialized: bool = False
        self._epoch_cutoff_sum_z: float = 0.0
        self._epoch_cutoff_sum_xy: float = 0.0
        self._epoch_cutoff_count: int = 0
        # Store initial cutoff values for comparison
        self._init_z_cutoff: float = 0.0
        self._init_xy_cutoff: float = 0.0
        # Track epoch for detecting epoch boundary
        self._last_logged_epoch: int = -1
        # Frequency constraint switch
        self._freq_constraint_enabled: bool = True
        # Z-axis diversity regularization settings
        self._z_diversity_weight: float = 0.0
        # Logits divergence loss settings
        self._logits_div_loss: LogitsDivergenceLoss | None = None
        self._logits_div_enabled: bool = False
        self.logger = get_logger()

    @staticmethod
    def _norm_inplace(x: torch.Tensor, mean, std):
        for c, (m, s) in enumerate(zip(mean, std)):
            x[:, c].sub_(float(m)).div_(float(s))
        return x

    def _get_seg_loss(self, trainer) -> DiceCELoss:
        if self._seg_loss is not None:
            return self._seg_loss
        cfg = trainer.config
        crit_cfg = get_config(cfg, "training.criterion", DictConfig({}))
        self._seg_loss = DiceCELoss(
            include_background=bool(get_config(crit_cfg, "include_background", False)),
            to_onehot_y=True,
            softmax=True,
            squared_pred=bool(get_config(crit_cfg, "squared_pred", False)),
            jaccard=bool(get_config(crit_cfg, "jaccard", False)),
            lambda_dice=float(get_config(crit_cfg, "lambda_dice", 1.0)),
            lambda_ce=float(get_config(crit_cfg, "lambda_ce", 1.0)),
            reduction="mean",
        )
        return self._seg_loss

    def _init_components(self, trainer, in_channels: int, spatial_dims: int = 3):
        if self._initialized:
            return

        cfg = trainer.config
        device = trainer.device
        params = get_config(cfg, "ue.algorithm.params", DictConfig({}))

        eps = float(get_config(params, "epsilon", 8 / 255))
        z_cutoff_low = float(get_config(params, "z_cutoff_low", 0.1))
        z_sigma = float(get_config(params, "z_sigma", 0.05))
        xy_cutoff_high = float(get_config(params, "xy_cutoff_high", 0.3))
        xy_sigma = float(get_config(params, "xy_sigma", 0.1))

        # Noise UNet
        noise_unet_cfg = get_config(cfg, "ue.noise_unet", DictConfig({}))
        base_unet = _build_noise_unet(noise_unet_cfg, in_channels, spatial_dims)
        self._noise_unet = NoiseUNetWrapper(base_unet, epsilon=eps).to(device)

        # Global learnable cutoff (2 scalar parameters)
        self._global_cutoff = GlobalLearnableCutoff(
            z_cutoff_init=z_cutoff_low,
            xy_cutoff_init=xy_cutoff_high,
        ).to(device)

        # Frequency constraint
        self._freq_constraint = FrequencyDomainConstraint(
            z_cutoff_low=z_cutoff_low,
            z_sigma=z_sigma,
            xy_cutoff_high=xy_cutoff_high,
            xy_sigma=xy_sigma,
        )

        # Optimizers
        opt_cfg = get_config(noise_unet_cfg, "optimizer", DictConfig({}))
        lr = float(get_config(opt_cfg, "lr", 1e-4))
        wd = float(get_config(opt_cfg, "weight_decay", 1e-5))
        betas = tuple(get_config(opt_cfg, "betas", (0.9, 0.999)))

        self._opt_unet = torch.optim.Adam(
            self._noise_unet.parameters(), lr=lr, weight_decay=wd, betas=betas,
        )

        cutoff_lr = lr * float(get_config(params, "cutoff_lr_scale", 1.0))
        self._opt_cutoff = torch.optim.Adam(
            self._global_cutoff.parameters(), lr=cutoff_lr, weight_decay=0.0, betas=betas,
        )

        # ROI mask
        self._roi_aware = bool(get_config(params, "roi_aware", True))
        self._roi_mask_builder = SoftROIMask(
            soft_edge=bool(get_config(params, "soft_edge", True)),
            dilate_iterations=int(get_config(params, "dilate_iterations", 2)),
            dilate_kernel_size=int(get_config(params, "dilate_kernel_size", 3)),
            gaussian_sigma=float(get_config(params, "gaussian_sigma", 2.0)),
        )

        # Frequency constraint switch (false = skip freq mask entirely)
        self._freq_constraint_enabled = bool(get_config(params, "freq_constraint_enabled", True))

        # Z-axis diversity regularization (weight=0 disables it)
        self._z_diversity_weight = float(get_config(params, "z_diversity_weight", 0.0))

        # Logits divergence loss (weight=0 or enabled=false disables it)
        self._logits_div_enabled = bool(get_config(params, "logits_div_enabled", False))
        logits_div_weight = float(get_config(params, "logits_div_weight", 0.0))
        if self._logits_div_enabled and logits_div_weight > 0:
            logits_div_mode = str(get_config(params, "logits_div_mode", "fft_l1"))
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

        z_val, xy_val = self._global_cutoff()
        self._init_z_cutoff = z_val.item()
        self._init_xy_cutoff = xy_val.item()
        self.logger.info(
            f"[FreqLearnable] Initialized: in_ch={in_channels}, eps={eps:.6f}, "
            f"freq_constraint={self._freq_constraint_enabled}, "
            f"z_cutoff_init={self._init_z_cutoff:.4f}, xy_cutoff_init={self._init_xy_cutoff:.4f}, "
            f"z_diversity_weight={self._z_diversity_weight:.4f}, "
            f"logits_div_enabled={self._logits_div_enabled}, logits_div_mode={logits_div_mode}, "
            f"logits_div_weight={logits_div_weight:.4f}"
        )

    # ────────────── epoch boundary: log cutoffs ────────────── #
    def _log_epoch_cutoff_stats(self, epoch: int):
        """Log cutoff statistics for the completed epoch."""
        if self._global_cutoff is None or self._epoch_cutoff_count == 0:
            return

        with torch.no_grad():
            z_val, xy_val = self._global_cutoff()

        z_curr = z_val.item()
        xy_curr = xy_val.item()

        # Compute changes from initial values
        z_delta = z_curr - self._init_z_cutoff
        xy_delta = xy_curr - self._init_xy_cutoff

        self.logger.info(
            f"[FreqLearnable] Epoch {epoch} cutoff update: "
            f"z_cutoff={z_curr:.6f} (init={self._init_z_cutoff:.4f}, Δ={z_delta:+.6f}), "
            f"xy_cutoff={xy_curr:.6f} (init={self._init_xy_cutoff:.4f}, Δ={xy_delta:+.6f})"
        )

        # Reset running averages for next epoch
        self._epoch_cutoff_sum_z = 0.0
        self._epoch_cutoff_sum_xy = 0.0
        self._epoch_cutoff_count = 0

    def on_noise_epoch_end(self, trainer, epoch: int):
        """Called by ue_trainer at end of each noise epoch to log cutoff values."""
        self._log_epoch_cutoff_stats(epoch)

    # ────────────── S-step: update surrogate ────────────── #
    def surrogate_step_batch(self, trainer, batch) -> Dict[str, float]:
        cfg = trainer.config
        device = trainer.device
        nb = trainer.noise_backend
        if nb is None:
            raise RuntimeError("[UE] noise_backend is required.")

        x = batch["image"].to(device).float()
        y = batch["label"]
        y = y.to(device).long() if torch.is_tensor(y) else torch.as_tensor(
            y, device=device, dtype=torch.long,
        )
        keys: Iterable[int] = batch["key"]
        B, C_in = x.shape[:2]
        self._init_components(trainer, C_in, len(x.shape) - 2)

        mean = tuple(get_config(cfg, "training.data.transforms.mean", (0.0,) * C_in))
        std = tuple(get_config(cfg, "training.data.transforms.std", (1.0,) * C_in))

        delta = nb.batch_noise(list(keys)).to(device).float()
        if delta.shape[:2] != x.shape[:2]:
            raise RuntimeError(f"[UE] noise shape mismatch: {tuple(delta.shape)} vs {tuple(x.shape)}")

        if not trainer.surrogates:
            raise RuntimeError("[UE] No surrogate bound.")
        name, s_model = next(iter(trainer.surrogates.items()))
        opt = trainer.opt_surrogates.get(name)
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

        return {"surrogate_loss": float(loss.detach().cpu()), "loss": float(loss.detach().cpu())}

    # ────────────── N-step: update noise + cutoff ────────────── #
    def noise_step_batch(self, trainer, batch) -> Dict[str, float]:
        """
        Update noise UNet and global cutoff parameters.

        Data flow:
          x → NoiseUNet → δ_raw
          GlobalCutoff() → (z_c, xy_c)         ← 2 scalar params
          FreqConstraint(δ_raw, z_c, xy_c) → δ_filtered
          δ_filtered → surrogate → DiceCE Loss
          Loss.backward() → opt_unet.step() + opt_cutoff.step()

        No per-sample loop needed because cutoffs are global scalars
        (not per-sample), so the mask is identical for all samples in
        the batch → standard batch forward + single backward.
        """
        cfg = trainer.config
        device = trainer.device
        nb = trainer.noise_backend
        if nb is None:
            raise RuntimeError("[UE] noise_backend is required.")

        x = batch["image"].to(device).float()
        y = batch["label"]
        y = y.to(device).long() if torch.is_tensor(y) else torch.as_tensor(
            y, device=device, dtype=torch.long,
        )
        keys_list: List[int] = list(batch["key"])

        B, C_in = x.shape[:2]
        self._init_components(trainer, C_in, len(x.shape) - 2)

        # Detect epoch boundary and log previous epoch's cutoff stats
        current_epoch = getattr(trainer, "current_epoch", None) or getattr(trainer, "epoch", 0)
        if current_epoch > self._last_logged_epoch and self._last_logged_epoch >= 0:
            self._log_epoch_cutoff_stats(self._last_logged_epoch)
        if current_epoch != self._last_logged_epoch:
            self._last_logged_epoch = current_epoch

        params = require_config(require_config(cfg, "ue.algorithm"), "params")
        eps = float(get_config(params, "epsilon", 8 / 255.0))
        num_steps = int(get_config(params, "noise_step", 1))

        mean = tuple(get_config(cfg, "training.data.transforms.mean", (0.0,) * C_in))
        std = tuple(get_config(cfg, "training.data.transforms.std", (1.0,) * C_in))

        seg_loss_fn = self._get_seg_loss(trainer)

        # Freeze surrogate
        if not trainer.surrogates:
            raise RuntimeError("[UE] No surrogate bound.")
        _, s_model = next(iter(trainer.surrogates.items()))
        s_model.eval()
        for p in s_model.parameters():
            p.requires_grad = False

        # ROI mask
        roi_mask = self._roi_mask_builder(y, C_in).to(device) if self._roi_aware else None

        # Get clean image predictions (for logits divergence loss)
        logits_clean = None
        if self._logits_div_enabled and self._logits_div_loss is not None:
            with torch.no_grad():
                x_clean_norm = x.clone()
                self._norm_inplace(x_clean_norm, mean, std)
                out_clean = s_model(x_clean_norm)
                logits_clean = out_clean[0] if isinstance(out_clean, (tuple, list)) else out_clean
                logits_clean = logits_clean.detach()

        self._noise_unet.train()
        last_loss = torch.tensor(0.0, device=device)
        last_z_diversity_loss = torch.tensor(0.0, device=device)
        last_div_loss = torch.tensor(0.0, device=device)

        for _ in range(max(1, num_steps)):
            # NoiseUNet forward
            delta_raw = self._noise_unet(x)

            # Apply frequency constraint (if enabled)
            if self._freq_constraint_enabled:
                z_c, xy_c = self._global_cutoff()  # scalar tensors with grad
                delta_filtered = self._freq_constraint(delta_raw, z_c, xy_c)
            else:
                delta_filtered = delta_raw

            if roi_mask is not None:
                delta = delta_filtered * roi_mask
            else:
                delta = delta_filtered
            delta = delta.clamp(-eps, eps)

            perturb_img = (x + delta).clamp(0.0, 1.0)
            xn = perturb_img.clone()
            self._norm_inplace(xn, mean, std)

            out = s_model(xn)
            logits_noisy = out[0] if isinstance(out, (tuple, list)) else out
            seg_loss = seg_loss_fn(logits_noisy, y.unsqueeze(1))

            # Start with seg_loss
            loss = seg_loss

            # Z-axis diversity loss (if weight > 0)
            # We want to MAXIMIZE diversity, so we add negative diversity to loss
            if self._z_diversity_weight > 0:
                z_diversity = self._compute_z_diversity(delta)
                z_diversity_loss = -z_diversity  # negative because we want to maximize
                loss = loss + self._z_diversity_weight * z_diversity_loss
                last_z_diversity_loss = z_diversity_loss.detach()

            # Logits divergence loss (if enabled)
            # Maximize difference between clean and noisy predictions
            if self._logits_div_loss is not None and logits_clean is not None:
                div_loss = self._logits_div_loss(logits_clean, logits_noisy)
                loss = loss + div_loss
                last_div_loss = div_loss.detach()

            last_loss = loss.detach()

            # Update networks from the same loss
            self._opt_unet.zero_grad(set_to_none=True)
            if self._freq_constraint_enabled:
                self._opt_cutoff.zero_grad(set_to_none=True)
            loss.backward()
            self._opt_unet.step()
            if self._freq_constraint_enabled:
                self._opt_cutoff.step()

        # Read current cutoff values (after update)
        with torch.no_grad():
            z_val, xy_val = self._global_cutoff()
        z_val_f = float(z_val.cpu())
        xy_val_f = float(xy_val.cpu())

        # Accumulate for epoch-level logging
        self._epoch_cutoff_sum_z += z_val_f
        self._epoch_cutoff_sum_xy += xy_val_f
        self._epoch_cutoff_count += 1

        # Store final noise to backend
        self._noise_unet.eval()
        with torch.no_grad():
            final_noise = self._noise_unet(x)
            if self._freq_constraint_enabled:
                final_filtered = self._freq_constraint(final_noise, z_val, xy_val)
            else:
                final_filtered = final_noise
            if roi_mask is not None:
                final_delta = final_filtered * roi_mask
            else:
                final_delta = final_filtered
            final_delta = final_delta.clamp(-eps, eps)

        nb.commit_batch(keys_list, final_delta.detach().cpu())

        delta_linf = float(final_delta.detach().abs().max().cpu())

        with torch.no_grad():
            z_energy, xy_energy = self._compute_freq_stats(final_delta)
            z_diversity_value = self._compute_z_diversity(final_delta)

            # Compute final logits divergence for logging
            logits_diff_l1 = 0.0
            if logits_clean is not None:
                perturb_final = (x + final_delta).clamp(0.0, 1.0)
                xn_final = perturb_final.clone()
                self._norm_inplace(xn_final, mean, std)
                out_final = s_model(xn_final)
                logits_final = out_final[0] if isinstance(out_final, (tuple, list)) else out_final
                logits_diff_l1 = (logits_final - logits_clean).abs().mean().cpu().item()

        result = {
            "noise_loss": float(last_loss.cpu()),
            "delta_linf": delta_linf,
            "z_high_freq_energy": z_energy,
            "xy_low_freq_energy": xy_energy,
            "z_cutoff": z_val_f,
            "xy_cutoff": xy_val_f,
            "z_diversity": float(z_diversity_value.cpu()),
        }

        # Add z_diversity_loss to result (if enabled)
        if self._z_diversity_weight > 0:
            result["z_diversity_loss"] = float(last_z_diversity_loss.cpu())

        # Add logits_div metrics to result (if enabled)
        if self._logits_div_loss is not None:
            result["div_loss"] = float(last_div_loss.cpu())
            result["logits_diff_l1"] = logits_diff_l1

        return result

    def _compute_z_diversity(self, delta: torch.Tensor) -> torch.Tensor:
        """
        Compute z-axis inter-slice diversity in frequency domain.

        Computes the mean L2 distance between adjacent slices after 2D FFT,
        encouraging noise to have high variation along the z-axis.

        Args:
            delta: [B, C, D, H, W] noise tensor

        Returns:
            z_diversity: Scalar tensor representing mean inter-slice L2 difference
        """
        # Apply 2D FFT on each slice (xy-plane)
        delta_fft_2d = torch.fft.fft2(delta, dim=(-2, -1))  # [B, C, D, H, W] complex

        # Compute magnitude spectrum for each slice
        delta_fft_mag = delta_fft_2d.abs()  # [B, C, D, H, W]

        # Compute L2 difference between adjacent slices along z-axis
        slice_diff = delta_fft_mag[:, :, 1:, :, :] - delta_fft_mag[:, :, :-1, :, :]  # [B, C, D-1, H, W]

        # Compute L2 norm for each pair of slices
        l2_per_pair = torch.sqrt((slice_diff ** 2).sum(dim=(-2, -1)) + 1e-10)  # [B, C, D-1]

        # Mean over all pairs, channels, and batches
        z_diversity = l2_per_pair.mean()

        return z_diversity

    def _compute_freq_stats(self, delta: torch.Tensor) -> Tuple[float, float]:
        B, C, D, H, W = delta.shape
        delta_fft = torch.fft.fftn(delta, dim=(-3, -2, -1))
        power = (delta_fft.abs() ** 2).mean(dim=(0, 1))

        freq_z = torch.fft.fftfreq(D, device=delta.device).abs()
        z_high_energy = power[freq_z >= 0.1].sum() / power.sum().clamp_min(1e-10)

        freq_y = torch.fft.fftfreq(H, device=delta.device).abs()
        freq_x = torch.fft.fftfreq(W, device=delta.device).abs()
        _, yy, xx = torch.meshgrid(freq_z, freq_y, freq_x, indexing='ij')
        r_xy = torch.sqrt(yy ** 2 + xx ** 2)
        xy_low_energy = power[r_xy <= 0.3].sum() / power.sum().clamp_min(1e-10)

        return float(z_high_energy.cpu()), float(xy_low_energy.cpu())