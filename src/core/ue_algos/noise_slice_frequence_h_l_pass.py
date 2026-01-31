# file: src/core/ue_algos/noise_slice_frequence.py
"""
Frequency-Domain Constrained Noise for 3D Medical Image Segmentation UE.

Core Design:
  - Use UNet to generate base noise (same as unet_roi_noise)
  - Apply frequency domain constraints:
    * Z-axis (inter-slice): HIGH-PASS -> maximize z-direction frequency diversity
    * XY-plane (intra-slice): LOW/MID-PASS -> smooth within slices
  - ROI mask with soft edges: 二值化 → 膨胀 → 高斯模糊

This approach creates noise that:
  - Has high inter-slice diversity (z-axis high-frequency components)
  - Is smooth within each slice (xy low/mid-frequency)
  - Only affects ROI regions with smooth boundaries
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


class CutoffPredictor(nn.Module):
    """
    Lightweight network that predicts per-sample (z_cutoff, xy_cutoff) from input image.

    Architecture: AdaptiveAvgPool3d(4) → Flatten → MLP(2 layers) → sigmoid → range mapping
    """

    def __init__(
        self,
        in_channels: int,
        z_cutoff_init: float = 0.1,
        xy_cutoff_init: float = 0.3,
        z_range: Tuple[float, float] = (0.01, 0.45),
        xy_range: Tuple[float, float] = (0.05, 0.45),
    ):
        super().__init__()
        self.z_range = z_range
        self.xy_range = xy_range

        self.pool = nn.AdaptiveAvgPool3d(4)
        hidden = 64
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels * 4 * 4 * 4, hidden),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(hidden, 2),
        )

        # Initialize so that initial output = default cutoff values
        with torch.no_grad():
            z_t = (z_cutoff_init - z_range[0]) / (z_range[1] - z_range[0])
            xy_t = (xy_cutoff_init - xy_range[0]) / (xy_range[1] - xy_range[0])
            self.mlp[-1].weight.zero_()
            self.mlp[-1].bias.copy_(torch.tensor([
                torch.logit(torch.tensor(z_t).clamp(0.01, 0.99)),
                torch.logit(torch.tensor(xy_t).clamp(0.01, 0.99)),
            ]))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, C, D, H, W]
        Returns:
            z_cutoff [B], xy_cutoff [B]
        """
        h = self.pool(x)
        out = torch.sigmoid(self.mlp(h))  # [B, 2] in (0, 1)
        z_cutoff = out[:, 0] * (self.z_range[1] - self.z_range[0]) + self.z_range[0]
        xy_cutoff = out[:, 1] * (self.xy_range[1] - self.xy_range[0]) + self.xy_range[0]
        return z_cutoff, xy_cutoff


class FrequencyDomainConstraint(nn.Module):
    """
    Apply frequency domain constraints to noise.

    Design:
      - Z-axis (inter-slice): HIGH-PASS filter -> maximize layer diversity
      - XY-plane (intra-slice): LOW/MID-PASS filter -> smooth within layers

    The spectral mask M is constructed as: M = M_z_highpass * M_xy_lowpass

    Supports per-sample learnable cutoffs when called with cutoff tensors.
    """

    def __init__(
        self,
        z_cutoff_low: float = 0.1,      # Z高通的截止频率 (去除低于此的z频率)
        z_sigma: float = 0.05,           # Z方向软过渡sigma
        xy_cutoff_high: float = 0.3,     # XY低通的截止频率 (去除高于此的xy频率)
        xy_sigma: float = 0.1,           # XY方向软过渡sigma
    ):
        super().__init__()
        self.z_cutoff_low = z_cutoff_low
        self.z_sigma = z_sigma
        self.xy_cutoff_high = xy_cutoff_high
        self.xy_sigma = xy_sigma

        # Cache for spectral mask (static mode only)
        self._cached_mask = None
        self._cached_shape = None

        # Cache for frequency grids (reusable across calls)
        self._cached_freq_grids = None
        self._cached_grid_shape = None

    def _get_freq_grids(
        self,
        D: int, H: int, W: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return cached (abs_k_z, r_xy) grids, each [D, H, W]."""
        if self._cached_freq_grids is not None and self._cached_grid_shape == (D, H, W):
            abs_k_z, r_xy = self._cached_freq_grids
            return abs_k_z.to(device=device, dtype=dtype), r_xy.to(device=device, dtype=dtype)

        freq_z = torch.fft.fftfreq(D, device=device, dtype=dtype)
        freq_y = torch.fft.fftfreq(H, device=device, dtype=dtype)
        freq_x = torch.fft.fftfreq(W, device=device, dtype=dtype)
        k_z, k_y, k_x = torch.meshgrid(freq_z, freq_y, freq_x, indexing='ij')
        abs_k_z = torch.abs(k_z)
        r_xy = torch.sqrt(k_x ** 2 + k_y ** 2)
        self._cached_freq_grids = (abs_k_z, r_xy)
        self._cached_grid_shape = (D, H, W)
        return abs_k_z, r_xy

    def _build_spectral_mask(
        self,
        D: int, H: int, W: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """
        Build 3D spectral mask [1, 1, D, H, W] using static cutoff values.
        """
        abs_k_z, r_xy = self._get_freq_grids(D, H, W, device, dtype)

        M_z = torch.where(
            abs_k_z >= self.z_cutoff_low,
            torch.ones_like(abs_k_z),
            torch.exp(-((self.z_cutoff_low - abs_k_z) ** 2) / (2 * self.z_sigma ** 2))
        )

        M_xy = torch.where(
            r_xy <= self.xy_cutoff_high,
            torch.ones_like(r_xy),
            torch.exp(-((r_xy - self.xy_cutoff_high) ** 2) / (2 * self.xy_sigma ** 2))
        )

        M = M_z * M_xy
        M[0, 0, 0] = 0.1

        return M.unsqueeze(0).unsqueeze(0)  # [1, 1, D, H, W]

    def _get_separable_freq_grids(
        self,
        D: int, H: int, W: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return 1-D frequency vectors for separable mask construction.

        Returns:
            abs_freq_z: [D]  absolute z-frequencies
            r_xy:       [H, W]  radial xy-frequencies (2-D, much smaller than [D,H,W])
        """
        freq_z = torch.fft.fftfreq(D, device=device, dtype=dtype)
        freq_y = torch.fft.fftfreq(H, device=device, dtype=dtype)
        freq_x = torch.fft.fftfreq(W, device=device, dtype=dtype)
        abs_freq_z = freq_z.abs()                                     # [D]
        r_xy = torch.sqrt(freq_y.unsqueeze(1) ** 2 + freq_x.unsqueeze(0) ** 2)  # [H, W]
        return abs_freq_z, r_xy

    def forward(
        self,
        noise: torch.Tensor,
        z_cutoff: torch.Tensor | None = None,
        xy_cutoff: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Apply frequency domain constraints to noise.

        Memory-efficient: when using per-sample cutoffs, the mask is applied
        as two sequential multiplications (M_z then M_xy) in separable form
        instead of materialising a full [B, D, H, W] mask tensor.

        Args:
            noise: [B, C, D, H, W] raw noise from UNet
            z_cutoff: [B] per-sample z cutoff (if None, use static self.z_cutoff_low)
            xy_cutoff: [B] per-sample xy cutoff (if None, use static self.xy_cutoff_high)

        Returns:
            filtered_noise: [B, C, D, H, W] frequency-constrained noise
        """
        B, C, D, H, W = noise.shape
        device = noise.device
        dtype = noise.dtype

        # FFT (shared by both paths)
        noise_fft = torch.fft.fftn(noise, dim=(-3, -2, -1))  # [B, C, D, H, W] complex

        if z_cutoff is not None and xy_cutoff is not None:
            # ---------- Separable per-sample path (memory-efficient) ----------
            abs_freq_z, r_xy = self._get_separable_freq_grids(D, H, W, device, dtype)
            # abs_freq_z: [D],  r_xy: [H, W]

            # M_z: [B, 1, D, 1, 1]  —  only D elements per sample
            z_c = z_cutoff.view(B, 1, 1, 1, 1)                       # [B,1,1,1,1]
            freq_z_5d = abs_freq_z.view(1, 1, D, 1, 1)               # [1,1,D,1,1]
            M_z = torch.sigmoid((freq_z_5d - z_c) / self.z_sigma)    # [B,1,D,1,1]

            # M_xy: [B, 1, 1, H, W]  —  only H*W elements per sample
            xy_c = xy_cutoff.view(B, 1, 1, 1, 1)                     # [B,1,1,1,1]
            r_xy_5d = r_xy.view(1, 1, 1, H, W)                       # [1,1,1,H,W]
            M_xy = torch.sigmoid((xy_c - r_xy_5d) / self.xy_sigma)   # [B,1,1,H,W]

            # Correct DC component: M_z[0]*M_xy[0] should be 0.1, not sigmoid(...)
            # Build a DC correction factor applied to M_z at index 0 (out-of-place)
            dc_mz = M_z[:, :, 0:1, :, :]   # [B,1,1,1,1]
            dc_mxy = M_xy[:, :, :, 0:1, 0:1]  # [B,1,1,1,1]
            dc_product = (dc_mz * dc_mxy).clamp_min(1e-6)
            dc_correction = 0.1 / dc_product  # want final DC = 0.1

            # Apply correction to M_z at DC position (out-of-place)
            M_z_corrected = M_z.clone()
            M_z_corrected[:, :, 0:1, :, :] = M_z[:, :, 0:1, :, :] * dc_correction

            # Apply M_z and M_xy sequentially (never materialise [B,D,H,W])
            noise_fft = noise_fft * M_z_corrected  # broadcast: [B,C,D,H,W] * [B,1,D,1,1]
            noise_fft = noise_fft * M_xy            # broadcast: [B,C,D,H,W] * [B,1,1,H,W]
        else:
            # ---------- Static cached path (original behaviour) ----------
            if self._cached_mask is None or self._cached_shape != (D, H, W):
                self._cached_mask = self._build_spectral_mask(D, H, W, device, dtype)
                self._cached_shape = (D, H, W)
            else:
                self._cached_mask = self._cached_mask.to(device=device, dtype=dtype)
            M = self._cached_mask  # [1, 1, D, H, W]
            noise_fft = noise_fft * M.expand(B, C, -1, -1, -1)

        filtered_noise = torch.fft.ifftn(noise_fft, dim=(-3, -2, -1)).real
        return filtered_noise


class SoftROIMask(nn.Module):
    """
    Create ROI mask with optional soft edges: 二值化 → (膨胀 → 高斯模糊)

    Options:
      - soft_edge=True: 二值化 → 膨胀 → 高斯模糊 (smooth boundaries)
      - soft_edge=False: 二值化 only (hard edges)
    """

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

        # Build Gaussian kernel
        self._gaussian_kernel = None

    def _build_gaussian_kernel(
        self,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Build 3D Gaussian kernel for smoothing."""
        sigma = self.gaussian_sigma
        kernel_size = int(6 * sigma + 1)
        if kernel_size % 2 == 0:
            kernel_size += 1

        coords = torch.arange(kernel_size, device=device, dtype=dtype) - kernel_size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = g / g.sum()

        # Create separable 3D kernel
        kernel_3d = g.view(-1, 1, 1) * g.view(1, -1, 1) * g.view(1, 1, -1)
        kernel_3d = kernel_3d / kernel_3d.sum()

        return kernel_3d.view(1, 1, kernel_size, kernel_size, kernel_size)

    def forward(self, label: torch.Tensor, num_channels: int) -> torch.Tensor:
        """
        Create ROI mask from label.

        Args:
            label: [B, D, H, W] or [B, 1, D, H, W] segmentation label
            num_channels: Number of channels to expand to

        Returns:
            mask: [B, C, D, H, W] mask in [0, 1]
        """
        device = label.device
        dtype = torch.float32

        # Step 1: 二值化 (Binarization) - always applied
        if label.dim() == 5:
            label = label.squeeze(1)
        mask = (label > 0).float()  # [B, D, H, W]
        mask = mask.unsqueeze(1)     # [B, 1, D, H, W]

        # soft_edge=True: 膨胀 + 高斯模糊
        # soft_edge=False: 只使用二值化硬边缘
        if self.soft_edge:
            # Step 2: 膨胀 (Dilation) using max pooling
            if self.dilate_iterations > 0 and self.dilate_kernel_size > 0:
                k = self.dilate_kernel_size
                pad = k // 2
                for _ in range(self.dilate_iterations):
                    mask = F.max_pool3d(mask, kernel_size=k, stride=1, padding=pad)

            # Step 3: 高斯模糊 (Gaussian blur)
            if self.gaussian_sigma > 0:
                if self._gaussian_kernel is None:
                    self._gaussian_kernel = self._build_gaussian_kernel(device, dtype)
                kernel = self._gaussian_kernel.to(device=device, dtype=dtype)
                k_size = kernel.shape[-1]
                pad = k_size // 2

                mask = F.pad(mask, (pad, pad, pad, pad, pad, pad), mode='replicate')
                mask = F.conv3d(mask, kernel)

            # Normalize to [0, 1]
            mask = mask / mask.max().clamp_min(1e-6)

        # Expand to [B, C, D, H, W]
        mask = mask.expand(-1, num_channels, -1, -1, -1)

        return mask


@register_plugin("noise_slice_frequence")
class NoiseSliceFrequenceUE:
    """
    Frequency-Domain Constrained Noise for 3D Medical Image Segmentation.

    Key Features:
      1. UNet-based noise generation (like unet_roi_noise)
      2. Frequency domain constraints:
         - Z-axis HIGH-PASS: maximize inter-slice diversity
         - XY-plane LOW/MID-PASS: ensure intra-slice smoothness
      3. Soft ROI mask: 二值化 → 膨胀 → 高斯模糊

    Configuration (ue.algorithm.params):
      epsilon: L_inf bound (default: 8/255)
      noise_step: UNet training iterations per batch (default: 1)

      # Frequency constraints
      z_cutoff_low: Z-axis high-pass cutoff (default: 0.1)
      z_sigma: Z-axis soft transition sigma (default: 0.05)
      xy_cutoff_high: XY low-pass cutoff (default: 0.3)
      xy_sigma: XY soft transition sigma (default: 0.1)

      # ROI mask
      dilate_iterations: Number of dilation iterations (default: 2)
      dilate_kernel_size: Dilation kernel size (default: 3)
      gaussian_sigma: Gaussian blur sigma (default: 2.0)
    """

    def __init__(self):
        self._seg_loss: DiceCELoss | None = None
        self._noise_unet: NoiseUNetWrapper | None = None
        self._cutoff_predictor: CutoffPredictor | None = None
        self._opt_unet: torch.optim.Optimizer | None = None
        self._opt_cutoff: torch.optim.Optimizer | None = None
        self._freq_constraint: FrequencyDomainConstraint | None = None
        self._learnable_cutoff: bool = False
        self._roi_mask_builder: SoftROIMask | None = None
        self._initialized: bool = False
        self.logger = get_logger()

    @staticmethod
    def _norm_inplace(x: torch.Tensor, mean, std):
        """In-place per-channel normalize for ND volume."""
        for c, (m, s) in enumerate(zip(mean, std)):
            x[:, c].sub_(float(m)).div_(float(s))
        return x

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

        # Get parameters
        params = get_config(cfg, "ue.algorithm.params", DictConfig({}))
        eps = float(get_config(params, "epsilon", 8/255))

        # Frequency constraint parameters
        z_cutoff_low = float(get_config(params, "z_cutoff_low", 0.1))
        z_sigma = float(get_config(params, "z_sigma", 0.05))
        xy_cutoff_high = float(get_config(params, "xy_cutoff_high", 0.3))
        xy_sigma = float(get_config(params, "xy_sigma", 0.1))

        self._learnable_cutoff = bool(get_config(params, "learnable_cutoff", False))

        # Build noise UNet
        noise_unet_cfg = get_config(cfg, "ue.noise_unet", DictConfig({}))
        base_unet = _build_noise_unet(noise_unet_cfg, in_channels, spatial_dims)
        self._noise_unet = NoiseUNetWrapper(base_unet, epsilon=eps).to(device)

        # Build frequency constraint module
        self._freq_constraint = FrequencyDomainConstraint(
            z_cutoff_low=z_cutoff_low,
            z_sigma=z_sigma,
            xy_cutoff_high=xy_cutoff_high,
            xy_sigma=xy_sigma,
        )

        # Build optimizers
        opt_cfg = get_config(noise_unet_cfg, "optimizer", DictConfig({}))
        lr = float(get_config(opt_cfg, "lr", 1e-4))
        weight_decay = float(get_config(opt_cfg, "weight_decay", 1e-5))
        betas = tuple(get_config(opt_cfg, "betas", (0.9, 0.999)))

        self._opt_unet = torch.optim.Adam(
            self._noise_unet.parameters(),
            lr=lr, weight_decay=weight_decay, betas=betas,
        )

        # Build cutoff predictor (optional)
        if self._learnable_cutoff:
            cutoff_lr = lr * float(get_config(params, "cutoff_lr_scale", 1.0))
            self._cutoff_predictor = CutoffPredictor(
                in_channels=in_channels,
                z_cutoff_init=z_cutoff_low,
                xy_cutoff_init=xy_cutoff_high,
            ).to(device)
            self._opt_cutoff = torch.optim.Adam(
                self._cutoff_predictor.parameters(),
                lr=cutoff_lr, weight_decay=weight_decay, betas=betas,
            )

        # ROI mask配置
        self._roi_aware = bool(get_config(params, "roi_aware", True))
        soft_edge = bool(get_config(params, "soft_edge", True))
        dilate_iterations = int(get_config(params, "dilate_iterations", 2))
        dilate_kernel_size = int(get_config(params, "dilate_kernel_size", 3))
        gaussian_sigma = float(get_config(params, "gaussian_sigma", 2.0))

        self._roi_mask_builder = SoftROIMask(
            soft_edge=soft_edge,
            dilate_iterations=dilate_iterations,
            dilate_kernel_size=dilate_kernel_size,
            gaussian_sigma=gaussian_sigma,
        )

        self._initialized = True
        self.logger.info(
            f"[NoiseSliceFrequence] Initialized: in_channels={in_channels}, "
            f"eps={eps:.6f}, z_cutoff_low={z_cutoff_low}, xy_cutoff_high={xy_cutoff_high}, "
            f"learnable_cutoff={self._learnable_cutoff}, "
            f"roi_aware={self._roi_aware}, soft_edge={soft_edge}, gaussian_sigma={gaussian_sigma}"
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

    # ---------------- N-step: Update noise via UNet + Frequency Filter ---------------- #
    def noise_step_batch(self, trainer, batch) -> Dict[str, float]:
        """
        Update noise using UNet + frequency domain constraints.

        Process:
          1. Forward image through noise UNet to get base noise
          2. Apply frequency domain constraints (z-highpass, xy-lowpass)
          3. Create soft ROI mask (二值化 → 膨胀 → 高斯模糊)
          4. Apply ROI mask to filtered noise
          5. Forward through frozen surrogate, compute loss
          6. Backprop to update noise UNet
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

        self._init_components(trainer, C_in, spatial_dims)

        algo = require_config(cfg, "ue.algorithm")
        params = require_config(algo, "params")
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

        # Create soft ROI mask: 二值化 → 膨胀 → 高斯模糊 (only if roi_aware=True)
        if self._roi_aware:
            roi_mask = self._roi_mask_builder(y, C_in).to(device)  # [B, C, D, H, W]
        else:
            roi_mask = None  # 全图噪声，不使用ROI mask

        # Train noise UNet (+ cutoff predictor if learnable)
        self._noise_unet.train()
        if self._learnable_cutoff:
            self._cutoff_predictor.train()
        last_loss = torch.tensor(0.0, device=device)
        last_z_cutoff = None
        last_xy_cutoff = None

        for _ in range(max(1, num_steps)):
            if not self._learnable_cutoff:
                # ---- Static cutoff path: batch forward as before ----
                delta_raw = self._noise_unet(x)
                delta_filtered = self._freq_constraint(delta_raw)

                if roi_mask is not None:
                    delta = delta_filtered * roi_mask
                else:
                    delta = delta_filtered
                delta = delta.clamp(-eps, eps)

                perturb_img = (x + delta).clamp(0.0, 1.0)
                xn = perturb_img.clone()
                self._norm_inplace(xn, mean, std)

                out = s_model(xn)
                logits = out[0] if isinstance(out, (tuple, list)) else out
                loss = seg_loss_fn(logits, y.unsqueeze(1))
                last_loss = loss.detach()

                self._opt_unet.zero_grad(set_to_none=True)
                loss.backward()
                self._opt_unet.step()
            else:
                # ---- Learnable cutoff: two-phase update ----
                # Phase 1: fix cutoff, update NoiseUNet
                #   Image → NoiseUNet → δ → FreqConstraint(fixed cutoff) → Loss → update UNet
                with torch.no_grad():
                    z_c, xy_c = self._cutoff_predictor(x)  # [B], [B]

                delta_raw = self._noise_unet(x)
                delta_filtered = self._freq_constraint(delta_raw, z_c, xy_c)

                if roi_mask is not None:
                    delta = delta_filtered * roi_mask
                else:
                    delta = delta_filtered
                delta = delta.clamp(-eps, eps)

                perturb_img = (x + delta).clamp(0.0, 1.0)
                xn = perturb_img.clone()
                self._norm_inplace(xn, mean, std)

                out = s_model(xn)
                logits = out[0] if isinstance(out, (tuple, list)) else out
                loss_p1 = seg_loss_fn(logits, y.unsqueeze(1))

                self._opt_unet.zero_grad(set_to_none=True)
                loss_p1.backward()
                self._opt_unet.step()

                # Phase 2: fix noise, update CutoffPredictor
                #   Image → CutoffPredictor → (z_c, xy_c) → FreqConstraint(δ.detach()) → Loss → update Cutoff
                delta_raw_det = delta_raw.detach()
                z_c2, xy_c2 = self._cutoff_predictor(x)
                delta_filtered2 = self._freq_constraint(delta_raw_det, z_c2, xy_c2)

                if roi_mask is not None:
                    delta2 = delta_filtered2 * roi_mask
                else:
                    delta2 = delta_filtered2
                delta2 = delta2.clamp(-eps, eps)

                perturb_img2 = (x + delta2).clamp(0.0, 1.0)
                xn2 = perturb_img2.clone()
                self._norm_inplace(xn2, mean, std)

                out2 = s_model(xn2)
                logits2 = out2[0] if isinstance(out2, (tuple, list)) else out2
                loss_p2 = seg_loss_fn(logits2, y.unsqueeze(1))

                self._opt_cutoff.zero_grad(set_to_none=True)
                loss_p2.backward()
                self._opt_cutoff.step()

                last_loss = loss_p2.detach()
                last_z_cutoff = z_c2.detach()
                last_xy_cutoff = xy_c2.detach()

        # Store final noise to backend
        self._noise_unet.eval()
        with torch.no_grad():
            final_noise = self._noise_unet(x)
            if self._learnable_cutoff:
                z_cutoff, xy_cutoff = self._cutoff_predictor(x)
                final_delta_filtered = self._freq_constraint(final_noise, z_cutoff, xy_cutoff)
                last_z_cutoff = z_cutoff
                last_xy_cutoff = xy_cutoff
            else:
                final_delta_filtered = self._freq_constraint(final_noise)
            if roi_mask is not None:
                final_delta = final_delta_filtered * roi_mask
            else:
                final_delta = final_delta_filtered
            final_delta = final_delta.clamp(-eps, eps)

        nb.commit_batch(keys_list, final_delta.detach().cpu())

        delta_linf = float(final_delta.detach().abs().max().cpu())

        # Compute frequency statistics for logging
        with torch.no_grad():
            z_energy, xy_energy = self._compute_freq_stats(final_delta)

        result = {
            "noise_loss": float(last_loss.cpu()),
            "delta_linf": delta_linf,
            "z_high_freq_energy": z_energy,
            "xy_low_freq_energy": xy_energy,
        }

        if self._learnable_cutoff and last_z_cutoff is not None:
            result["z_cutoff_mean"] = float(last_z_cutoff.mean().cpu())
            result["xy_cutoff_mean"] = float(last_xy_cutoff.mean().cpu())

        return result

    def _compute_freq_stats(self, delta: torch.Tensor) -> Tuple[float, float]:
        """
        Compute frequency statistics for monitoring.

        Returns:
            z_high_freq_energy: Ratio of high-frequency energy in z-axis
            xy_low_freq_energy: Ratio of low-frequency energy in xy-plane
        """
        B, C, D, H, W = delta.shape

        # FFT
        delta_fft = torch.fft.fftn(delta, dim=(-3, -2, -1))
        power = (delta_fft.abs() ** 2).mean(dim=(0, 1))  # [D, H, W]

        # Z-axis frequency grid
        freq_z = torch.fft.fftfreq(D, device=delta.device).abs()

        # Z high-frequency energy ratio
        z_high_mask = freq_z >= 0.1
        z_high_energy = power[z_high_mask].sum() / power.sum().clamp_min(1e-10)

        # XY low-frequency energy ratio
        freq_y = torch.fft.fftfreq(H, device=delta.device).abs()
        freq_x = torch.fft.fftfreq(W, device=delta.device).abs()
        _, yy, xx = torch.meshgrid(freq_z, freq_y, freq_x, indexing='ij')
        r_xy = torch.sqrt(yy ** 2 + xx ** 2)
        xy_low_mask = r_xy <= 0.3
        xy_low_energy = power[xy_low_mask].sum() / power.sum().clamp_min(1e-10)

        return float(z_high_energy.cpu()), float(xy_low_energy.cpu())