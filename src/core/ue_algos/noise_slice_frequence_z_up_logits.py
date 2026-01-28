# file: src/core/ue_algos/noise_slice_frequence_z_up_logits.py
"""
Frequency-Domain Constrained Noise with Logits Divergence Loss.

Extension of noise_slice_frequence that adds a logits divergence loss to
maximize the difference between predictions on clean vs noisy images.

Core Design:
  - Use UNet to generate base noise (same as noise_slice_frequence)
  - Apply frequency domain constraints:
    * Z-axis (inter-slice): HIGH-PASS -> maximize z-direction frequency diversity
    * XY-plane (intra-slice): LOW/MID-PASS -> smooth within slices
  - ROI mask with soft edges: 二值化 → 膨胀 → 高斯模糊
  - NEW: Logits divergence loss to maximize prediction difference

Logits Divergence Loss:
  - Computes the difference between clean and noisy image predictions
  - Supports multiple modes: l1, l2, fft_l1, fft_l2, kl_div
  - Loss is negated to maximize divergence (minimize negative divergence)
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


class FrequencyDomainConstraint(nn.Module):
    """
    Apply frequency domain constraints to noise.

    Design:
      - Z-axis (inter-slice): HIGH-PASS filter -> maximize layer diversity
      - XY-plane (intra-slice): LOW/MID-PASS filter -> smooth within layers

    The spectral mask M is constructed as: M = M_z_highpass * M_xy_lowpass
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

        # Cache for spectral mask
        self._cached_mask = None
        self._cached_shape = None

    def _build_spectral_mask(
        self,
        D: int, H: int, W: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """
        Build 3D spectral mask [1, 1, D, H, W].

        Z-axis: HIGH-PASS (remove low frequencies, keep high frequencies)
          -> Maximizes inter-slice diversity

        XY-plane: LOW/MID-PASS (keep low/mid frequencies, remove high)
          -> Ensures intra-slice smoothness
        """
        # Frequency grids (normalized to [-0.5, 0.5))
        freq_z = torch.fft.fftfreq(D, device=device, dtype=dtype)  # [D]
        freq_y = torch.fft.fftfreq(H, device=device, dtype=dtype)  # [H]
        freq_x = torch.fft.fftfreq(W, device=device, dtype=dtype)  # [W]

        # Create 3D meshgrid
        k_z, k_y, k_x = torch.meshgrid(freq_z, freq_y, freq_x, indexing='ij')

        abs_k_z = torch.abs(k_z)
        r_xy = torch.sqrt(k_x ** 2 + k_y ** 2)

        # Z-axis: HIGH-PASS (soft transition)
        # M_z = 1 when |f_z| > z_cutoff_low, smooth transition below
        M_z = torch.where(
            abs_k_z >= self.z_cutoff_low,
            torch.ones_like(abs_k_z),
            torch.exp(-((self.z_cutoff_low - abs_k_z) ** 2) / (2 * self.z_sigma ** 2))
        )

        # XY-plane: LOW/MID-PASS (soft transition)
        # M_xy = 1 when r_xy < xy_cutoff_high, smooth transition above
        M_xy = torch.where(
            r_xy <= self.xy_cutoff_high,
            torch.ones_like(r_xy),
            torch.exp(-((r_xy - self.xy_cutoff_high) ** 2) / (2 * self.xy_sigma ** 2))
        )

        # Combine: both constraints must be satisfied
        M = M_z * M_xy

        # Keep DC component partially (avoid complete removal)
        # But reduce it to maintain some inter-slice diversity
        M[0, 0, 0] = 0.1

        return M.unsqueeze(0).unsqueeze(0)  # [1, 1, D, H, W]

    def forward(self, noise: torch.Tensor) -> torch.Tensor:
        """
        Apply frequency domain constraints to noise.

        Args:
            noise: [B, C, D, H, W] raw noise from UNet

        Returns:
            filtered_noise: [B, C, D, H, W] frequency-constrained noise
        """
        B, C, D, H, W = noise.shape
        device = noise.device
        dtype = noise.dtype

        # Build or retrieve cached mask
        if self._cached_mask is None or self._cached_shape != (D, H, W):
            self._cached_mask = self._build_spectral_mask(D, H, W, device, dtype)
            self._cached_shape = (D, H, W)
        else:
            self._cached_mask = self._cached_mask.to(device=device, dtype=dtype)

        M = self._cached_mask  # [1, 1, D, H, W]

        # FFT -> Apply mask -> IFFT
        # Process each batch and channel
        noise_fft = torch.fft.fftn(noise, dim=(-3, -2, -1))  # [B, C, D, H, W] complex

        # Expand mask to match
        M_expanded = M.expand(B, C, -1, -1, -1)  # [B, C, D, H, W]

        # Apply spectral mask
        noise_fft_filtered = noise_fft * M_expanded

        # Inverse FFT
        filtered_noise = torch.fft.ifftn(noise_fft_filtered, dim=(-3, -2, -1)).real

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
        """
        Args:
            mode: Divergence computation mode ('l1', 'l2', 'fft_l1', 'fft_l2', 'kl_div')
            weight: Loss weight multiplier
            temperature: Temperature for softmax in kl_div mode
            fft_dims: Dimensions to apply FFT over (default: spatial dims D, H, W)
        """
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
        # Compute logits difference
        diff = logits_noisy - logits_clean  # [B, C, D, H, W]

        if self.mode == 'l1':
            # Direct L1 norm
            divergence = diff.abs().mean()

        elif self.mode == 'l2':
            # Direct L2 norm
            divergence = (diff ** 2).mean().sqrt()

        elif self.mode == 'fft_l1':
            # FFT then L1 norm
            diff_fft = torch.fft.fftn(diff, dim=self.fft_dims)
            # Use magnitude of complex FFT
            diff_fft_mag = diff_fft.abs()
            divergence = diff_fft_mag.mean()

        elif self.mode == 'fft_l2':
            # FFT then L2 norm
            diff_fft = torch.fft.fftn(diff, dim=self.fft_dims)
            diff_fft_mag = diff_fft.abs()
            divergence = (diff_fft_mag ** 2).mean().sqrt()

        elif self.mode == 'kl_div':
            # KL divergence between softmax distributions
            # Flatten spatial dimensions for softmax
            B, C = logits_clean.shape[:2]
            spatial_shape = logits_clean.shape[2:]

            # Apply temperature scaling
            logits_clean_scaled = logits_clean / self.temperature
            logits_noisy_scaled = logits_noisy / self.temperature

            # Softmax over class dimension
            prob_clean = F.softmax(logits_clean_scaled, dim=1)
            log_prob_noisy = F.log_softmax(logits_noisy_scaled, dim=1)

            # KL(clean || noisy) = sum(prob_clean * (log_prob_clean - log_prob_noisy))
            # We use F.kl_div which expects log_prob as first arg
            kl_loss = F.kl_div(
                log_prob_noisy,
                prob_clean,
                reduction='batchmean',
                log_target=False
            )
            divergence = kl_loss

        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        # Return negative divergence (to maximize divergence by minimizing loss)
        # Higher divergence = more difference = better for our goal
        return -self.weight * divergence


@register_plugin("noise_slice_frequence_z_up_logits")
class NoiseSliceFrequenceZUpLogitsUE:
    """
    Frequency-Domain Constrained Noise with Logits Divergence Loss.

    Extension of NoiseSliceFrequenceUE that adds a logits divergence loss to
    maximize the prediction difference between clean and noisy images.

    Key Features:
      1. UNet-based noise generation (like noise_slice_frequence)
      2. Frequency domain constraints:
         - Z-axis HIGH-PASS: maximize inter-slice diversity
         - XY-plane LOW/MID-PASS: ensure intra-slice smoothness
      3. Soft ROI mask: 二值化 → 膨胀 → 高斯模糊
      4. NEW: Logits divergence loss to maximize prediction differences

    Logits Divergence Modes:
      - 'l1': Direct L1 norm of logits difference
      - 'l2': Direct L2 norm of logits difference
      - 'fft_l1': FFT of logits difference, then L1 norm (recommended)
      - 'fft_l2': FFT of logits difference, then L2 norm
      - 'kl_div': KL divergence between softmax distributions

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

      # Logits divergence (NEW)
      logits_div_mode: Divergence mode (default: 'fft_l1')
      logits_div_weight: Weight for divergence loss (default: 1.0)
      logits_div_temperature: Temperature for KL div mode (default: 1.0)
    """

    def __init__(self):
        self._seg_loss: DiceCELoss | None = None
        self._noise_unet: NoiseUNetWrapper | None = None
        self._opt_noise_unet: torch.optim.Optimizer | None = None
        self._freq_constraint: FrequencyDomainConstraint | None = None
        self._roi_mask_builder: SoftROIMask | None = None
        self._logits_div_loss: LogitsDivergenceLoss | None = None
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

        # Build noise UNet
        noise_unet_cfg = get_config(cfg, "ue.noise_unet", DictConfig({}))
        base_unet = _build_noise_unet(noise_unet_cfg, in_channels, spatial_dims)
        self._noise_unet = NoiseUNetWrapper(base_unet, epsilon=eps)
        self._noise_unet = self._noise_unet.to(device)

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

        # Build frequency constraint module
        z_cutoff_low = float(get_config(params, "z_cutoff_low", 0.1))
        z_sigma = float(get_config(params, "z_sigma", 0.05))
        xy_cutoff_high = float(get_config(params, "xy_cutoff_high", 0.3))
        xy_sigma = float(get_config(params, "xy_sigma", 0.1))

        self._freq_constraint = FrequencyDomainConstraint(
            z_cutoff_low=z_cutoff_low,
            z_sigma=z_sigma,
            xy_cutoff_high=xy_cutoff_high,
            xy_sigma=xy_sigma,
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

        # Build logits divergence loss (NEW)
        logits_div_mode = str(get_config(params, "logits_div_mode", "fft_l1"))
        logits_div_weight = float(get_config(params, "logits_div_weight", 1.0))
        logits_div_temperature = float(get_config(params, "logits_div_temperature", 1.0))

        self._logits_div_loss = LogitsDivergenceLoss(
            mode=logits_div_mode,
            weight=logits_div_weight,
            temperature=logits_div_temperature,
        )

        self._initialized = True
        self.logger.info(
            f"[NoiseSliceFrequenceZUpLogits] Initialized: in_channels={in_channels}, "
            f"eps={eps:.6f}, z_cutoff_low={z_cutoff_low}, xy_cutoff_high={xy_cutoff_high}, "
            f"roi_aware={self._roi_aware}, soft_edge={soft_edge}, gaussian_sigma={gaussian_sigma}, "
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

    # ---------------- N-step: Update noise via UNet + Frequency Filter + Logits Divergence ---------------- #
    def noise_step_batch(self, trainer, batch) -> Dict[str, float]:
        """
        Update noise using UNet + frequency domain constraints + logits divergence.

        Process:
          1. Forward image through noise UNet to get base noise
          2. Apply frequency domain constraints (z-highpass, xy-lowpass)
          3. Create soft ROI mask (二值化 → 膨胀 → 高斯模糊)
          4. Apply ROI mask to filtered noise
          5. Forward both clean and noisy images through frozen surrogate
          6. Compute combined loss: seg_loss + logits_divergence_loss
          7. Backprop to update noise UNet
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

        # Get clean image predictions (for logits divergence)
        with torch.no_grad():
            x_clean_norm = x.clone()
            self._norm_inplace(x_clean_norm, mean, std)
            out_clean = s_model(x_clean_norm)
            logits_clean = out_clean[0] if isinstance(out_clean, (tuple, list)) else out_clean
            logits_clean = logits_clean.detach()  # No grad needed for clean

        # Train noise UNet
        self._noise_unet.train()
        last_loss = torch.tensor(0.0, device=device)
        last_seg_loss = torch.tensor(0.0, device=device)
        last_div_loss = torch.tensor(0.0, device=device)

        for _ in range(max(1, num_steps)):
            # Step 1: Generate base noise from UNet
            delta_raw = self._noise_unet(x)  # [B, C, D, H, W]

            # Step 2: Apply frequency domain constraints
            delta_filtered = self._freq_constraint(delta_raw)  # [B, C, D, H, W]

            # Step 3: Apply ROI mask (if roi_aware=True)
            if roi_mask is not None:
                delta = delta_filtered * roi_mask
            else:
                delta = delta_filtered

            # Step 4: Clip to epsilon
            delta = delta.clamp(-eps, eps)

            # Step 5: Create perturbed image
            perturb_img = (x + delta).clamp(0.0, 1.0)
            xn = perturb_img.clone()
            self._norm_inplace(xn, mean, std)

            # Step 6: Forward through surrogate (noisy)
            out = s_model(xn)
            logits_noisy = out[0] if isinstance(out, (tuple, list)) else out

            # Step 7: Compute combined loss
            # Segmentation loss (on noisy image)
            seg_loss = seg_loss_fn(logits_noisy, y.unsqueeze(1))

            # Logits divergence loss (maximize difference between clean and noisy)
            div_loss = self._logits_div_loss(logits_clean, logits_noisy)

            # Combined loss
            loss = seg_loss + div_loss

            last_loss = loss.detach()
            last_seg_loss = seg_loss.detach()
            last_div_loss = div_loss.detach()

            # Step 8: Backprop to update noise UNet
            self._opt_noise_unet.zero_grad(set_to_none=True)
            loss.backward()
            self._opt_noise_unet.step()

        # Store final noise to backend
        self._noise_unet.eval()
        with torch.no_grad():
            final_delta_raw = self._noise_unet(x)
            final_delta_filtered = self._freq_constraint(final_delta_raw)
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

            # Compute final logits divergence for logging
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
            "z_high_freq_energy": z_energy,
            "xy_low_freq_energy": xy_energy,
        }

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
