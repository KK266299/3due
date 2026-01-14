# file: src/core/ue_algos/unet_frequency.py
"""
Coherent Spectral UE: Frequency-Domain Perturbation for 3D Medical Image Segmentation.

Motivation:
  3D segmentation models rely heavily on inter-slice consistency. This method
  generates perturbations in the frequency domain with explicit control over
  spectral characteristics.

Core Idea:
  - Learn a spectral tensor P (real + imag) in frequency domain
  - Apply fixed spectral mask M to control frequency support
  - Generate perturbation: delta = IFFT3D(M * P)
  - Apply soft ROI mask with Gaussian smoothing

What is being optimized:
  - The spectral tensor P (learnable parameters: P_real, P_imag)
  - NOT the noise directly, but the frequency-domain representation

What is stored to backend:
  - The generated noise delta (spatial domain), after ROI mask

Two Modes:
  - "enhance": z-axis lowpass -> high inter-slice coherence
  - "destroy": z-axis highpass -> low inter-slice coherence

Training loop (same structure as UNetROINoiseUE):
  1. Surrogate-step: Train surrogate on noisy images (noise from backend)
  2. Noise-step:
     - Generate delta = IFFT(M * P) with soft ROI mask
     - Forward through frozen surrogate, compute seg loss
     - Backprop to update P (spectral parameters)
     - Store generated noise to backend

Maximum noise bound: 4/255 (configurable via epsilon parameter)
"""
from __future__ import annotations
from typing import Dict, Iterable, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
from monai.losses import DiceCELoss

from ...registry import register_plugin
from ...utils.config import get_config, require_config
from ...utils.logger import get_logger


class SpectralNoiseGenerator(nn.Module):
    """
    Frequency-domain noise generator.

    Learnable Parameters:
      - P_real: Real part of spectral tensor [C, D, H, W]
      - P_imag: Imaginary part of spectral tensor [C, D, H, W]

    Fixed:
      - spectral_mask M: Controls which frequencies are allowed [D, H, W]

    Output:
      - delta = IFFT3D(M * P) * epsilon, bounded in [-eps, eps]
    """

    def __init__(
        self,
        spatial_shape: Tuple[int, int, int],
        num_channels: int,
        epsilon: float = 4 / 255,
        mode: str = "enhance",
        z_cutoff: float = 0.1,
        z_sigma: float = 0.05,
        xy_center: float = 0.15,
        xy_sigma: float = 0.1,
        init_scale: float = 0.01,
    ):
        super().__init__()
        self.spatial_shape = spatial_shape
        self.num_channels = num_channels
        self.epsilon = epsilon
        self.mode = mode

        D, H, W = spatial_shape

        # ========== Learnable Parameters: P_real and P_imag ==========
        # These are what we optimize during training
        self.P_real = nn.Parameter(torch.randn(num_channels, D, H, W) * init_scale)
        self.P_imag = nn.Parameter(torch.randn(num_channels, D, H, W) * init_scale)

        # ========== Fixed Spectral Mask M (not learnable) ==========
        # Registered as buffer, will be created on first forward
        self.register_buffer('spectral_mask', None)
        self._mask_config = {
            'mode': mode,
            'z_cutoff': z_cutoff,
            'z_sigma': z_sigma,
            'xy_center': xy_center,
            'xy_sigma': xy_sigma,
        }

    def _create_spectral_mask(self, device: torch.device) -> torch.Tensor:
        """
        Create fixed spectral mask M based on 3D characteristics.

        M = M_z * M_xy where:
          - M_z: Controls inter-slice coherence (z-direction)
          - M_xy: Band-pass for intra-slice structure (xy-plane)
        """
        D, H, W = self.spatial_shape

        # Frequency grids (normalized to [-0.5, 0.5))
        freq_z = torch.fft.fftfreq(D, device=device)
        freq_y = torch.fft.fftfreq(H, device=device)
        freq_x = torch.fft.fftfreq(W, device=device)
        k_z, k_y, k_x = torch.meshgrid(freq_z, freq_y, freq_x, indexing='ij')

        # Radial frequency in xy-plane
        r_xy = torch.sqrt(k_x ** 2 + k_y ** 2)

        z_cutoff = self._mask_config['z_cutoff']
        z_sigma = self._mask_config['z_sigma']
        xy_center = self._mask_config['xy_center']
        xy_sigma = self._mask_config['xy_sigma']

        # Z-axis mask
        abs_k_z = torch.abs(k_z)
        if self._mask_config['mode'] == "enhance":
            # Low-pass: high inter-slice coherence (misleading 3D structure)
            M_z = torch.where(
                abs_k_z <= z_cutoff,
                torch.ones_like(abs_k_z),
                torch.exp(-((abs_k_z - z_cutoff) ** 2) / (2 * z_sigma ** 2))
            )
        else:  # "destroy"
            # High-pass: low inter-slice coherence (disrupt 3D consistency)
            M_z = 1.0 - torch.exp(-(k_z ** 2) / (2 * z_sigma ** 2))

        # XY-plane mask: Band-pass (mid-frequency, avoid pure DC bias)
        M_xy = torch.exp(-((r_xy - xy_center) ** 2) / (2 * xy_sigma ** 2))

        # Combined mask
        M = M_z * M_xy

        # Slight DC suppression for "enhance" mode to avoid global bias
        if self._mask_config['mode'] == "enhance":
            M[0, 0, 0] = M[0, 0, 0] * 0.5

        return M

    def _ensure_mask(self, device: torch.device):
        """Create spectral mask if not already created."""
        if self.spectral_mask is None:
            self.spectral_mask = self._create_spectral_mask(device)

    def forward(self) -> torch.Tensor:
        """
        Generate perturbation from learnable spectral tensor P.

        Process:
          1. P_complex = P_real + i * P_imag
          2. P_masked = M * P_complex
          3. Enforce Hermitian symmetry (ensure real output)
          4. delta = IFFT3D(P_masked)
          5. delta = tanh(delta) * epsilon

        Returns:
          delta: [C, D, H, W] perturbation in [-epsilon, epsilon]
        """
        device = self.P_real.device
        self._ensure_mask(device)

        # Step 1: Construct complex spectral tensor from learnable params
        P_complex = torch.complex(self.P_real, self.P_imag)  # [C, D, H, W]

        # Step 2: Apply fixed spectral mask M
        # Expand M from [D,H,W] to [C,D,H,W]
        M = self.spectral_mask.unsqueeze(0).expand(self.num_channels, -1, -1, -1)
        P_masked = P_complex * M

        # Step 3: Enforce Hermitian symmetry for real output
        P_flipped = torch.flip(P_masked, dims=[-3, -2, -1])
        P_symmetric = (P_masked + torch.conj(P_flipped)) / 2

        # Step 4: Inverse 3D FFT -> spatial domain
        delta = torch.fft.ifftn(P_symmetric, dim=(-3, -2, -1)).real  # [C, D, H, W]

        # Step 5: Bound to [-epsilon, epsilon] using tanh
        delta = torch.tanh(delta) * self.epsilon

        return delta

    def get_coherence_stats(self) -> Dict[str, float]:
        """Compute inter-slice correlation for analysis."""
        with torch.no_grad():
            delta = self.forward()  # [C, D, H, W]
            delta_mean = delta.mean(dim=0)  # [D, H, W]
            D = delta_mean.shape[0]

            correlations = []
            for z in range(D - 1):
                s1 = delta_mean[z].flatten()
                s2 = delta_mean[z + 1].flatten()
                corr = F.cosine_similarity(s1.unsqueeze(0), s2.unsqueeze(0))
                correlations.append(corr.item())

            return {
                'mean_corr': sum(correlations) / len(correlations) if correlations else 0.0,
                'min_corr': min(correlations) if correlations else 0.0,
                'max_corr': max(correlations) if correlations else 0.0,
            }


def _soft_roi_mask(label: torch.Tensor, sigma: float, num_channels: int) -> torch.Tensor:
    """
    Create soft ROI mask with Gaussian smoothing.

    Args:
        label: [B, D, H, W] segmentation label
        sigma: Gaussian blur sigma (0 = hard mask)
        num_channels: Number of channels to expand to

    Returns:
        mask: [B, C, D, H, W] soft mask in [0, 1]
    """
    if label.dim() == 5:
        label = label.squeeze(1)

    # Binary mask: ROI = label > 0
    mask = (label > 0).float()  # [B, D, H, W]

    if sigma > 0:
        # 3D Gaussian blur using separable convolutions
        kernel_size = int(6 * sigma + 1)
        if kernel_size % 2 == 0:
            kernel_size += 1

        x = torch.arange(kernel_size, device=label.device, dtype=torch.float32)
        x = x - kernel_size // 2
        gaussian_1d = torch.exp(-x ** 2 / (2 * sigma ** 2))
        gaussian_1d = gaussian_1d / gaussian_1d.sum()

        kernel_d = gaussian_1d.view(1, 1, -1, 1, 1)
        kernel_h = gaussian_1d.view(1, 1, 1, -1, 1)
        kernel_w = gaussian_1d.view(1, 1, 1, 1, -1)

        mask = mask.unsqueeze(1)  # [B, 1, D, H, W]
        pad = kernel_size // 2
        mask = F.pad(mask, (pad, pad, pad, pad, pad, pad), mode='replicate')
        mask = F.conv3d(mask, kernel_d)
        mask = F.conv3d(mask, kernel_h)
        mask = F.conv3d(mask, kernel_w)
        mask = mask.clamp(0.0, 1.0)
    else:
        mask = mask.unsqueeze(1)  # [B, 1, D, H, W]

    # Expand to [B, C, D, H, W]
    mask = mask.expand(-1, num_channels, -1, -1, -1)
    return mask


@register_plugin("unet_frequency")
class FrequencyDomainUE:
    """
    Coherent Spectral UE for 3D Medical Image Segmentation.

    ========== What is being optimized? ==========
    The spectral tensor P (P_real and P_imag), NOT the noise directly.
    P lives in frequency domain; noise is generated via IFFT(M * P).

    ========== What is stored to backend? ==========
    The generated noise delta (in spatial domain), after applying ROI mask.

    Interface compatible with UNetROINoiseUE:
      - surrogate_step_batch(): Update surrogate using noise from backend
      - noise_step_batch(): Update P via backprop, store generated noise

    Assumptions:
      - batch["image"]: [B, C, D, H, W]
      - batch["label"]: [B, D, H, W]
      - batch["key"]: sample-wise key
    """

    def __init__(self):
        self._seg_loss: DiceCELoss | None = None
        self._noise_gen: SpectralNoiseGenerator | None = None
        self._optimizer: torch.optim.Optimizer | None = None
        self._initialized: bool = False
        self.logger = get_logger()

    @staticmethod
    def _norm_inplace(x: torch.Tensor, mean, std):
        """In-place per-channel normalization."""
        for c, (m, s) in enumerate(zip(mean, std)):
            x[:, c].sub_(float(m)).div_(float(s))
        return x

    def _get_seg_loss(self, trainer) -> DiceCELoss:
        """Build DiceCELoss consistent with SegTrainer."""
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

    def _init_noise_generator(self, trainer, spatial_shape: Tuple[int, int, int], num_channels: int):
        """
        Initialize SpectralNoiseGenerator and optimizer (lazy, once).

        Similar to _init_noise_unet in UNetROINoiseUE.
        """
        if self._initialized:
            return

        cfg = trainer.config
        device = trainer.device

        # Get parameters from config (similar structure to noise_unet config)
        params = get_config(cfg, "ue.algorithm.params", DictConfig({}))

        epsilon = float(get_config(params, "epsilon", 4 / 255))
        mode = str(get_config(params, "mode", "enhance"))
        z_cutoff = float(get_config(params, "z_cutoff", 0.1))
        z_sigma = float(get_config(params, "z_sigma", 0.05))
        xy_center = float(get_config(params, "xy_center", 0.15))
        xy_sigma = float(get_config(params, "xy_sigma", 0.1))
        init_scale = float(get_config(params, "init_scale", 0.01))

        # Create spectral noise generator
        self._noise_gen = SpectralNoiseGenerator(
            spatial_shape=spatial_shape,
            num_channels=num_channels,
            epsilon=epsilon,
            mode=mode,
            z_cutoff=z_cutoff,
            z_sigma=z_sigma,
            xy_center=xy_center,
            xy_sigma=xy_sigma,
            init_scale=init_scale,
        ).to(device)

        # Optimizer for spectral parameters P_real and P_imag
        lr = float(get_config(params, "lr", 1e-2))
        weight_decay = float(get_config(params, "weight_decay", 0.0))

        self._optimizer = torch.optim.Adam(
            self._noise_gen.parameters(),  # This optimizes P_real and P_imag
            lr=lr,
            weight_decay=weight_decay,
        )

        self._initialized = True
        self.logger.info(
            f"[FrequencyUE] Initialized: shape={spatial_shape}, channels={num_channels}, "
            f"eps={epsilon:.6f}, mode={mode}, lr={lr}"
        )

    # ---------------- Surrogate-step: Update surrogate ---------------- #
    def surrogate_step_batch(self, trainer, batch) -> Dict[str, float]:
        """
        Update surrogate parameters only, using noise from backend.

        Same as UNetROINoiseUE.surrogate_step_batch().
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
        spatial_shape = tuple(x.shape[2:])

        self._init_noise_generator(trainer, spatial_shape, C_in)

        mean = tuple(get_config(cfg, "training.data.transforms.mean", (0.0,) * C_in))
        std = tuple(get_config(cfg, "training.data.transforms.std", (1.0,) * C_in))

        # Get noise from backend (stored from previous noise_step)
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

    # ---------------- N-step: Update spectral params P ---------------- #
    def noise_step_batch(self, trainer, batch) -> Dict[str, float]:
        """
        Update spectral parameters P via gradient descent.

        ========== Key Difference from UNet methods ==========
        - UNet: noise = UNet(x), optimizes UNet parameters
        - This: noise = IFFT(M * P), optimizes P (P_real, P_imag)

        Process:
          1. Generate delta = IFFT(M * P) using SpectralNoiseGenerator
          2. Create soft ROI mask from label
          3. Apply ROI mask: delta_masked = delta * roi_mask
          4. Add noise to image, forward through frozen surrogate
          5. Compute segmentation loss
          6. Backprop to update P (spectral parameters)
          7. Store delta_masked to backend
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
        spatial_shape = tuple(x.shape[2:])

        self._init_noise_generator(trainer, spatial_shape, C_in)

        params = get_config(cfg, "ue.algorithm.params", DictConfig({}))
        eps = float(get_config(params, "epsilon", 4 / 255.0))
        num_steps = int(get_config(params, "noise_step", 1))
        roi_sigma = float(get_config(params, "roi_sigma", 2.0))

        mean = tuple(get_config(cfg, "training.data.transforms.mean", (0.0,) * C_in))
        std = tuple(get_config(cfg, "training.data.transforms.std", (1.0,) * C_in))

        seg_loss_fn = self._get_seg_loss(trainer)

        # -------- Freeze surrogate --------
        if not trainer.surrogates:
            raise RuntimeError("[UE] No surrogate bound.")
        _, s_model = next(iter(trainer.surrogates.items()))
        s_model.eval()
        for p in s_model.parameters():
            p.requires_grad = False

        # -------- Create soft ROI mask --------
        roi_mask = _soft_roi_mask(y, sigma=roi_sigma, num_channels=C_in).to(device)

        # -------- Train spectral parameters P --------
        self._noise_gen.train()
        last_loss = torch.tensor(0.0, device=device)

        for _ in range(max(1, num_steps)):
            # Step 1: Generate delta from spectral tensor P
            # This is where P_real and P_imag are used
            delta_base = self._noise_gen()  # [C, D, H, W]

            # Expand to batch: [B, C, D, H, W]
            delta = delta_base.unsqueeze(0).expand(B, -1, -1, -1, -1)

            # Step 2: Apply soft ROI mask
            delta_masked = delta * roi_mask

            # Step 3: Create perturbed image
            perturb_img = (x + delta_masked).clamp(0.0, 1.0)
            xn = perturb_img.clone()
            self._norm_inplace(xn, mean, std)

            # Step 4: Forward through frozen surrogate
            out = s_model(xn)
            logits = out[0] if isinstance(out, (tuple, list)) else out

            # Step 5: Compute segmentation loss
            loss = seg_loss_fn(logits, y.unsqueeze(1))
            last_loss = loss.detach()

            # Step 6: Backprop to update P (P_real, P_imag)
            self._optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self._optimizer.step()

        # -------- Store generated noise to backend --------
        self._noise_gen.eval()
        with torch.no_grad():
            delta_base = self._noise_gen()
            delta = delta_base.unsqueeze(0).expand(B, -1, -1, -1, -1)
            delta_masked = delta * roi_mask
            delta_masked = delta_masked.clamp(-eps, eps)

        nb.commit_batch(keys_list, delta_masked.detach().cpu())

        delta_linf = float(delta_masked.detach().abs().max().cpu())

        # Get coherence stats for logging
        coherence_stats = self._noise_gen.get_coherence_stats()

        return {
            "noise_loss": float(last_loss.cpu()),
            "delta_linf": delta_linf,
            "inter_slice_corr": coherence_stats['mean_corr'],
        }