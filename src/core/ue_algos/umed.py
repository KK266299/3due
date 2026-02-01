# file: src/core/ue_algos/umed.py
"""
UMed: Unlearnable Medical Image Segmentation via Contour- and Texture-Aware
Perturbations.

Reference: arXiv:2403.14250

Core idea:
  - Two trainable perturbation generators: G_c (contour) and G_t (texture)
  - G_c uses a U-Net with Central-Difference-Aware Convolution (CDC) in the
    encoder to capture edge/contour features. Perturbation is masked to the
    boundary band extracted from GT labels.
  - G_t uses a standard U-Net. Perturbation magnitude is adaptively scaled by
    a Local Binary Pattern (LBP) texture intensity map and restricted to the
    ROI interior (excluding contour regions).
  - Final protected image: x_p = clip[0,1](x + delta_c + delta_t)
  - Bi-level optimization: generators and surrogate are updated alternately.

Adapted to 3D segmentation (BraTS19) following the existing codebase pattern.
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


# ---------------------------------------------------------------------------
# Central-Difference-Aware Convolution (CDC) for 3D
# ---------------------------------------------------------------------------
class CDConv3d(nn.Module):
    """
    Central-Difference-Aware 3D Convolution.

    Combines vanilla convolution and central-difference convolution:
        f_out(r) = conv_v(f_in)(r) + theta * [conv_c(f_in)(r) - f_in(r)*sum(w_c)]

    In practice, theta is a learnable blend factor. We implement this as:
        out = (1 - theta) * vanilla_conv(x) + theta * cd_conv(x)
    where cd_conv uses a kernel whose center weight is zeroed and subtracted.
    """

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int = 3, stride: int = 1, padding: int = 1,
                 bias: bool = True, theta: float = 0.5):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=padding, bias=bias)
        self.theta = nn.Parameter(torch.tensor(theta))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Vanilla convolution
        out_vanilla = self.conv(x)

        # Central-difference: zero-out center weight
        w = self.conv.weight
        k = w.shape[2] // 2
        # Mask center element
        mask = torch.ones_like(w)
        mask[:, :, k, k, k] = 0.0
        w_cd = w * mask
        center_sum = w[:, :, k, k, k].sum(dim=1, keepdim=True)  # [out, 1]

        out_cd = F.conv3d(x, w_cd, bias=None, stride=self.conv.stride,
                          padding=self.conv.padding)
        # Subtract center contribution (broadcast over spatial dims)
        # center_sum shape: [out_ch, 1] -> [1, out_ch, 1, 1, 1]
        # This is an approximation following the paper's spirit
        out_cd = out_cd - F.conv3d(x, w * (1 - mask), bias=None,
                                   stride=self.conv.stride,
                                   padding=self.conv.padding)

        theta_s = torch.sigmoid(self.theta)
        out = (1.0 - theta_s) * out_vanilla + theta_s * out_cd
        return out


# ---------------------------------------------------------------------------
# Build generators (G_c and G_t)
# ---------------------------------------------------------------------------
def _build_noise_unet(cfg: DictConfig, in_channels: int,
                      spatial_dims: int = 3) -> nn.Module:
    """Standard MONAI U-Net for noise generation (used as G_t backbone)."""
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


class CDCUNet3d(nn.Module):
    """
    Lightweight CDC-enhanced U-Net for contour perturbation (G_c).

    Uses CDConv3d in the encoder path to capture contour/edge features.
    Decoder uses standard transposed convolutions.
    """

    def __init__(self, in_channels: int, channels: list = None,
                 theta: float = 0.5):
        super().__init__()
        if channels is None:
            channels = [16, 32, 64, 128]

        # Encoder (CDC convolutions)
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        prev_ch = in_channels
        for ch in channels:
            self.encoders.append(nn.Sequential(
                CDConv3d(prev_ch, ch, 3, 1, 1, theta=theta),
                nn.InstanceNorm3d(ch),
                nn.LeakyReLU(inplace=True),
                CDConv3d(ch, ch, 3, 1, 1, theta=theta),
                nn.InstanceNorm3d(ch),
                nn.LeakyReLU(inplace=True),
            ))
            self.pools.append(nn.MaxPool3d(2))
            prev_ch = ch

        # Decoder (standard convolutions)
        self.decoders = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        for i in range(len(channels) - 1, 0, -1):
            self.upsamples.append(
                nn.ConvTranspose3d(channels[i], channels[i - 1],
                                   kernel_size=2, stride=2)
            )
            self.decoders.append(nn.Sequential(
                nn.Conv3d(channels[i - 1] * 2, channels[i - 1], 3, 1, 1),
                nn.InstanceNorm3d(channels[i - 1]),
                nn.LeakyReLU(inplace=True),
            ))

        self.head = nn.Conv3d(channels[0], in_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        h = x
        for enc, pool in zip(self.encoders, self.pools):
            h = enc(h)
            skips.append(h)
            h = pool(h)

        # Remove last skip (bottleneck)
        h = skips.pop()
        for up, dec in zip(self.upsamples, self.decoders):
            h = up(h)
            s = skips.pop()
            # Handle size mismatch from pooling
            if h.shape != s.shape:
                h = F.interpolate(h, size=s.shape[2:], mode='trilinear',
                                  align_corners=False)
            h = torch.cat([h, s], dim=1)
            h = dec(h)

        return self.head(h)


# ---------------------------------------------------------------------------
# Wrapper modules
# ---------------------------------------------------------------------------
class ContourGenerator(nn.Module):
    """G_c: CDC U-Net -> tanh -> clip to [-eps, eps] -> mask to contour band."""

    def __init__(self, cdc_unet: nn.Module, epsilon: float = 4 / 255):
        super().__init__()
        self.net = cdc_unet
        self.epsilon = epsilon

    def forward(self, x: torch.Tensor,
                contour_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, D, H, W]
            contour_mask: [B, 1, D, H, W] binary contour band
        Returns:
            delta_c: [B, C, D, H, W] contour perturbation
        """
        raw = self.net(x)
        delta = torch.tanh(raw) * self.epsilon
        delta = delta * contour_mask
        return delta


class TextureGenerator(nn.Module):
    """G_t: Standard U-Net -> adaptive LBP-based clipping -> mask to interior."""

    def __init__(self, unet: nn.Module, epsilon: float = 4 / 255):
        super().__init__()
        self.net = unet
        self.epsilon = epsilon

    def forward(self, x: torch.Tensor, interior_mask: torch.Tensor,
                lbp_map: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, D, H, W]
            interior_mask: [B, 1, D, H, W] interior ROI (excl. contour)
            lbp_map: [B, 1, D, H, W] normalised LBP texture intensity in [0,1]
        Returns:
            delta_t: [B, C, D, H, W] texture perturbation
        """
        raw = self.net(x)
        # Adaptive clipping: bound scales with local texture intensity
        upper = self.epsilon * lbp_map * interior_mask
        lower = -upper
        delta = torch.tanh(raw) * self.epsilon
        delta = torch.clamp(delta, lower, upper)
        return delta


# ---------------------------------------------------------------------------
# Morphological boundary extraction (reuse pattern from boundary_noise)
# ---------------------------------------------------------------------------
def _morphological_dilate_3d(mask: torch.Tensor, ksize: int) -> torch.Tensor:
    pad = ksize // 2
    padded = F.pad(mask, (pad, pad, pad, pad, pad, pad), mode='constant',
                   value=0)
    return F.max_pool3d(padded, kernel_size=ksize, stride=1, padding=0)


def _morphological_erode_3d(mask: torch.Tensor, ksize: int) -> torch.Tensor:
    pad = ksize // 2
    padded = F.pad(mask, (pad, pad, pad, pad, pad, pad), mode='constant',
                   value=1)
    return -F.max_pool3d(-padded, kernel_size=ksize, stride=1, padding=0)


def _extract_contour_and_interior(
    label: torch.Tensor,
    device: torch.device,
    boundary_width: int = 3,
):
    """
    From GT label extract:
      - contour_mask: boundary band [B, 1, D, H, W]
      - interior_mask: interior ROI excluding boundary [B, 1, D, H, W]

    boundary = dilate(fg) - erode(fg)
    interior = fg - boundary
    """
    if label.dim() == 5:
        label = label.squeeze(1)
    fg = (label > 0).float().unsqueeze(1).to(device)  # [B, 1, D, H, W]
    ksize = 2 * boundary_width + 1
    dilated = _morphological_dilate_3d(fg, ksize)
    eroded = _morphological_erode_3d(fg, ksize)
    contour = (dilated - eroded).clamp(0, 1)
    interior = (fg - contour).clamp(0, 1)
    return contour, interior


# ---------------------------------------------------------------------------
# 3D Local Binary Pattern (simplified for volumes)
# ---------------------------------------------------------------------------
def _compute_lbp_3d(x: torch.Tensor) -> torch.Tensor:
    """
    Compute a simplified 3D LBP-like texture intensity map.

    For each voxel, compare with its 6-connected neighbours (±D, ±H, ±W).
    LBP code = sum of binary comparisons. Normalise to [0, 1].

    Args:
        x: [B, C, D, H, W] in [0, 1]
    Returns:
        lbp: [B, 1, D, H, W] normalised texture intensity
    """
    # Average over channels to get grayscale
    gray = x.mean(dim=1, keepdim=True)  # [B, 1, D, H, W]
    padded = F.pad(gray, (1, 1, 1, 1, 1, 1), mode='replicate')

    shifts = [
        (0, 0, -1), (0, 0, 1),   # W neighbours
        (0, -1, 0), (0, 1, 0),   # H neighbours
        (-1, 0, 0), (1, 0, 0),   # D neighbours
    ]

    D, H, W = gray.shape[2], gray.shape[3], gray.shape[4]
    lbp_code = torch.zeros_like(gray)
    for dd, dh, dw in shifts:
        neighbour = padded[
            :, :,
            1 + dd: 1 + dd + D,
            1 + dh: 1 + dh + H,
            1 + dw: 1 + dw + W,
        ]
        lbp_code = lbp_code + (gray >= neighbour).float()

    # Normalise: max possible = 6
    lbp_norm = lbp_code / 6.0
    return lbp_norm


# ---------------------------------------------------------------------------
# Main UMed Plugin
# ---------------------------------------------------------------------------
@register_plugin("umed")
class UMedUE:
    """
    UMed: Contour- and Texture-Aware Perturbation for Medical Image
    Segmentation Protection.

    Two generators:
      - G_c (contour perturbator): CDC U-Net, perturbation masked to boundary
      - G_t (texture perturbator): Standard U-Net, LBP-adaptive clipping to
        interior

    Bi-level optimisation (Algorithm 1 from the paper):
      - Every `surrogate_step` iterations: update surrogate F_s
      - Other iterations: update G_c and G_t

    Assumptions (same as codebase convention):
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
        self._contour_gen: ContourGenerator | None = None
        self._texture_gen: TextureGenerator | None = None
        self._opt_generators: torch.optim.Optimizer | None = None
        self._device: torch.device | None = None
        self._initialized: bool = False
        self.logger = get_logger()

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
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

    # ------------------------------------------------------------------ #
    # Initialisation
    # ------------------------------------------------------------------ #
    def _init_generators(self, trainer, in_channels: int,
                         spatial_dims: int = 3):
        if self._initialized:
            return

        cfg = trainer.config
        device = trainer.device
        self._device = device

        params = get_config(cfg, "ue.algorithm.params", DictConfig({}))
        eps = float(get_config(params, "epsilon", 4 / 255))

        # --- Contour generator (CDC U-Net) ---
        gc_cfg = get_config(cfg, "ue.contour_unet", DictConfig({}))
        gc_channels = list(get_config(gc_cfg, "channels", [16, 32, 64, 128]))
        gc_theta = float(get_config(gc_cfg, "cdc_theta", 0.5))
        cdc_unet = CDCUNet3d(in_channels, channels=gc_channels,
                             theta=gc_theta)
        self._contour_gen = ContourGenerator(cdc_unet, epsilon=eps).to(device)

        # --- Texture generator (standard U-Net) ---
        gt_cfg = get_config(cfg, "ue.texture_unet", DictConfig({}))
        base_unet = _build_noise_unet(gt_cfg, in_channels, spatial_dims)
        self._texture_gen = TextureGenerator(base_unet, epsilon=eps).to(device)

        # --- Shared optimizer for both generators ---
        opt_cfg = get_config(cfg, "ue.generator_optimizer", DictConfig({}))
        lr = float(get_config(opt_cfg, "lr", 1e-4))
        weight_decay = float(get_config(opt_cfg, "weight_decay", 1e-5))
        betas = tuple(get_config(opt_cfg, "betas", (0.9, 0.999)))

        all_params = list(self._contour_gen.parameters()) + \
            list(self._texture_gen.parameters())
        self._opt_generators = torch.optim.Adam(
            all_params, lr=lr, weight_decay=weight_decay, betas=betas,
        )

        self._initialized = True
        self.logger.info(
            f"[UMed] Initialized: in_ch={in_channels}, eps={eps:.6f}, "
            f"lr_g={lr}, gc_channels={gc_channels}, theta={gc_theta}"
        )

    # ------------------------------------------------------------------ #
    # Surrogate step  (S-step)
    # ------------------------------------------------------------------ #
    def surrogate_step_batch(self, trainer, batch) -> Dict[str, float]:
        """Update surrogate only, using stored noise from backend."""
        cfg = trainer.config
        device = trainer.device
        nb = trainer.noise_backend
        if nb is None:
            raise RuntimeError("[UE] noise_backend is required.")

        x = batch["image"].to(device).float()
        y = batch["label"]
        y = y.to(device).long() if torch.is_tensor(y) else torch.as_tensor(
            y, device=device, dtype=torch.long)
        keys: Iterable[int] = batch["key"]

        B, C_in = x.shape[:2]
        spatial_dims = len(x.shape) - 2
        self._init_generators(trainer, C_in, spatial_dims)

        mean = tuple(get_config(cfg, "training.data.transforms.mean",
                                (0.0,) * C_in))
        std = tuple(get_config(cfg, "training.data.transforms.std",
                               (1.0,) * C_in))

        delta = nb.batch_noise(list(keys)).to(device).float()
        if delta.shape[:2] != x.shape[:2]:
            raise RuntimeError(
                f"[UE] noise shape mismatch: {tuple(delta.shape)} vs "
                f"{tuple(x.shape)}")

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
        return {"surrogate_loss": loss_val, "loss": loss_val}

    # ------------------------------------------------------------------ #
    # Noise step  (G-step: update generators)
    # ------------------------------------------------------------------ #
    def noise_step_batch(self, trainer, batch) -> Dict[str, float]:
        """
        Update both G_c and G_t generators.

        Process:
          1. Extract contour_mask and interior_mask from GT labels
          2. Compute LBP texture map from input image
          3. G_c(x) * contour_mask  -> delta_c
          4. G_t(x) adaptive-clipped by LBP * interior_mask -> delta_t
          5. x_p = clip[0,1](x + delta_c + delta_t)
          6. Forward through frozen surrogate, compute seg loss
          7. Backprop to update G_c, G_t
          8. Store combined delta to noise backend
        """
        cfg = trainer.config
        device = trainer.device
        nb = trainer.noise_backend
        if nb is None:
            raise RuntimeError("[UE] noise_backend is required.")

        x = batch["image"].to(device).float()
        y = batch["label"]
        y = y.to(device).long() if torch.is_tensor(y) else torch.as_tensor(
            y, device=device, dtype=torch.long)
        keys = batch["key"]
        keys_list: List[int] = list(keys)

        B, C_in = x.shape[:2]
        spatial_dims = len(x.shape) - 2
        self._init_generators(trainer, C_in, spatial_dims)

        algo = require_config(cfg, "ue.algorithm")
        params = require_config(algo, "params")
        eps = float(get_config(params, "epsilon", 4 / 255.0))
        num_steps = int(get_config(params, "noise_step", 1))
        boundary_width = int(get_config(params, "boundary_width", 3))

        mean = tuple(get_config(cfg, "training.data.transforms.mean",
                                (0.0,) * C_in))
        std = tuple(get_config(cfg, "training.data.transforms.std",
                               (1.0,) * C_in))

        seg_loss_fn = self._get_seg_loss(trainer)

        # -------- freeze surrogate --------
        if not trainer.surrogates:
            raise RuntimeError("[UE] No surrogate bound.")
        _, s_model = next(iter(trainer.surrogates.items()))
        s_model.eval()
        for p in s_model.parameters():
            p.requires_grad = False

        # -------- masks --------
        contour_mask, interior_mask = _extract_contour_and_interior(
            y, device, boundary_width=boundary_width)
        # Expand to C channels
        contour_mask_c = contour_mask.expand(-1, C_in, -1, -1, -1)
        interior_mask_c = interior_mask.expand(-1, C_in, -1, -1, -1)

        # -------- LBP texture map --------
        lbp_map = _compute_lbp_3d(x)  # [B, 1, D, H, W] in [0, 1]
        lbp_map_c = lbp_map.expand(-1, C_in, -1, -1, -1)

        # -------- train generators --------
        self._contour_gen.train()
        self._texture_gen.train()
        last_loss = torch.tensor(0.0, device=device)

        for _ in range(max(1, num_steps)):
            delta_c = self._contour_gen(x, contour_mask_c)
            delta_t = self._texture_gen(x, interior_mask_c, lbp_map_c)

            # Combined perturbation
            delta = delta_c + delta_t
            perturb_img = (x + delta).clamp(0.0, 1.0)
            xn = perturb_img.clone()
            self._norm_inplace(xn, mean, std)

            out = s_model(xn)
            logits = out[0] if isinstance(out, (tuple, list)) else out

            loss = seg_loss_fn(logits, y.unsqueeze(1))
            last_loss = loss.detach()

            self._opt_generators.zero_grad(set_to_none=True)
            loss.backward()
            self._opt_generators.step()

        # -------- Store noise to backend --------
        self._contour_gen.eval()
        self._texture_gen.eval()
        with torch.no_grad():
            final_dc = self._contour_gen(x, contour_mask_c)
            final_dt = self._texture_gen(x, interior_mask_c, lbp_map_c)
            final_delta = (final_dc + final_dt).clamp(-eps, eps)

        nb.commit_batch(keys_list, final_delta.detach().cpu())

        delta_linf = float(final_delta.detach().abs().max().cpu())
        return {
            "noise_loss": float(last_loss.cpu()),
            "delta_linf": delta_linf,
        }