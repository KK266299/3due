# file: src/core/ue_algos/unet_noise_slice.py
"""
Slice-by-Slice UNet Noise Generator for 3D segmentation (e.g., BraTS19).

This algorithm generates noise independently for each 2D slice using a 2D UNet,
then assembles the per-slice noise back into a 3D volume.

Core idea:
  - Instead of using a 3D UNet on the entire volume, use a 2D UNet
  - Process each depth slice independently: x[:,:,d,:,:] → noise[:,:,d,:,:]
  - Assemble per-slice noise into full 3D noise volume
  - This approach is lighter in memory and allows independent per-slice noise patterns

Key advantages:
  - Lower GPU memory: 2D UNet is much smaller than 3D UNet
  - Independent per-slice noise: each slice can have unique noise patterns
  - Matches the anisotropic nature of medical imaging (e.g., CT/MRI slice acquisition)

Core differences from UNetNoise:
  - UNetNoise: 3D UNet generates noise for entire volume at once
  - UNetNoiseSlice: 2D UNet generates noise for each slice independently

Training loop:
  1. Surrogate-step: Same as UNetNoise, train 3D surrogate on noisy images
  2. Noise-step:
     - Reshape volume slices to batch: [B,C,D,H,W] → [B*D,C,H,W]
     - Forward through 2D noise UNet to get per-slice noise
     - Reshape back: [B*D,C,H,W] → [B,C,D,H,W]
     - Add noise to image, pass through 3D surrogate, compute loss
     - Backprop to update 2D noise UNet parameters
     - Store generated noise to backend

Maximum noise bound: 8/255 ≈ 0.0313725 (L∞ constraint)
"""
from __future__ import annotations
from typing import Dict, Iterable, List, Optional, Any

import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from monai.losses import DiceCELoss
from monai.networks.nets import UNet as MonaiUNet

from ...registry import register_plugin
from ...utils.config import get_config, require_config
from ...utils.logger import get_logger


def _build_noise_unet_2d(cfg: DictConfig, in_channels: int) -> nn.Module:
    """
    Build a 2D U-Net for per-slice noise generation.
    Input: single slice [B, C, H, W]
    Output: noise for single slice [B, C, H, W] (same shape as input)
    """
    channels = list(get_config(cfg, "channels", [16, 32, 64, 128]))
    strides = list(get_config(cfg, "strides", [2, 2, 2]))
    num_res_units = int(get_config(cfg, "num_res_units", 1))
    act = get_config(cfg, "act", "LEAKYRELU")
    norm = get_config(cfg, "norm", "INSTANCE")
    dropout = float(get_config(cfg, "dropout", 0.0))

    unet = MonaiUNet(
        spatial_dims=2,  # 2D UNet for per-slice processing
        in_channels=in_channels,
        out_channels=in_channels,  # output same channels as input
        channels=channels,
        strides=strides,
        num_res_units=num_res_units,
        act=act,
        norm=norm,
        dropout=dropout,
    )
    return unet


class NoiseUNet2DWrapper(nn.Module):
    """
    Wrapper for 2D noise U-Net that processes 3D volumes slice by slice.

    Takes a 3D volume [B, C, D, H, W], reshapes to [B*D, C, H, W],
    passes through the 2D UNet, then reshapes back to [B, C, D, H, W].

    Supports chunked processing for memory efficiency when B*D is large.
    """
    def __init__(self, unet: nn.Module, epsilon: float = 8/255, slice_batch_size: int = 0):
        super().__init__()
        self.unet = unet
        self.epsilon = epsilon
        self.slice_batch_size = slice_batch_size  # 0 = process all slices at once

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input image [B, C, D, H, W] in [0, 1]
        Returns:
            Noise [B, C, D, H, W] in [-eps, eps]
        """
        B, C, D, H, W = x.shape

        # Reshape: [B, C, D, H, W] → [B*D, C, H, W]
        # permute to [B, D, C, H, W] then reshape to [B*D, C, H, W]
        x_2d = x.permute(0, 2, 1, 3, 4).reshape(B * D, C, H, W)

        if self.slice_batch_size > 0 and B * D > self.slice_batch_size:
            # Chunked processing for memory efficiency
            chunks = []
            for i in range(0, B * D, self.slice_batch_size):
                chunk = x_2d[i:i + self.slice_batch_size]
                raw_chunk = self.unet(chunk)
                chunks.append(torch.tanh(raw_chunk) * self.epsilon)
            noise_2d = torch.cat(chunks, dim=0)
        else:
            # Process all slices at once
            raw_noise = self.unet(x_2d)
            noise_2d = torch.tanh(raw_noise) * self.epsilon

        # Reshape back: [B*D, C, H, W] → [B, D, C, H, W] → [B, C, D, H, W]
        noise = noise_2d.reshape(B, D, C, H, W).permute(0, 2, 1, 3, 4)

        return noise


@register_plugin("unet_noise_slice")
class UNetNoiseSliceUE:
    """
    Slice-by-Slice UNet noise generation for 3D segmentation UE.

    Instead of using a 3D UNet on the entire volume, this method uses a 2D UNet
    to generate noise independently for each depth slice. The per-slice noise is
    then assembled back into a 3D volume.

    Assumptions (same as UNetNoise):
      - Batch:
          batch["image"]: FloatTensor [B, C, D, H, W] (3D volume)
          batch["label"]: LongTensor  [B, D, H, W]
          batch["key"]:   sample-wise key
      - Surrogate:
          s_model(x) -> logits: [B, C_seg, D, H, W]
      - Noise backend:
          noise_backend.batch_noise(keys) -> [N, C_in, D, H, W]

    Additional components:
      - noise_unet: 2D U-Net that generates per-slice noise
      - opt_noise_unet: Optimizer for the 2D noise U-Net
    """

    def __init__(self):
        self._seg_loss: DiceCELoss | None = None
        self._noise_unet: NoiseUNet2DWrapper | None = None
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

    def _init_noise_unet(self, trainer, in_channels: int):
        """
        Initialize the 2D noise U-Net and its optimizer (lazy, once).
        """
        if self._initialized:
            return

        cfg = trainer.config
        device = trainer.device

        # Get noise UNet config
        noise_unet_cfg = get_config(cfg, "ue.noise_unet", DictConfig({}))
        params = get_config(cfg, "ue.algorithm.params", DictConfig({}))
        eps = float(get_config(params, "epsilon", 8/255))
        slice_batch_size = int(get_config(params, "slice_batch_size", 0))

        # Build 2D noise UNet
        base_unet = _build_noise_unet_2d(noise_unet_cfg, in_channels)
        self._noise_unet = NoiseUNet2DWrapper(
            base_unet, epsilon=eps, slice_batch_size=slice_batch_size
        )
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

        self._initialized = True
        self.logger.info(
            f"[UNetNoiseSlice] Initialized 2D noise UNet: in_channels={in_channels}, "
            f"eps={eps:.6f}, lr={lr}, slice_batch_size={slice_batch_size}"
        )

    # ---------------- Surrogate-step: Update surrogate ---------------- #
    def surrogate_step_batch(self, trainer, batch) -> Dict[str, float]:
        """
        Update surrogate parameters only, do not update noise.
        Uses noise from backend (3D noise volumes).

        This is identical to UNetNoise: the surrogate trains on 3D noisy images.
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

        # Initialize noise UNet if not done
        self._init_noise_unet(trainer, C_in)

        # normalization config
        mean = tuple(get_config(cfg, "training.data.transforms.mean", (0.0,) * C_in))
        std = tuple(get_config(cfg, "training.data.transforms.std", (1.0,) * C_in))

        # Get noise from backend (3D noise)
        delta = nb.batch_noise(list(keys)).to(device).float()  # [B, C_in, D, H, W]
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

    # ---------------- N-step: Update noise via 2D UNet (slice-by-slice) ---------------- #
    def noise_step_batch(self, trainer, batch) -> Dict[str, float]:
        """
        Update noise using a 2D UNet that processes each slice independently.

        Process:
          1. Reshape 3D volume to 2D slices: [B,C,D,H,W] → [B*D,C,H,W]
          2. Forward through 2D noise UNet to get per-slice noise
          3. Reshape back to 3D: [B*D,C,H,W] → [B,C,D,H,W]
          4. Add noise to 3D image, normalize, pass through frozen 3D surrogate
          5. Compute segmentation loss
          6. Backprop to update 2D noise UNet parameters
          7. Store generated 3D noise to backend
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

        # Initialize if needed
        self._init_noise_unet(trainer, C_in)

        algo = require_config(cfg, "ue.algorithm")
        params = require_config(algo, "params")
        eps = float(get_config(params, "epsilon", 8 / 255.0))
        num_steps = int(get_config(params, "noise_step", 1))

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

        # -------- train 2D noise UNet --------
        self._noise_unet.train()
        last_loss = torch.tensor(0.0, device=device)

        for _ in range(max(1, num_steps)):
            # Generate per-slice noise using 2D UNet
            # Internally: [B,C,D,H,W] → [B*D,C,H,W] → 2D UNet → [B,C,D,H,W]
            delta = self._noise_unet(x)  # [B, C_in, D, H, W], already in [-eps, eps]

            # Create perturbed image (3D)
            perturb_img = (x + delta).clamp(0.0, 1.0)
            xn = perturb_img.clone()
            self._norm_inplace(xn, mean, std)

            # Forward through 3D surrogate
            out = s_model(xn)
            logits = out[0] if isinstance(out, (tuple, list)) else out

            # Compute loss (min-min: minimize loss)
            loss = seg_loss_fn(logits, y.unsqueeze(1))
            last_loss = loss.detach()

            # Backprop to update 2D noise UNet
            self._opt_noise_unet.zero_grad(set_to_none=True)
            loss.backward()
            self._opt_noise_unet.step()

        # -------- Store generated noise to backend --------
        # Generate final noise and commit
        self._noise_unet.eval()
        with torch.no_grad():
            final_delta = self._noise_unet(x)  # [B, C_in, D, H, W]
            # Extra clamp for safety
            final_delta = final_delta.clamp(-eps, eps)

        nb.commit_batch(keys_list, final_delta.detach().cpu())

        delta_linf = float(final_delta.detach().abs().max().cpu())
        return {
            "noise_loss": float(last_loss.cpu()),
            "delta_linf": delta_linf,
        }