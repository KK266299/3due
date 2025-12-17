# file: src/core/ue_algos/unet_noise.py
"""
UNet-based Noise Generator for 3D Segmentation UE.

This module replaces PGD-based noise update in MinMin with a learnable
UNet noise generator. The generator takes [image, current_noise] as input
and outputs noise UPDATE (delta_update), similar to PGD iteration.

Key difference from PGD:
  - PGD:  delta = delta - step_size * sign(gradient)
  - UNet: delta = delta + UNet([x, delta])  (learned update)

Usage:
    method=unet_noise
"""
from __future__ import annotations
from typing import Dict, Iterable, List, Optional

import torch
import torch.nn as nn
from omegaconf import DictConfig
from monai.losses import DiceCELoss
from monai.networks.nets import UNet as MonaiUNet

from ...registry import register_plugin
from ...utils.config import get_config, require_config


class NoiseGenerator(nn.Module):
    """
    UNet-based noise UPDATE generator.

    Input:  [image, current_noise] concatenated [B, 2*C_in, D, H, W]
    Output: noise_update [B, C_in, D, H, W]

    Update rule:
        delta_new = clamp(delta_old + delta_update, -eps, eps)
        Ensure: x + delta_new ∈ [0, 1]
    """
    def __init__(
        self,
        in_channels: int,
        spatial_dims: int = 3,
        channels: List[int] = None,
        strides: List[int] = None,
        num_res_units: int = 2,
        norm: str = "INSTANCE",
        act: str = "LEAKYRELU",
        dropout: float = 0.0,
        epsilon: float = 8/255.0,
        step_size: float = 2/255.0,
    ):
        super().__init__()

        if channels is None:
            channels = [16, 32, 64, 128]
        if strides is None:
            strides = [2, 2, 2]

        self.epsilon = epsilon
        self.step_size = step_size

        # UNet backbone: [image, noise] concat -> noise_update
        # Input: 2 * in_channels (image + current noise)
        # Output: in_channels (noise update)
        self.unet = MonaiUNet(
            spatial_dims=spatial_dims,
            in_channels=in_channels * 2,  # [x, delta] concatenated
            out_channels=in_channels,      # Output noise update
            channels=channels,
            strides=strides,
            num_res_units=num_res_units,
            act=act,
            norm=norm,
            dropout=dropout,
        )

    def forward(
        self,
        x: torch.Tensor,
        delta: torch.Tensor
    ) -> torch.Tensor:
        """
        Generate noise UPDATE (not the full noise).

        Args:
            x:     Input image [B, C, D, H, W]
            delta: Current noise [B, C, D, H, W]

        Returns:
            delta_new: Updated noise [B, C, D, H, W] satisfying:
                       - |delta_new| <= epsilon (L_inf constraint)
                       - x + delta_new ∈ [0, 1] (valid image constraint)
        """
        # Concatenate image and current noise
        concat_input = torch.cat([x, delta], dim=1)  # [B, 2*C, D, H, W]

        # Generate update direction (like sign(gradient) in PGD)
        raw_update = self.unet(concat_input)

        # Use tanh to bound update magnitude, scale by step_size
        # This replaces: step_size * sign(gradient)
        delta_update = torch.tanh(raw_update) * self.step_size

        # Apply update: delta_new = delta_old + delta_update
        delta_new = delta + delta_update

        # Constraint 1: L_inf bound |delta| <= epsilon
        delta_new = delta_new.clamp(-self.epsilon, self.epsilon)

        # Constraint 2: x + delta ∈ [0, 1]
        # delta >= -x  (so x + delta >= 0)
        # delta <= 1-x (so x + delta <= 1)
        delta_new = torch.max(delta_new, -x)
        delta_new = torch.min(delta_new, 1.0 - x)

        return delta_new


@register_plugin("unet_noise")
class UNetNoiseUE:
    """
    UNet-based Noise Generator for 3D segmentation (e.g., BraTS19).

    This replaces PGD noise updates with a learnable UNet noise generator.
    The generator learns to produce adversarial noise that minimizes
    segmentation loss.

    Batch requirements:
      - batch["image"]: FloatTensor [B, C, D, H, W]
      - batch["label"]: LongTensor  [B, D, H, W]
      - batch["key"]:   sample-wise key (for noise backend storage)

    Surrogate output:
      - s_model(x) -> logits: [B, C_seg, D, H, W]

    Noise backend (for caching/export):
      - noise_backend.batch_noise(keys) -> [N, C_in, D, H, W]
      - noise_backend.commit_batch(keys, delta) writes back updated noise
    """

    def __init__(self):
        self._seg_loss: Optional[DiceCELoss] = None
        self._noise_gen: Optional[NoiseGenerator] = None
        self._noise_opt: Optional[torch.optim.Optimizer] = None

    @staticmethod
    def _norm_inplace(x: torch.Tensor, mean, std):
        """In-place per-channel normalize for ND volume."""
        for c, (m, s) in enumerate(zip(mean, std)):
            x[:, c].sub_(float(m)).div_(float(s))
        return x

    def _get_seg_loss(self, trainer) -> DiceCELoss:
        """Build DiceCELoss consistent with SegTrainer config."""
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

    def _get_noise_generator(self, trainer) -> NoiseGenerator:
        """Build or return cached noise generator."""
        if self._noise_gen is not None:
            return self._noise_gen

        cfg = trainer.config
        device = trainer.device

        algo = require_config(cfg, "ue.algorithm")
        params = require_config(algo, "params")
        eps = float(get_config(params, "epsilon", 8 / 255.0))
        step_size = float(get_config(params, "step_size", 2 / 255.0))

        # Generator config - from ue.noise_generator or defaults
        gen_cfg = get_config(cfg, "ue.noise_generator", DictConfig({}))
        in_channels = int(get_config(gen_cfg, "in_channels", 4))
        spatial_dims = int(get_config(gen_cfg, "spatial_dims", 3))
        channels = list(get_config(gen_cfg, "channels", [16, 32, 64, 128]))
        strides = list(get_config(gen_cfg, "strides", [2, 2, 2]))
        num_res_units = int(get_config(gen_cfg, "num_res_units", 2))
        norm = str(get_config(gen_cfg, "norm", "INSTANCE"))
        act = str(get_config(gen_cfg, "act", "LEAKYRELU"))
        dropout = float(get_config(gen_cfg, "dropout", 0.0))

        self._noise_gen = NoiseGenerator(
            in_channels=in_channels,
            spatial_dims=spatial_dims,
            channels=channels,
            strides=strides,
            num_res_units=num_res_units,
            norm=norm,
            act=act,
            dropout=dropout,
            epsilon=eps,
            step_size=step_size,
        ).to(device)

        # Build optimizer for noise generator
        opt_cfg = get_config(gen_cfg, "optimizer", DictConfig({"name": "adam", "lr": 1e-4}))
        opt_name = str(get_config(opt_cfg, "name", "adam")).lower()
        lr = float(get_config(opt_cfg, "lr", 1e-4))
        weight_decay = float(get_config(opt_cfg, "weight_decay", 0.0))

        if opt_name == "adam":
            betas = tuple(get_config(opt_cfg, "betas", (0.9, 0.999)))
            self._noise_opt = torch.optim.Adam(
                self._noise_gen.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                betas=betas,
            )
        elif opt_name == "sgd":
            momentum = float(get_config(opt_cfg, "momentum", 0.9))
            self._noise_opt = torch.optim.SGD(
                self._noise_gen.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                momentum=momentum,
            )
        else:
            self._noise_opt = torch.optim.Adam(
                self._noise_gen.parameters(),
                lr=lr,
                weight_decay=weight_decay,
            )

        return self._noise_gen

    # ---------------- Surrogate-step: Update surrogate ---------------- #
    def surrogate_step_batch(self, trainer, batch) -> Dict[str, float]:
        """
        Update surrogate parameters only, noise is fixed.
        Loss: segmentation DiceCE
        """
        cfg = trainer.config
        device = trainer.device
        nb = trainer.noise_backend
        if nb is None:
            raise RuntimeError("[UE] noise_backend is required.")

        # data
        x = batch["image"].to(device).float()
        y = batch["label"]
        y = y.to(device).long() if torch.is_tensor(y) else torch.as_tensor(
            y, device=device, dtype=torch.long
        )
        keys: Iterable[int] = batch["key"]

        B, C_in = x.shape[:2]

        # normalization config
        mean = tuple(get_config(cfg, "training.data.transforms.mean", (0.0,) * C_in))
        std = tuple(get_config(cfg, "training.data.transforms.std", (1.0,) * C_in))

        # Get noise from backend (previously generated by noise_step)
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

    # ---------------- N-step: Update noise via UNet generator ---------------- #
    def noise_step_batch(self, trainer, batch) -> Dict[str, float]:
        """
        Update noise generator to minimize segmentation loss.

        Key difference from PGD:
          - PGD iterates multiple steps per batch, updating only delta
          - UNet generator updates its parameters once per batch
          - noise_step controls gradient accumulation steps (default=1)
        """
        cfg = trainer.config
        device = trainer.device
        nb = trainer.noise_backend
        if nb is None:
            raise RuntimeError("[UE] noise_backend is required.")

        # -------- data & config --------
        x = batch["image"].to(device).float()
        y = batch["label"]
        y = y.to(device).long() if torch.is_tensor(y) else torch.as_tensor(
            y, device=device, dtype=torch.long
        )
        keys = batch["key"]
        keys_list: List[int] = list(keys)

        N, C_in = x.shape[:2]

        algo = require_config(cfg, "ue.algorithm")
        params = require_config(algo, "params")
        eps = float(get_config(params, "epsilon", 8 / 255.0))

        # normalization config
        mean = tuple(get_config(cfg, "training.data.transforms.mean", (0.0,) * C_in))
        std = tuple(get_config(cfg, "training.data.transforms.std", (1.0,) * C_in))

        seg_loss_fn = self._get_seg_loss(trainer)
        noise_gen = self._get_noise_generator(trainer)
        noise_opt = self._noise_opt

        # -------- freeze surrogate --------
        if not trainer.surrogates:
            raise RuntimeError("[UE] No surrogate bound.")
        _, s_model = next(iter(trainer.surrogates.items()))
        s_model.eval()
        for p in s_model.parameters():
            p.requires_grad = False

        # -------- Get current noise from backend --------
        delta = nb.batch_noise(keys_list).to(device).float()
        if delta.shape[:2] != x.shape[:2]:
            raise RuntimeError(
                f"[UE] noise shape mismatch: noise {tuple(delta.shape)} vs input {tuple(x.shape)}"
            )
        delta = delta.clamp(-eps, eps)

        # -------- Single forward-backward pass (efficient) --------
        noise_gen.train()

        # UNet generates updated noise
        delta_new = noise_gen(x, delta)

        # Apply noise to image
        perturb_img = (x + delta_new).clamp(0.0, 1.0)
        xn = perturb_img.clone()
        self._norm_inplace(xn, mean, std)

        out = s_model(xn)
        logits = out[0] if isinstance(out, (tuple, list)) else out

        # Minimize segmentation loss
        loss = seg_loss_fn(logits, y.unsqueeze(1))

        # Update generator (single step per batch)
        noise_opt.zero_grad(set_to_none=True)
        loss.backward()
        noise_opt.step()

        # -------- Write back noise to backend for export --------
        with torch.no_grad():
            noise_gen.eval()
            delta_final = noise_gen(x, delta)
            delta_final = delta_final.clamp(-eps, eps)
            nb.commit_batch(keys_list, delta_final.detach().cpu())

        delta_linf = float(delta_final.detach().abs().max().cpu())
        return {
            "noise_loss": float(loss.detach().cpu()),
            "delta_linf": delta_linf,
        }