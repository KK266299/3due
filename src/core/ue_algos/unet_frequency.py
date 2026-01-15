# file: src/core/ue_algos/unet_frequency.py
from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from monai.losses import DiceCELoss

from ...registry import register_plugin
from ...utils.config import get_config, require_config


def _gaussian_kernel_3d(sigma: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if sigma <= 0:
        raise ValueError("sigma must be > 0 for gaussian kernel.")
    radius = int(round(2.0 * sigma))
    coords = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g = g / g.sum()
    kernel_1d = g.view(1, 1, -1)
    kernel_3d = (
        kernel_1d[:, :, :, None, None]
        * kernel_1d[:, :, None, :, None]
        * kernel_1d[:, :, None, None, :]
    )
    return kernel_3d


@register_plugin("unet_frequency")
class UNetFrequencyUE:
    """
    Spectrally-coherent UE based on frequency-domain masking.

    This algorithm keeps MinMin's training flow, but enforces spectral
    structure on perturbations by projecting noise into a fixed frequency
    support mask on each update.
    """

    def __init__(self):
        self._seg_loss: DiceCELoss | None = None
        self._mask_cache: dict[Tuple[int, int, int], torch.Tensor] = {}

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

    def _build_spectral_mask(
        self, spatial_shape: Tuple[int, int, int], device: torch.device, dtype: torch.dtype, cfg
    ) -> torch.Tensor:
        z_size, y_size, x_size = spatial_shape
        mask_cfg = get_config(cfg, "ue.algorithm.params.spectral_mask", DictConfig({}))
        z_max = float(get_config(mask_cfg, "z_max", 0.125))
        xy_min = float(get_config(mask_cfg, "xy_min", 0.0))
        xy_max = float(get_config(mask_cfg, "xy_max", 0.5))

        fz = torch.fft.fftfreq(z_size, device=device, dtype=dtype).abs()
        fy = torch.fft.fftfreq(y_size, device=device, dtype=dtype).abs()
        fx = torch.fft.fftfreq(x_size, device=device, dtype=dtype).abs()
        zz, yy, xx = torch.meshgrid(fz, fy, fx, indexing="ij")
        fxy = torch.sqrt(yy**2 + xx**2)

        mask = (zz <= z_max) & (fxy >= xy_min) & (fxy <= xy_max)
        mask = mask.to(dtype=dtype)
        return mask.unsqueeze(0).unsqueeze(0)

    def _get_spectral_mask(
        self, spatial_shape: Tuple[int, int, int], device: torch.device, dtype: torch.dtype, cfg
    ) -> torch.Tensor:
        if spatial_shape not in self._mask_cache:
            self._mask_cache[spatial_shape] = self._build_spectral_mask(
                spatial_shape, device, dtype, cfg
            )
        return self._mask_cache[spatial_shape].to(device=device, dtype=dtype)

    @staticmethod
    def _apply_spectral_mask(delta: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        delta_fft = torch.fft.fftn(delta, dim=(-3, -2, -1))
        delta_fft = delta_fft * mask
        delta_spatial = torch.fft.ifftn(delta_fft, dim=(-3, -2, -1)).real
        return delta_spatial

    @staticmethod
    def _roi_gate(delta: torch.Tensor, labels: torch.Tensor, sigma: float) -> torch.Tensor:
        if labels.dim() == 5:
            labels = labels.squeeze(1)
        roi_mask = (labels > 0).float().unsqueeze(1)
        if sigma > 0:
            kernel = _gaussian_kernel_3d(sigma, device=roi_mask.device, dtype=roi_mask.dtype)
            padding = kernel.shape[-1] // 2
            roi_mask = F.conv3d(roi_mask, kernel, padding=padding)
            roi_mask = roi_mask / roi_mask.max().clamp_min(1e-6)
        return delta * roi_mask

    # ---------------- Surrogate-step: Update surrogate ---------------- #
    def surrogate_step_batch(self, trainer, batch) -> Dict[str, float]:
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

        _, C_in = x.shape[:2]
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

    # ---------------- N-step: Update noise with spectral mask ---------------- #
    def noise_step_batch(self, trainer, batch) -> Dict[str, float]:
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

        _, C_in = x.shape[:2]

        algo = require_config(cfg, "ue.algorithm")
        params = require_config(algo, "params")
        eps_cfg = float(get_config(params, "epsilon", 4 / 255.0))
        eps = min(eps_cfg, 4 / 255.0)
        step_size = float(get_config(params, "step_size", 2 / 255.0))
        num_steps = int(get_config(params, "noise_step", 10))

        roi_cfg = get_config(params, "roi_gate", DictConfig({}))
        roi_enabled = bool(get_config(roi_cfg, "enabled", False))
        roi_sigma = float(get_config(roi_cfg, "smooth_sigma", 0.0))

        mean = tuple(get_config(cfg, "training.data.transforms.mean", (0.0,) * C_in))
        std = tuple(get_config(cfg, "training.data.transforms.std", (1.0,) * C_in))

        seg_loss_fn = self._get_seg_loss(trainer)

        if not trainer.surrogates:
            raise RuntimeError("[UE] No surrogate bound.")
        _, s_model = next(iter(trainer.surrogates.items()))
        s_model.eval()
        for p in s_model.parameters():
            p.requires_grad = False

        delta_tbl = nb.batch_noise(keys_list).to(device).float()
        if delta_tbl.shape[:2] != x.shape[:2]:
            raise RuntimeError(
                f"[UE] noise shape mismatch: noise {tuple(delta_tbl.shape)} vs input {tuple(x.shape)}"
            )

        delta_tbl = delta_tbl.clamp(-eps, eps)

        spatial_shape = tuple(delta_tbl.shape[-3:])
        mask = self._get_spectral_mask(spatial_shape, device, delta_tbl.dtype, cfg)

        last_loss = torch.tensor(0.0, device=device)

        with torch.enable_grad():
            for _ in range(max(1, num_steps)):
                perturb_img = (x + delta_tbl).clamp(0.0, 1.0).detach().requires_grad_(True)
                xn = perturb_img.clone()
                self._norm_inplace(xn, mean, std)

                out = s_model(xn)
                logits = out[0] if isinstance(out, (tuple, list)) else out

                loss = seg_loss_fn(logits, y.unsqueeze(1))
                last_loss = loss.detach()

                (g,) = torch.autograd.grad(
                    loss, perturb_img, retain_graph=False, create_graph=False
                )

                delta_tbl = delta_tbl - step_size * g.sign()
                delta_tbl = self._apply_spectral_mask(delta_tbl, mask)
                if roi_enabled:
                    delta_tbl = self._roi_gate(delta_tbl, y, roi_sigma)
                delta_tbl = delta_tbl.clamp(-eps, eps)

        nb.commit_batch(keys_list, delta_tbl.detach().cpu())

        delta_linf = float(delta_tbl.detach().abs().max().cpu())
        return {
            "noise_loss": float(last_loss.cpu()),
            "delta_linf": delta_linf,
        }
