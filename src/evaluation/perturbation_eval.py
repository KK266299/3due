# file: src/evaluation/perturbation_eval.py
"""
Perturbation quality evaluation using IQA-PyTorch.

Evaluates adversarial perturbation quality with metrics:
  - PSNR (Peak Signal-to-Noise Ratio) ↑
  - SSIM (Structural Similarity Index) ↑
  - LPIPS (Learned Perceptual Image Patch Similarity) ↓ (optional)

Supports both 2D and 3D medical images.

Usage:
    from src.evaluation.perturbation_eval import PerturbationQualityEvaluator

    evaluator = PerturbationQualityEvaluator(use_pyiqa=True)
    metrics = evaluator.evaluate_batch(original, perturbed)
"""
from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Any
import math

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader
from omegaconf import DictConfig
from tqdm import tqdm
import numpy as np

from ..utils.config import get_config
from ..registry import register_evaluation_strategy

# Import built-in metrics
from ..utils.eval_metrics import (
    compute_psnr,
    compute_ssim,
    compute_noise_jacobian_metrics,
    HAS_PYIQA,
)

# Conditional import for IQA-PyTorch
if HAS_PYIQA:
    from ..utils.eval_metrics import IQAPyTorchMetrics


class PerturbationQualityEvaluator:
    """
    Evaluator for adversarial perturbation quality.

    Computes image quality metrics between original and perturbed images.
    Supports both built-in metrics and IQA-PyTorch metrics.
    """

    def __init__(
        self,
        data_range: float = 1.0,
        use_pyiqa: bool = True,
        pyiqa_metrics: List[str] = ['psnr', 'ssim'],
        device: Optional[str] = None,
    ):
        """
        Initialize perturbation quality evaluator.

        Args:
            data_range: Value range of images (1.0 for [0,1], 255 for [0,255])
            use_pyiqa: Whether to use IQA-PyTorch (if available)
            pyiqa_metrics: List of IQA-PyTorch metrics to compute
            device: Device to use (auto-detect if None)
        """
        self.data_range = data_range
        self.use_pyiqa = use_pyiqa and HAS_PYIQA
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        # Initialize IQA-PyTorch if available and requested
        self._iqa_evaluator = None
        if self.use_pyiqa:
            try:
                self._iqa_evaluator = IQAPyTorchMetrics(
                    metrics=pyiqa_metrics,
                    device=self.device,
                )
            except Exception as e:
                print(f"Warning: Failed to initialize IQA-PyTorch: {e}")
                self.use_pyiqa = False

    def compute_metrics_2d(
        self,
        original: Tensor,
        perturbed: Tensor,
    ) -> Dict[str, float]:
        """
        Compute metrics for 2D images.

        Args:
            original: [B, C, H, W] original images
            perturbed: [B, C, H, W] perturbed images

        Returns:
            Dict of metric values
        """
        results = {}

        # Built-in PSNR
        psnr = compute_psnr(original, perturbed, data_range=self.data_range)
        results['psnr_builtin'] = float(psnr.mean().item())

        # Built-in SSIM (via 3D version treating as [B,C,1,H,W])
        orig_3d = original.unsqueeze(2)
        pert_3d = perturbed.unsqueeze(2)
        ssim = compute_ssim(orig_3d, pert_3d, data_range=self.data_range)
        results['ssim_builtin'] = float(ssim.item())

        # IQA-PyTorch metrics
        if self.use_pyiqa and self._iqa_evaluator is not None:
            iqa_results = self._iqa_evaluator.compute_2d(original, perturbed)
            for k, v in iqa_results.items():
                results[f'{k}_iqa'] = v

        return results

    def compute_metrics_3d(
        self,
        original: Tensor,
        perturbed: Tensor,
        noise: Optional[Tensor] = None,
        sample_slices: Optional[int] = None,
    ) -> Dict[str, float]:
        """
        Compute metrics for 3D volumes.

        Args:
            original: [B, C, D, H, W] original volumes
            perturbed: [B, C, D, H, W] perturbed volumes
            noise: Optional [B, C, D, H, W] noise tensor
            sample_slices: If set, sample this many slices for IQA-PyTorch

        Returns:
            Dict of metric values
        """
        results = {}

        # Built-in PSNR (volumetric)
        psnr = compute_psnr(original, perturbed, data_range=self.data_range)
        results['psnr'] = float(psnr.mean().item())

        # Built-in SSIM (volumetric, 3D Gaussian window)
        ssim = compute_ssim(original, perturbed, data_range=self.data_range)
        results['ssim'] = float(ssim.item())

        # Noise smoothness metrics
        if noise is not None:
            noise_metrics = compute_noise_jacobian_metrics(noise)
            results.update(noise_metrics)

        # IQA-PyTorch metrics (slice-wise)
        if self.use_pyiqa and self._iqa_evaluator is not None:
            iqa_results = self._iqa_evaluator.compute_3d_slicewise(
                original, perturbed, sample_slices=sample_slices
            )
            for k, v in iqa_results.items():
                results[f'{k}_iqa'] = v

        return results

    def compute(
        self,
        original: Tensor,
        perturbed: Tensor,
        noise: Optional[Tensor] = None,
        **kwargs,
    ) -> Dict[str, float]:
        """
        Auto-detect 2D/3D and compute metrics.

        Args:
            original: Original image tensor
            perturbed: Perturbed image tensor
            noise: Optional noise tensor

        Returns:
            Dict of metric values
        """
        ndim = original.dim()

        if ndim <= 4:  # [C,H,W] or [B,C,H,W]
            if ndim == 3:
                original = original.unsqueeze(0)
                perturbed = perturbed.unsqueeze(0)
            return self.compute_metrics_2d(original, perturbed)
        else:  # [C,D,H,W] or [B,C,D,H,W]
            if ndim == 4:
                original = original.unsqueeze(0)
                perturbed = perturbed.unsqueeze(0)
                if noise is not None:
                    noise = noise.unsqueeze(0)
            return self.compute_metrics_3d(original, perturbed, noise, **kwargs)


@register_evaluation_strategy("perturbation_quality")
class PerturbationQualityEvaluationStrategy:
    """
    Evaluation strategy for perturbation quality assessment.

    Requires dataset to return both clean and perturbed images,
    or a noise accessor to compute perturbations.

    Config:
        evaluation.perturbation:
            data_range: 1.0
            use_pyiqa: true
            pyiqa_metrics: [psnr, ssim]
            sample_slices: null  # Sample N slices for IQA-PyTorch (null = all)
    """

    def __init__(self, config: Optional[DictConfig] = None):
        self.config = config or DictConfig({})

        pert_cfg = get_config(self.config, "evaluation.perturbation", DictConfig({}))
        data_range = float(get_config(pert_cfg, "data_range", 1.0))
        use_pyiqa = bool(get_config(pert_cfg, "use_pyiqa", True))
        pyiqa_metrics = list(get_config(pert_cfg, "pyiqa_metrics", ['psnr', 'ssim']))
        self.sample_slices = get_config(pert_cfg, "sample_slices", None)

        self.evaluator = PerturbationQualityEvaluator(
            data_range=data_range,
            use_pyiqa=use_pyiqa,
            pyiqa_metrics=pyiqa_metrics,
        )

        # Accumulators
        self._metrics_sum: Dict[str, float] = {}
        self._n_samples: int = 0

    def reset(self):
        """Reset accumulators."""
        self._metrics_sum = {}
        self._n_samples = 0

    @torch.no_grad()
    def evaluate_batch(
        self,
        original: Tensor,
        perturbed: Tensor,
        noise: Optional[Tensor] = None,
    ) -> Dict[str, float]:
        """
        Evaluate a single batch.

        Args:
            original: Original images [B, C, ...]
            perturbed: Perturbed images [B, C, ...]
            noise: Optional noise tensor

        Returns:
            Batch metrics
        """
        metrics = self.evaluator.compute(
            original, perturbed, noise,
            sample_slices=self.sample_slices,
        )

        # Accumulate
        bs = original.shape[0]
        for k, v in metrics.items():
            if not math.isnan(v):
                self._metrics_sum[k] = self._metrics_sum.get(k, 0.0) + v * bs
        self._n_samples += bs

        return metrics

    def aggregate(self) -> Dict[str, float]:
        """Aggregate accumulated metrics."""
        if self._n_samples == 0:
            return {}

        results = {}
        for k, v in self._metrics_sum.items():
            results[k] = v / self._n_samples

        return results

    @torch.no_grad()
    def evaluate_with_noise_accessor(
        self,
        data_loader: DataLoader,
        noise_accessor,
        key_spec: Dict[str, Any],
        device: torch.device,
    ) -> Dict[str, float]:
        """
        Evaluate perturbation quality using a noise accessor.

        Args:
            data_loader: DataLoader for clean images
            noise_accessor: UEShardsAccessor or similar
            key_spec: Key specification for noise lookup
            device: Device to use

        Returns:
            Aggregated metrics
        """
        from ..core.ue_keys import extract_key

        self.reset()

        pbar = tqdm(data_loader, desc="Evaluate Perturbation Quality", leave=False)
        for batch_idx, batch in enumerate(pbar):
            x = batch["image"].to(device).float()

            # Get noise for each sample
            batch_noise = []
            keys = batch.get("key", list(range(batch_idx * x.shape[0], (batch_idx + 1) * x.shape[0])))

            for i, sample_key in enumerate(keys):
                if hasattr(noise_accessor, 'get'):
                    noise = noise_accessor.get(sample_key)
                else:
                    noise = noise_accessor.get_noise(sample_key)
                batch_noise.append(noise)

            noise = torch.stack(batch_noise, dim=0).to(device).float()

            # Ensure noise shape matches image
            if noise.shape[1:] != x.shape[1:]:
                # Handle channel mismatch
                if noise.shape[1] == 1 and x.shape[1] > 1:
                    noise = noise.expand(-1, x.shape[1], *noise.shape[2:])

            perturbed = (x + noise).clamp(0.0, 1.0)

            self.evaluate_batch(x, perturbed, noise)

        return self.aggregate()