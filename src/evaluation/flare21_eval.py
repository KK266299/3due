# file: src/evaluation/flare21_eval.py
"""
Evaluation for FLARE21 abdominal CT organ segmentation.

Combines:
  1. Segmentation quality metrics (Dice, IoU per organ)
  2. Perturbation quality metrics (PSNR, SSIM via IQA-PyTorch)

FLARE21 has 4 organs:
  - Liver (1)
  - Kidney (2)
  - Spleen (3)
  - Pancreas (4)

Usage:
    from src.evaluation.flare21_eval import Flare21SegmentationEvaluationStrategy

    evaluator = Flare21SegmentationEvaluationStrategy(config)
    metrics = evaluator.evaluate_epoch(model, data_loader, device)
"""
from __future__ import annotations
from typing import Dict, Optional, List
import math

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader
from omegaconf import DictConfig
from tqdm import tqdm
import numpy as np

from monai.metrics import DiceMetric, MeanIoU
from monai.losses import DiceCELoss

from ..utils.config import get_config
from ..registry import register_evaluation_strategy
from ..utils.eval_metrics import (
    compute_psnr,
    compute_ssim,
    HAS_PYIQA,
)

if HAS_PYIQA:
    from ..utils.eval_metrics import IQAPyTorchMetrics


@register_evaluation_strategy("flare21_seg")
class Flare21SegmentationEvaluationStrategy:
    """
    Evaluation for FLARE21 abdominal organ segmentation.

    Computes:
      - Per-organ Dice scores (Liver, Kidney, Spleen, Pancreas)
      - Mean Dice across organs
      - Mean IoU
      - Validation loss

    Assumptions:
      - Dataset returns:
          batch["image"] -> FloatTensor [B, 1, D, H, W] (CT, single channel)
          batch["label"] -> LongTensor  [B, D, H, W] with {0:bg, 1:liver, 2:kidney, 3:spleen, 4:pancreas}

      - Model outputs:
          logits -> FloatTensor [B, 5, D, H, W] (5-class: bg + 4 organs)

    Config keys (optional):
      evaluation.seg:
        num_classes: 5
        organ_names: [liver, kidney, spleen, pancreas]

      evaluation.loss:
        include_background: false
        squared_pred: false
        jaccard: false
        lambda_dice: 1.0
        lambda_ce: 1.0
    """

    def __init__(self, config: Optional[DictConfig] = None):
        self.config = config or DictConfig({})

        seg_cfg = get_config(self.config, "evaluation.seg", DictConfig({}))
        self.num_classes = int(get_config(seg_cfg, "num_classes", 5))
        self.organ_names = list(get_config(
            seg_cfg, "organ_names",
            ["liver", "kidney", "spleen", "pancreas"]
        ))

        # Dice metric per class (excluding background)
        self.dice_metric = DiceMetric(
            include_background=False,  # Exclude background (class 0)
            reduction="none",
            get_not_nans=True,
        )
        self.miou_metric = MeanIoU(
            include_background=False,
            reduction="none",
            get_not_nans=True,
        )

        # Loss function
        loss_cfg = get_config(self.config, "evaluation.loss", DictConfig({}))
        self.loss_fn = DiceCELoss(
            include_background=bool(get_config(loss_cfg, "include_background", False)),
            to_onehot_y=True,
            softmax=True,
            squared_pred=bool(get_config(loss_cfg, "squared_pred", False)),
            jaccard=bool(get_config(loss_cfg, "jaccard", False)),
            lambda_dice=float(get_config(loss_cfg, "lambda_dice", 1.0)),
            lambda_ce=float(get_config(loss_cfg, "lambda_ce", 1.0)),
            reduction="mean",
        )

    def _one_hot_encode(self, y: Tensor, num_classes: int) -> Tensor:
        """
        Convert class indices to one-hot encoding.

        Args:
            y: [B, D, H, W] class indices
            num_classes: Number of classes

        Returns:
            [B, num_classes, D, H, W] one-hot tensor
        """
        B, D, H, W = y.shape
        one_hot = torch.zeros(B, num_classes, D, H, W, device=y.device, dtype=torch.float32)
        one_hot.scatter_(1, y.unsqueeze(1), 1)
        return one_hot

    @torch.no_grad()
    def evaluate_epoch(
        self,
        model: nn.Module,
        data_loader: DataLoader,
        device: torch.device,
    ) -> Dict[str, float]:
        """
        Evaluate segmentation quality for one epoch.

        Args:
            model: Segmentation model
            data_loader: Validation data loader
            device: Device to use

        Returns:
            Dictionary of metrics
        """
        model.eval()
        model.to(device)

        total_loss = 0.0
        n_samples = 0

        self.dice_metric.reset()
        self.miou_metric.reset()

        pbar = tqdm(data_loader, desc="Evaluate SEG (FLARE21)", leave=False)
        for batch in pbar:
            x = batch["image"].to(device).float()
            y_raw = batch["label"].to(device).long()

            # Ensure [B, D, H, W]
            if y_raw.ndim == 5:
                y_id = y_raw[:, 0]
            else:
                y_id = y_raw

            # Forward
            logits = model(x)  # [B, 5, D, H, W]

            # Predictions
            prob = torch.softmax(logits, dim=1)
            y_pred_id = prob.argmax(dim=1)  # [B, D, H, W]

            # One-hot for metrics (MONAI expects [B, C, D, H, W])
            y_one_hot = self._one_hot_encode(y_id, self.num_classes)
            y_pred_one_hot = self._one_hot_encode(y_pred_id, self.num_classes)

            # Accumulate metrics (excluding background channel 0)
            self.dice_metric(y_pred=y_pred_one_hot[:, 1:], y=y_one_hot[:, 1:])
            self.miou_metric(y_pred=y_pred_one_hot[:, 1:], y=y_one_hot[:, 1:])

            # Loss
            loss = self.loss_fn(logits, y_id.unsqueeze(1))
            bs = x.size(0)
            total_loss += float(loss.item()) * bs
            n_samples += bs

        # Aggregate Dice
        dice, not_nans = self.dice_metric.aggregate()
        dice = dice.view(-1, len(self.organ_names))
        not_nans = not_nans.view(-1, len(self.organ_names))

        organ_dice = {}
        valid_dices = []

        for c, organ_name in enumerate(self.organ_names):
            val_mask = not_nans[:, c] > 0
            if val_mask.any():
                mean_c = float(dice[val_mask, c].mean().item())
                organ_dice[f"{organ_name}_dc"] = mean_c
                valid_dices.append(mean_c)
            else:
                organ_dice[f"{organ_name}_dc"] = 0.0

        avg_dc = float(np.mean(valid_dices)) if valid_dices else 0.0

        # Aggregate IoU
        miou_vals, miou_not_nans = self.miou_metric.aggregate()
        miou_vals = miou_vals.view(-1, len(self.organ_names))
        miou_not_nans = miou_not_nans.view(-1, len(self.organ_names))

        valid_ious = []
        for c, organ_name in enumerate(self.organ_names):
            val_mask = miou_not_nans[:, c] > 0
            if val_mask.any():
                mean_c = float(miou_vals[val_mask, c].mean().item())
                organ_dice[f"{organ_name}_iou"] = mean_c
                valid_ious.append(mean_c)

        miou = float(np.mean(valid_ious)) if valid_ious else 0.0

        metrics = {
            "loss": float(total_loss / max(1, n_samples)),
            **organ_dice,
            "avg_dc": avg_dc,
            "miou": miou,
        }

        # Reset for next epoch
        self.dice_metric.reset()
        self.miou_metric.reset()

        return metrics


@register_evaluation_strategy("flare21_perturbation")
class Flare21PerturbationEvaluationStrategy:
    """
    Evaluation for perturbation quality on FLARE21 dataset.

    Computes:
      - PSNR (Peak Signal-to-Noise Ratio) ↑
      - SSIM (Structural Similarity Index) ↑
      - IQA-PyTorch metrics (if available)

    Config:
        evaluation.perturbation:
            data_range: 1.0
            use_pyiqa: true
            pyiqa_metrics: [psnr, ssim]
            sample_slices: 16  # Sample N slices for efficiency (null = all)
    """

    def __init__(self, config: Optional[DictConfig] = None):
        self.config = config or DictConfig({})

        pert_cfg = get_config(self.config, "evaluation.perturbation", DictConfig({}))
        self.data_range = float(get_config(pert_cfg, "data_range", 1.0))
        self.use_pyiqa = bool(get_config(pert_cfg, "use_pyiqa", True)) and HAS_PYIQA
        self.pyiqa_metrics = list(get_config(pert_cfg, "pyiqa_metrics", ['psnr', 'ssim']))
        self.sample_slices = get_config(pert_cfg, "sample_slices", 16)

        # Initialize IQA-PyTorch
        self._iqa_evaluator = None
        if self.use_pyiqa:
            try:
                self._iqa_evaluator = IQAPyTorchMetrics(
                    metrics=self.pyiqa_metrics,
                    device='cuda' if torch.cuda.is_available() else 'cpu',
                )
            except Exception as e:
                print(f"Warning: Failed to initialize IQA-PyTorch: {e}")
                self.use_pyiqa = False

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
        Evaluate perturbation quality for a batch.

        Args:
            original: [B, C, D, H, W] original images
            perturbed: [B, C, D, H, W] perturbed images
            noise: Optional noise tensor

        Returns:
            Batch metrics
        """
        results = {}

        # Built-in PSNR (volumetric)
        psnr = compute_psnr(original, perturbed, data_range=self.data_range)
        results['psnr'] = float(psnr.mean().item())

        # Built-in SSIM (volumetric)
        ssim = compute_ssim(original, perturbed, data_range=self.data_range)
        results['ssim'] = float(ssim.item())

        # Noise L-infinity norm
        if noise is not None:
            results['noise_linf'] = float(noise.abs().max().item())
            results['noise_l2'] = float(noise.pow(2).mean().sqrt().item())

        # IQA-PyTorch metrics (slice-wise)
        if self.use_pyiqa and self._iqa_evaluator is not None:
            iqa_results = self._iqa_evaluator.compute_3d_slicewise(
                original, perturbed,
                sample_slices=self.sample_slices,
            )
            for k, v in iqa_results.items():
                results[f'{k}_iqa'] = v

        # Accumulate
        bs = original.shape[0]
        for k, v in results.items():
            if not math.isnan(v):
                self._metrics_sum[k] = self._metrics_sum.get(k, 0.0) + v * bs
        self._n_samples += bs

        return results

    def aggregate(self) -> Dict[str, float]:
        """Aggregate accumulated metrics."""
        if self._n_samples == 0:
            return {}

        results = {}
        for k, v in self._metrics_sum.items():
            results[k] = v / self._n_samples

        return results

    @torch.no_grad()
    def evaluate_with_loader(
        self,
        data_loader: DataLoader,
        noise_accessor,
        device: torch.device,
    ) -> Dict[str, float]:
        """
        Evaluate perturbation quality over entire dataset.

        Args:
            data_loader: DataLoader for clean images
            noise_accessor: UEShardsAccessor to get noise
            device: Device to use

        Returns:
            Aggregated metrics
        """
        self.reset()

        pbar = tqdm(data_loader, desc="Evaluate Perturbation (FLARE21)", leave=False)
        for batch_idx, batch in enumerate(pbar):
            x = batch["image"].to(device).float()
            keys = batch.get("key", range(batch_idx * x.shape[0], (batch_idx + 1) * x.shape[0]))

            # Get noise
            batch_noise = []
            for key in keys:
                noise = noise_accessor.get(key)
                batch_noise.append(noise)

            noise = torch.stack(batch_noise, dim=0).to(device).float()

            # Handle channel mismatch
            if noise.shape[1] != x.shape[1]:
                if noise.shape[1] == 1:
                    noise = noise.expand(-1, x.shape[1], *noise.shape[2:])

            perturbed = (x + noise).clamp(0.0, 1.0)

            self.evaluate_batch(x, perturbed, noise)

            # Update progress bar
            if self._n_samples > 0:
                psnr = self._metrics_sum.get('psnr', 0) / self._n_samples
                ssim = self._metrics_sum.get('ssim', 0) / self._n_samples
                pbar.set_postfix({'psnr': f'{psnr:.2f}', 'ssim': f'{ssim:.4f}'})

        return self.aggregate()


@register_evaluation_strategy("flare21_combined")
class Flare21CombinedEvaluationStrategy:
    """
    Combined evaluation for FLARE21: Segmentation + Perturbation quality.

    Use this when you want to evaluate both segmentation performance
    and perturbation invisibility in a single pass.
    """

    def __init__(self, config: Optional[DictConfig] = None):
        self.config = config or DictConfig({})
        self.seg_eval = Flare21SegmentationEvaluationStrategy(config)
        self.pert_eval = Flare21PerturbationEvaluationStrategy(config)

    @torch.no_grad()
    def evaluate_epoch(
        self,
        model: nn.Module,
        data_loader: DataLoader,
        device: torch.device,
        noise_accessor=None,
    ) -> Dict[str, float]:
        """
        Evaluate both segmentation and perturbation quality.

        Args:
            model: Segmentation model
            data_loader: Validation data loader
            device: Device to use
            noise_accessor: Optional noise accessor for perturbation metrics

        Returns:
            Combined metrics dictionary
        """
        # Segmentation metrics
        seg_metrics = self.seg_eval.evaluate_epoch(model, data_loader, device)

        # Perturbation metrics (if noise accessor provided)
        pert_metrics = {}
        if noise_accessor is not None:
            pert_metrics = self.pert_eval.evaluate_with_loader(
                data_loader, noise_accessor, device
            )

        return {**seg_metrics, **pert_metrics}