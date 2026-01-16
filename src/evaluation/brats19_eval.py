# file: src/evaluation/brats19_seg.py
from __future__ import annotations
from typing import Dict, Optional
import math

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader
from omegaconf import DictConfig

from monai.metrics import DiceMetric, MeanIoU
from monai.losses import DiceCELoss
from tqdm import tqdm
import numpy as np

from ..utils.config import get_config
from ..registry import register_evaluation_strategy
from ..utils.eval_metrics import (
    compute_psnr,
    compute_ssim,
    HAS_PYIQA,
)

if HAS_PYIQA:
    from ..utils.eval_metrics import IQAPyTorchMetrics


@register_evaluation_strategy("brats19_seg")
class Brats19SegmentationEvaluationStrategy:
    """
    Evaluation for BraTS19 3D brain tumour segmentation.

    Assumptions:
      - Dataset returns:
          batch["image"] -> FloatTensor [B, C, D, H, W]
          batch["label"] -> LongTensor  [B, D, H, W] with {0:bg, 1:NCR/NET, 2:edema, 3:enhancing}
        （如果你的 Dataset 现在返回 [B,1,D,H,W]，请在这里先 squeeze 掉通道维）

      - Model outputs:
          logits        -> FloatTensor [B, 4, D, H, W] (multi-class)

      - Metrics computed on 3 standard BraTS regions:
          ET (enhancing tumour):   label == enh
          TC (tumour core):        label ∈ {ncr, enh}
          WT (whole tumour):       label > bg

    Config keys (optional):

      evaluation.seg:
        class_indices:
          bg:    0
          ncr:   1
          edema: 2
          enh:   3

      evaluation.loss (optional):
        # 如果不配置，使用默认 DiceCE (softmax, to_onehot_y=True)
        include_background: False
        squared_pred: False
        jaccard: False
        lambda_dice: 1.0
        lambda_ce: 1.0
    """

    def __init__(self, config: Optional[DictConfig] = None):
        self.config = config or DictConfig({})

        seg_cfg = get_config(self.config, "evaluation.seg", DictConfig({}))
        ci = get_config(seg_cfg, "class_indices", DictConfig({}))

        # 原始标签索引（可在 config 中改）
        self.idx_bg    = int(get_config(ci, "bg",    0))
        self.idx_ncr   = int(get_config(ci, "ncr",   1))
        self.idx_edema = int(get_config(ci, "edema", 2))
        self.idx_enh   = int(get_config(ci, "enh",   3))

        # MONAI metrics on [B, 3, D, H, W] for (ET, TC, WT)
        # 注意：这里的 3 个通道都是“前景 region”，没有真正的 background 通道。
        # include_background=True 的含义只是“不要把第 0 通道当作背景忽略掉”，
        # 否则 DiceMetric 会在内部把第 0 通道跳过，你就会只得到 2 个通道的结果。
        self.dice_metric = DiceMetric(
            include_background=True,   # 保证 3 个 region 通道都被计算
            reduction="none",          # 返回 per-sample, per-channel
            get_not_nans=True,         # 同时返回 not_nans，用来跳过空 region
        )
        self.miou_metric = MeanIoU(
            include_background=True,
            reduction="none",
            get_not_nans=True,
        )

        # Optional loss for reporting (align with training if desired)
        loss_cfg = get_config(self.config, "evaluation.loss", DictConfig({}))
        include_background = bool(get_config(loss_cfg, "include_background", False))
        squared_pred = bool(get_config(loss_cfg, "squared_pred", False))
        jaccard = bool(get_config(loss_cfg, "jaccard", False))
        lambda_dice = float(get_config(loss_cfg, "lambda_dice", 1.0))
        lambda_ce = float(get_config(loss_cfg, "lambda_ce", 1.0))

        # 验证时的 4-class multi-class Dice+CE loss
        # logits: [B,4,D,H,W]
        # y_id:  [B,D,H,W]   -> 这里会在 evaluate_epoch 里 unsqueeze 成 [B,1,D,H,W]
        self.loss_fn = DiceCELoss(
            include_background=include_background,
            to_onehot_y=True,
            softmax=True,
            squared_pred=squared_pred,
            jaccard=jaccard,
            lambda_dice=lambda_dice,
            lambda_ce=lambda_ce,
            reduction="mean",
        )

    # ------------------------------------------------------------------ #
    # helpers: build 3 BraTS regions from label id map
    # ------------------------------------------------------------------ #

    def _build_region_masks(self, y_id: torch.Tensor) -> torch.Tensor:
        """
        输入:
          y_id: [B, D, H, W] LongTensor

        输出:
          y_reg: [B, 3, D, H, W] float32
                 channel 0: ET
                 channel 1: TC
                 channel 2: WT
        """
        bg    = self.idx_bg
        ncr   = self.idx_ncr
        edema = self.idx_edema
        enh   = self.idx_enh

        # enhancing tumour (ET)
        y_et = y_id.eq(enh)

        # tumour core (TC): NCR/NET + Enhancing
        y_tc = y_id.eq(ncr) | y_id.eq(enh)

        # whole tumour (WT): all non-background
        y_wt = y_id.ne(bg)

        y_reg = torch.stack(
            [y_et.float(), y_tc.float(), y_wt.float()],
            dim=1,   # -> [B, 3, D, H, W]
        )
        return y_reg

    # ------------------------------------------------------------------ #
    # main API
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def evaluate_epoch(
        self,
        model: nn.Module,
        data_loader: DataLoader,
        device: torch.device,
    ) -> Dict[str, float]:
        model.eval()
        model.to(device)

        total_loss = 0.0
        n_samples = 0

        # reset accumulators
        self.dice_metric.reset()
        self.miou_metric.reset()

        pbar = tqdm(data_loader, desc="Evaluate SEG (BraTS19)", leave=False)
        for batch in pbar:
            x = batch["image"].to(device)                # [B, C, D, H, W]
            y_raw = batch["label"].to(device).long()     # 可能是 [B,D,H,W] 或 [B,1,D,H,W]

            # 统一成 [B,D,H,W]
            if y_raw.ndim == 5:
                # [B,1,D,H,W] -> [B,D,H,W]
                if y_raw.size(1) != 1:
                    raise ValueError(f"[Brats19SegEval] label ndim=5 but channel={y_raw.size(1)} != 1")
                y_id = y_raw[:, 0]
            elif y_raw.ndim == 4:
                y_id = y_raw
            else:
                raise ValueError(f"[Brats19SegEval] Unsupported label shape: {y_raw.shape}")

            # --- build BraTS region GT: [B,3,D,H,W] (ET,TC,WT) ---
            y_reg = self._build_region_masks(y_id)

            # --- forward ---
            logits = model(x)                            # [B, 4, D, H, W]

            # multi-class prediction
            prob = torch.softmax(logits, dim=1)          # [B, 4, D, H, W]
            y_pred_id = prob.argmax(dim=1)               # [B, D, H, W]

            # --- build BraTS region prediction ---
            y_pred_reg = self._build_region_masks(y_pred_id)  # [B,3,D,H,W]

            # --- accumulate metrics (region-based) ---
            self.dice_metric(y_pred=y_pred_reg, y=y_reg)
            self.miou_metric(y_pred=y_pred_reg, y=y_reg)

            # --- val loss（4-class multi-class DiceCE）---
            # DiceCELoss 要求 target 是 [B,1,D,H,W]（如果不是 one-hot）
            loss = self.loss_fn(logits, y_id.unsqueeze(1))
            bs = x.size(0)
            total_loss += float(loss.item()) * bs
            n_samples += bs

        # ---- aggregate Dice with not_nans ----
        # dice:      [*, 3]     （ET,TC,WT）
        # not_nans:  [*, 3]     同形状，表示每个样本/通道是否有效
        dice, not_nans = self.dice_metric.aggregate()
        dice = dice.view(-1, 3)
        not_nans = not_nans.view(-1, 3)

        region_dice = []
        region_has_samples = []

        for c in range(3):  # 0:ET, 1:TC, 2:WT
            val_mask = not_nans[:, c] > 0   # 这个 region 在哪些样本上是“非空例子”
            has_samples = bool(val_mask.any().item())
            region_has_samples.append(has_samples)

            if has_samples:
                # 只在有正样本的样本上做平均 -> 符合 BraTS 官方评测逻辑
                mean_c = dice[val_mask, c].mean()
                region_dice.append(float(mean_c.item()))
            else:
                # 整个 val/test 里都没有这个 region：
                # 数学上 Dice 是未定义的，这里约定记为 0.0，
                # 但 avg_dc 的时候会只在有样本的 region 上做平均。
                region_dice.append(0.0)

        et_dc, tc_dc, wt_dc = region_dice

        # avg_dc: 只在“有正样本的 region”上取平均，避免 avg 也变成 NaN
        if any(region_has_samples):
            valid_vals = [
                d for d, flag in zip(region_dice, region_has_samples) if flag
            ]
            avg_dc = float(sum(valid_vals) / len(valid_vals))
        else:
            # 极端情况：所有 region 都没正样本（基本不太会发生）
            avg_dc = 0.0

        # ---- aggregate IoU with not_nans（同样逻辑） ----
        miou_vals, miou_not_nans = self.miou_metric.aggregate()
        miou_vals = miou_vals.view(-1, 3)
        miou_not_nans = miou_not_nans.view(-1, 3)

        region_iou = []
        region_has_iou_samples = []

        for c in range(3):
            val_mask = miou_not_nans[:, c] > 0
            has_samples = bool(val_mask.any().item())
            region_has_iou_samples.append(has_samples)

            if has_samples:
                mean_c = miou_vals[val_mask, c].mean()
                region_iou.append(float(mean_c.item()))
            else:
                region_iou.append(0.0)

        if any(region_has_iou_samples):
            valid_iou_vals = [
                v for v, flag in zip(region_iou, region_has_iou_samples) if flag
            ]
            miou = float(sum(valid_iou_vals) / len(valid_iou_vals))
        else:
            miou = 0.0

        metrics = {
            "loss":   float(total_loss / max(1, n_samples)),
            "et_dc":  et_dc,
            "tc_dc":  tc_dc,
            "wt_dc":  wt_dc,
            "avg_dc": avg_dc,
            "miou":   miou,
            "jc":     miou,   # alias
        }

        # reset for next epoch call
        self.dice_metric.reset()
        self.miou_metric.reset()

        return metrics


@register_evaluation_strategy("brats19_perturbation")
class Brats19PerturbationEvaluationStrategy:
    """
    Evaluation for perturbation quality on BraTS19 dataset.

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

        # Noise statistics
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

        pbar = tqdm(data_loader, desc="Evaluate Perturbation (BraTS19)", leave=False)
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
                elif x.shape[1] == 1:
                    # Average noise across channels for single-channel image
                    noise = noise.mean(dim=1, keepdim=True)

            perturbed = (x + noise).clamp(0.0, 1.0)

            self.evaluate_batch(x, perturbed, noise)

            # Update progress bar
            if self._n_samples > 0:
                psnr = self._metrics_sum.get('psnr', 0) / self._n_samples
                ssim = self._metrics_sum.get('ssim', 0) / self._n_samples
                pbar.set_postfix({'psnr': f'{psnr:.2f}', 'ssim': f'{ssim:.4f}'})

        return self.aggregate()


@register_evaluation_strategy("brats19_combined")
class Brats19CombinedEvaluationStrategy:
    """
    Combined evaluation for BraTS19: Segmentation + Perturbation quality.

    Use this when you want to evaluate both segmentation performance
    and perturbation invisibility in a single pass.
    """

    def __init__(self, config: Optional[DictConfig] = None):
        self.config = config or DictConfig({})
        self.seg_eval = Brats19SegmentationEvaluationStrategy(config)
        self.pert_eval = Brats19PerturbationEvaluationStrategy(config)

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