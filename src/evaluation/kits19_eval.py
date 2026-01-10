# file: src/evaluation/kits19_eval.py
from __future__ import annotations
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from omegaconf import DictConfig

from monai.metrics import DiceMetric, MeanIoU
from monai.losses import DiceCELoss
from tqdm import tqdm

from ..utils.config import get_config
from ..registry import register_evaluation_strategy


@register_evaluation_strategy("kits19_seg")
class Kits19SegmentationEvaluationStrategy:
    """
    Evaluation for KiTS19 3D kidney tumour segmentation.

    Assumptions:
      - Dataset returns:
          batch["image"] -> FloatTensor [B, C, D, H, W], C=1 for CT
          batch["label"] -> LongTensor  [B, D, H, W] with {0:background, 1:kidney, 2:tumor}

      - Model outputs:
          logits        -> FloatTensor [B, 3, D, H, W] (multi-class)

      - Metrics computed on 2 KiTS19 regions:
          Kidney:  label == 1 or label == 2 (kidney includes tumor region)
          Tumor:   label == 2

    Config keys (optional):

      evaluation.seg:
        class_indices:
          bg:     0
          kidney: 1
          tumor:  2

      evaluation.loss (optional):
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

        # 原始标签索引
        self.idx_bg     = int(get_config(ci, "bg",     0))
        self.idx_kidney = int(get_config(ci, "kidney", 1))
        self.idx_tumor  = int(get_config(ci, "tumor",  2))

        # MONAI metrics on [B, 2, D, H, W] for (Kidney, Tumor)
        self.dice_metric = DiceMetric(
            include_background=True,
            reduction="none",
            get_not_nans=True,
        )
        self.miou_metric = MeanIoU(
            include_background=True,
            reduction="none",
            get_not_nans=True,
        )

        # Optional loss for reporting (should align with training config)
        # 优先使用 training.criterion 配置，如果没有则使用 evaluation.loss
        train_crit_cfg = get_config(self.config, "training.criterion", DictConfig({}))
        loss_cfg = get_config(self.config, "evaluation.loss", DictConfig({}))
        
        include_background = bool(get_config(loss_cfg, "include_background", 
                                             get_config(train_crit_cfg, "include_background", False)))
        squared_pred = bool(get_config(loss_cfg, "squared_pred", 
                                       get_config(train_crit_cfg, "squared_pred", False)))
        jaccard = bool(get_config(loss_cfg, "jaccard", 
                                  get_config(train_crit_cfg, "jaccard", False)))
        lambda_dice = float(get_config(loss_cfg, "lambda_dice", 
                                       get_config(train_crit_cfg, "lambda_dice", 1.0)))
        lambda_ce = float(get_config(loss_cfg, "lambda_ce", 
                                     get_config(train_crit_cfg, "lambda_ce", 1.0)))
        # 类别权重：优先使用训练配置，用于确保评估时的损失计算与训练一致
        ce_weight = get_config(loss_cfg, "ce_weight", 
                              get_config(train_crit_cfg, "ce_weight", None))
        weight = None
        if ce_weight is not None:
            # Note: 这里不能直接使用 device，因为此时 device 还没有传入
            # 权重会在 evaluate_epoch 中设置到正确的 device
            self._ce_weight_list = ce_weight
        else:
            self._ce_weight_list = None

        # 3-class multi-class Dice+CE loss (weight will be set in evaluate_epoch)
        self.loss_fn_config = {
            "include_background": include_background,
            "to_onehot_y": True,
            "softmax": True,
            "squared_pred": squared_pred,
            "jaccard": jaccard,
            "lambda_dice": lambda_dice,
            "lambda_ce": lambda_ce,
            "reduction": "mean",
        }
        self.loss_fn = None  # 将在 evaluate_epoch 中初始化

    # ------------------------------------------------------------------ #
    # helpers: build 2 KiTS19 regions from label id map
    # ------------------------------------------------------------------ #

    def _build_region_masks(self, y_id: torch.Tensor) -> torch.Tensor:
        """
        输入:
          y_id: [B, D, H, W] LongTensor

        输出:
          y_reg: [B, 2, D, H, W] float32
                 channel 0: Kidney (includes tumor)
                 channel 1: Tumor
        """
        kidney = self.idx_kidney
        tumor  = self.idx_tumor

        # Kidney region: kidney + tumor (in KiTS19, tumor is inside kidney)
        y_kidney = (y_id == kidney) | (y_id == tumor)

        # Tumor region
        y_tumor = (y_id == tumor)

        y_reg = torch.stack(
            [y_kidney.float(), y_tumor.float()],
            dim=1,   # -> [B, 2, D, H, W]
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

        # 初始化损失函数（确保权重在正确的 device 上）
        if self.loss_fn is None:
            weight = None
            if self._ce_weight_list is not None:
                weight = torch.tensor(self._ce_weight_list, dtype=torch.float32, device=device)
            self.loss_fn_config["weight"] = weight
            self.loss_fn = DiceCELoss(**self.loss_fn_config)

        total_loss = 0.0
        n_samples = 0

        # reset accumulators
        self.dice_metric.reset()
        self.miou_metric.reset()

        pbar = tqdm(data_loader, desc="Evaluate SEG (KiTS19)", leave=False)
        for batch in pbar:
            x = batch["image"].to(device)                # [B, C, D, H, W]
            y_raw = batch["label"].to(device).long()     # [B,D,H,W] or [B,1,D,H,W]

            # 统一成 [B,D,H,W]
            if y_raw.ndim == 5:
                if y_raw.size(1) != 1:
                    raise ValueError(f"[Kits19SegEval] label ndim=5 but channel={y_raw.size(1)} != 1")
                y_id = y_raw[:, 0]
            elif y_raw.ndim == 4:
                y_id = y_raw
            else:
                raise ValueError(f"[Kits19SegEval] Unsupported label shape: {y_raw.shape}")

            # --- build KiTS19 region GT: [B,2,D,H,W] (Kidney, Tumor) ---
            y_reg = self._build_region_masks(y_id)

            # --- forward ---
            logits = model(x)                            # [B, 3, D, H, W]

            # multi-class prediction
            prob = torch.softmax(logits, dim=1)          # [B, 3, D, H, W]
            y_pred_id = prob.argmax(dim=1)               # [B, D, H, W]

            # --- build KiTS19 region prediction ---
            y_pred_reg = self._build_region_masks(y_pred_id)  # [B,2,D,H,W]

            # --- accumulate metrics (region-based) ---
            self.dice_metric(y_pred=y_pred_reg, y=y_reg)
            self.miou_metric(y_pred=y_pred_reg, y=y_reg)

            # --- val loss（3-class multi-class DiceCE）---
            loss = self.loss_fn(logits, y_id.unsqueeze(1))
            bs = x.size(0)
            total_loss += float(loss.item()) * bs
            n_samples += bs

        # ---- aggregate Dice with not_nans ----
        # MONAI DiceMetric.aggregate() returns:
        #   dice: [N, C] where N is total number of samples, C is number of channels (2 for Kidney, Tumor)
        #   not_nans: [N, C] indicating which samples/channels have valid values
        dice, not_nans = self.dice_metric.aggregate()
        # Ensure correct shape: [N, 2] for (Kidney, Tumor)
        if dice.ndim == 1:
            # If single sample, reshape to [1, 2]
            dice = dice.view(1, -1)
            not_nans = not_nans.view(1, -1)
        elif dice.ndim == 2:
            # Already in correct shape [N, 2]
            pass
        else:
            # Flatten to [N, 2]
            dice = dice.view(-1, 2)
            not_nans = not_nans.view(-1, 2)

        region_dice = []
        region_has_samples = []

        for c in range(2):  # 0:Kidney, 1:Tumor
            val_mask = not_nans[:, c] > 0
            has_samples = bool(val_mask.any().item())
            region_has_samples.append(has_samples)

            if has_samples:
                mean_c = dice[val_mask, c].mean()
                region_dice.append(float(mean_c.item()))
            else:
                region_dice.append(0.0)

        kidney_dc, tumor_dc = region_dice

        # avg_dc: 只在有正样本的 region 上取平均
        if any(region_has_samples):
            valid_vals = [
                d for d, flag in zip(region_dice, region_has_samples) if flag
            ]
            avg_dc = float(sum(valid_vals) / len(valid_vals))
        else:
            avg_dc = 0.0

        # ---- aggregate IoU with not_nans ----
        # Similar shape handling as Dice
        miou_vals, miou_not_nans = self.miou_metric.aggregate()
        if miou_vals.ndim == 1:
            miou_vals = miou_vals.view(1, -1)
            miou_not_nans = miou_not_nans.view(1, -1)
        elif miou_vals.ndim == 2:
            pass
        else:
            miou_vals = miou_vals.view(-1, 2)
            miou_not_nans = miou_not_nans.view(-1, 2)

        region_iou = []
        region_has_iou_samples = []

        for c in range(2):
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
            "loss":      float(total_loss / max(1, n_samples)),
            "kidney_dc": kidney_dc,
            "tumor_dc":  tumor_dc,
            "avg_dc":    avg_dc,
            "miou":      miou,
            "jc":        miou,   # alias
        }

        # reset for next epoch call
        self.dice_metric.reset()
        self.miou_metric.reset()

        return metrics