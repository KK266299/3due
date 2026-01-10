# file: src/evaluation/brats19_seg.py
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
