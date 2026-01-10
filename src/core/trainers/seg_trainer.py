# src/trainers/grape_seg.py
from __future__ import annotations
from typing import Dict, Any

import torch
import torch.nn as nn

from ..trainer_base import TrainerBase
from omegaconf import DictConfig
from ...utils.config import get_config
from monai.losses import DiceCELoss


class SegTrainer(TrainerBase):
    """
    Generic supervised trainer for semantic segmentation.

    适用场景（例如 BraTS19 3D 肿瘤分割）：
      - Batch:
          batch["image"]: FloatTensor [B, C, ...]      (2D: [B,C,H,W]; 3D: [B,C,D,H,W])
          batch["label"]: LongTensor  [B,   ...]       (2D: [B,H,W];    3D: [B,D,H,W])
            取值为 {0,1,...,num_classes-1}
      - Model:
          model(x) -> logits: FloatTensor [B, num_classes, ...]
      - Loss:
          MONAI DiceCELoss (multi-class, softmax + CE)，可配置。

    Config 示例：
      training:
        criterion:
          include_background: False
          squared_pred: False
          jaccard: False
          lambda_dice: 1.0
          lambda_ce: 1.0
          ce_weight: [0.1, 1.0, 5.0]  # 类别权重，用于处理类别不平衡
    """

    def __init__(self, config: DictConfig, device: torch.device, evaluation_strategy):
        super().__init__(config, device)
        self.evaluation_strategy = evaluation_strategy

        # Loss config (mirrors MONAI DiceCELoss signature subset)
        crit_cfg = get_config(config, "training.criterion", DictConfig({}))
        self.include_background = bool(get_config(crit_cfg, "include_background", False))
        self.squared_pred = bool(get_config(crit_cfg, "squared_pred", False))
        self.jaccard = bool(get_config(crit_cfg, "jaccard", False))
        self.lambda_dice = float(get_config(crit_cfg, "lambda_dice", 1.0))
        self.lambda_ce = float(get_config(crit_cfg, "lambda_ce", 1.0))
        # 类别权重，用于处理类别不平衡（如KiTS19的肿瘤类别）
        self.ce_weight = get_config(crit_cfg, "ce_weight", None)

        self._loss = self._build_loss()

    def _build_loss(self) -> nn.Module:
        """
        Multi-class Dice + CE:
          - 输入: logits [B, C, ...]
          - 标签: y_id [B, ...]（class index），内部自动 one-hot
        """
        # 处理类别权重
        weight = None
        if self.ce_weight is not None:
            weight = torch.tensor(self.ce_weight, dtype=torch.float32, device=self.device)

        return DiceCELoss(
            include_background=self.include_background,
            to_onehot_y=True,          # 标签是 class index，[B,...]
            softmax=True,              # 多类别，用 softmax
            squared_pred=self.squared_pred,
            jaccard=self.jaccard,
            lambda_dice=self.lambda_dice,
            lambda_ce=self.lambda_ce,
            reduction="mean",
            weight=weight,             # 类别权重（用于Dice和CE）
        )

    def _init_epoch_metrics(self) -> Dict[str, Any]:
        """Initialize metrics for supervised training"""
        from ...utils.metrics import AverageMeter
        return {
            "loss": AverageMeter()
        }

    def _is_best_model(self, eval_stats: Dict[str, float]) -> bool:
        """Determine if current model is best - delegate to evaluation strategy"""
        if hasattr(self.evaluation_strategy, "is_best_model"):
            return self.evaluation_strategy.is_best_model(eval_stats, self.best_metrics)
        # Default judgment based on validation loss
        if eval_stats:
            metric_name = "loss"
            current_val = eval_stats.get(metric_name, 0.0)
            best_val = self.best_metrics.get(metric_name, float("inf"))
            self.logger.info(
                f"Current {metric_name}: {current_val:.4f}, Best {metric_name}: {best_val:.4f}"
            )
            return current_val < best_val
        return False

    def run_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        单个训练 step：
          - x: [B, C, ...] (2D or 3D)
          - y_id: [B, ...] with {0,...,C-1}
        """
        self.optimizer.zero_grad()

        x = batch["image"].to(self.device)           # 3D: [B,C,D,H,W]
        y_id = batch["label"].to(self.device).long() # 3D: [B,D,H,W]

        logits = self.model(x)                       # [B,num_classes,D,H,W]
        loss = self._loss(logits, y_id.unsqueeze(1)) # -> y: [B,1,D,H,W]
        loss.backward()
        self.optimizer.step()

        return {"loss": float(loss.item())}