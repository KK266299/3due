# src/models/attention_unet.py
"""
Attention U-Net Implementation based on MONAI

Reference:
    Oktay, O., et al. "Attention U-Net: Learning Where to Look for the Pancreas"
    MIDL 2018. https://arxiv.org/abs/1804.03999
"""
from __future__ import annotations
import torch
import torch.nn as nn
from typing import Any, Dict, Optional, Sequence
from omegaconf import DictConfig, OmegaConf
from monai.networks.nets import AttentionUnet

from src.utils.logger import get_logger
from src.utils.config import get_config
from src.registry import register_model


@register_model("attention_unet")
class AttentionUNet(AttentionUnet):
    """
    Attention U-Net for Medical Image Segmentation based on MONAI.

    Key Features:
    - Attention gates at skip connections
    - Automatic focus on relevant features
    - Suppression of irrelevant background regions

    Args:
        cfg: Configuration dictionary or DictConfig
        in_channels: Number of input channels (overrides config if provided)
        eps: Not used, kept for interface compatibility
    """

    def __init__(
        self,
        cfg: DictConfig | Dict[str, Any],
        in_channels: Optional[int] = None,
        eps: Optional[float] = None,
    ):
        if not isinstance(cfg, DictConfig):
            cfg = OmegaConf.create(cfg)
        log = get_logger()

        # 读配置 - 与原始 unet.py 保持一致
        c_in_cfg = get_config(cfg, "in_channels", 3)
        c_in = (
            in_channels
            if in_channels is not None
            else (None if c_in_cfg == "auto" else int(c_in_cfg))
        )
        if c_in is None:
            raise ValueError(
                "[AttentionUNet] in_channels is 'auto'; please pass in_channels at construction time."
            )

        out_ch = int(get_config(cfg, "num_classes", 1))

        # 与原始 unet.py 相同的配置读取方式
        channels = tuple(get_config(cfg, "channels", [32, 64, 128, 256, 512]))
        strides = tuple(get_config(cfg, "strides", [2, 2, 2, 2]))
        spatial_dims = int(get_config(cfg, "spatial_dims", 3))
        dropout = float(get_config(cfg, "dropout", 0.0))

        # 注意: MONAI AttentionUnet 不支持 act/norm 参数，使用内置默认值
        # 读取这些参数仅用于日志记录
        act = get_config(cfg, "act", "relu")
        norm = get_config(cfg, "norm", "BATCH")

        log.info(
            f"[Gen] AttentionUNet: spatial_dims={spatial_dims}, in={c_in}, out={out_ch}, "
            f"channels={channels}, strides={strides}, dropout={dropout}, "
            f"(note: act/norm use MONAI defaults)"
        )

        super().__init__(
            spatial_dims=spatial_dims,
            in_channels=c_in,
            out_channels=out_ch,
            channels=channels,
            strides=strides,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x)