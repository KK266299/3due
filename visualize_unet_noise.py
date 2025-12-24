#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
UNet Noise 可视化脚本

可视化中间层切片：
- 五行：四个模态 (T1, T1ce, T2, FLAIR) + 分割结果
- 四列：原图、噪声图、加噪声图、差异图
- 最后一行：原始 label 和模型分割结果

使用示例：
python scripts/visualize_unet_noise.py \
    --model_path outputs/brats19_seg/victim_unet_roi_20251219/checkpoints/best_model.pth \
    --noise_dir outputs/unet_noise_exp/noise/epoch_0100 \
    --output_dir outputs/visualize \
    --sample_idx 0 \
    --slice_idx -1
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import OmegaConf

# 添加项目根目录到 path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.registry import get_dataset_builder, get_model
from src.core.ue_artifacts import UEShardsAccessor
from src.core.ue_keys import extract_key
from src.utils.config import get_config


# BraTS 模态名称
MODALITY_NAMES = ["T1", "T1ce", "T2", "FLAIR"]

# 分割类别颜色 (背景, NCR/NET, ED, ET)
SEG_COLORS = np.array([
    [0, 0, 0],       # 0: 背景 - 黑色
    [255, 0, 0],     # 1: NCR/NET - 红色
    [0, 255, 0],     # 2: ED - 绿色
    [255, 255, 0],   # 3: ET - 黄色
], dtype=np.uint8)


def load_config(config_path: str = None, dataset_config: str = None) -> OmegaConf:
    """加载配置文件"""
    # 基础配置
    base_config = {
        "model": {
            "name": "unet",
            "in_channels": 4,
            "num_classes": 4,
            "spatial_dims": 3,
            "channels": [32, 64, 128, 256, 512],
            "strides": [2, 2, 2, 2],
            "num_res_units": 2,
            "norm": "INSTANCE",
            "act": "RELU",
            "dropout": 0.0,
        },
        "training": {
            "data": {
                "transforms": {
                    "normalize": False,
                    "geom_aug": False,
                    "intensity_aug": False,
                }
            }
        },
        "ue": {
            "key": {
                "type": "samplewise",
                "from": "field",
                "field": "case_id",
            }
        }
    }
    config = OmegaConf.create(base_config)

    # 如果提供了完整配置文件，合并
    if config_path and os.path.exists(config_path):
        user_config = OmegaConf.load(config_path)
        config = OmegaConf.merge(config, user_config)

    # 加载数据集配置
    if dataset_config and os.path.exists(dataset_config):
        ds_cfg = OmegaConf.load(dataset_config)
        config.dataset = ds_cfg
    elif not hasattr(config, 'dataset') or config.dataset is None:
        # 尝试加载默认数据集配置
        default_ds_path = Path(__file__).parent.parent / "configs" / "dataset" / "brats19.yaml"
        if default_ds_path.exists():
            config.dataset = OmegaConf.load(str(default_ds_path))
        else:
            raise ValueError(
                "未找到数据集配置，请通过 --dataset_config 指定，例如：\n"
                "  --dataset_config configs/dataset/brats19.yaml"
            )

    return config


def load_model(model_path: str, config: OmegaConf, device: torch.device) -> torch.nn.Module:
    """加载模型"""
    model_cfg = config.model
    model_cls = get_model(model_cfg.name)
    model = model_cls(model_cfg)

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()
    return model


def load_dataset(config: OmegaConf, split: str = "train"):
    """加载数据集"""
    ds_name = config.dataset.name
    # 处理 builder 名称：brats19 -> brats19_seg
    if ds_name == "brats19":
        ds_name = "brats19_seg"
    builder_cls = get_dataset_builder(ds_name)
    builder = builder_cls(config)
    return builder.get_dataset(split=split)


def load_noise(noise_dir: str, key: str) -> torch.Tensor | None:
    """加载噪声"""
    manifest_path = os.path.join(noise_dir, "manifest.json")
    if not os.path.exists(manifest_path):
        print(f"[Warning] Manifest not found: {manifest_path}")
        return None

    accessor = UEShardsAccessor.from_manifest(manifest_path)
    try:
        noise = accessor.get(key, perturb_type="samplewise")
        return noise
    except KeyError:
        print(f"[Warning] Noise not found for key={key}")
        return None


def colorize_noise(noise_2d: np.ndarray, eps: float = 8/255) -> np.ndarray:
    """
    将噪声可视化为蓝-白-红颜色图

    Args:
        noise_2d: [H, W] 噪声切片
        eps: 噪声范围 [-eps, eps]
    Returns:
        [H, W, 3] uint8 RGB 图像
    """
    # 归一化到 [-1, 1]
    normalized = np.clip(noise_2d / eps, -1, 1)

    # 创建 RGB 图像
    rgb = np.zeros((*noise_2d.shape, 3), dtype=np.float32)

    # 负值 -> 蓝色
    neg_mask = normalized < 0
    rgb[neg_mask, 2] = 1.0  # 蓝色通道
    rgb[neg_mask, 0] = 1.0 + normalized[neg_mask]  # 红色逐渐减少
    rgb[neg_mask, 1] = 1.0 + normalized[neg_mask]  # 绿色逐渐减少

    # 正值 -> 红色
    pos_mask = normalized >= 0
    rgb[pos_mask, 0] = 1.0  # 红色通道
    rgb[pos_mask, 1] = 1.0 - normalized[pos_mask]  # 绿色逐渐减少
    rgb[pos_mask, 2] = 1.0 - normalized[pos_mask]  # 蓝色逐渐减少

    return (rgb * 255).astype(np.uint8)


def label_to_rgb(label_2d: np.ndarray) -> np.ndarray:
    """
    将标签转换为 RGB 图像

    Args:
        label_2d: [H, W] 标签切片，值为 {0, 1, 2, 3}
    Returns:
        [H, W, 3] uint8 RGB 图像
    """
    h, w = label_2d.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)

    for cls_id, color in enumerate(SEG_COLORS):
        mask = label_2d == cls_id
        rgb[mask] = color

    return rgb


def visualize_sample(
    image: torch.Tensor,
    label: torch.Tensor,
    noise: torch.Tensor | None,
    pred: torch.Tensor,
    slice_idx: int,
    output_path: str,
    case_id: str = "",
    eps: float = 8/255,
):
    """
    可视化单个样本

    Args:
        image: [C, D, H, W] 原始图像
        label: [D, H, W] 原始标签
        noise: [C, D, H, W] 噪声，可为 None
        pred: [D, H, W] 模型预测
        slice_idx: 切片索引，-1 表示自动选择中间层
        output_path: 输出路径
        case_id: 样本 ID
        eps: 噪声范围
    """
    C, D, H, W = image.shape

    # 自动选择中间层切片
    if slice_idx < 0:
        slice_idx = D // 2
    slice_idx = min(max(0, slice_idx), D - 1)

    # 如果没有噪声，创建零噪声
    if noise is None:
        noise = torch.zeros_like(image)

    # 转换为 numpy
    image_np = image.numpy()  # [C, D, H, W]
    noise_np = noise.numpy()  # [C, D, H, W]
    label_np = label.numpy()  # [D, H, W]
    pred_np = pred.numpy()    # [D, H, W]

    # 计算加噪声图像
    noisy_np = np.clip(image_np + noise_np, 0, 1)

    # 计算差异图 (放大显示)
    diff_np = np.abs(noise_np) * 10  # 放大 10 倍以便观察
    diff_np = np.clip(diff_np, 0, 1)

    # 创建图形: 5 行 x 4 列
    fig, axes = plt.subplots(5, 4, figsize=(16, 20))

    # 列标题
    col_titles = ["Original", "Noise (x10)", "Noisy", "Diff (|noise|x10)"]

    # 行 1-4: 四个模态
    for row, mod_name in enumerate(MODALITY_NAMES):
        # 提取切片
        orig_slice = image_np[row, slice_idx]       # [H, W]
        noise_slice = noise_np[row, slice_idx]      # [H, W]
        noisy_slice = noisy_np[row, slice_idx]      # [H, W]
        diff_slice = diff_np[row, slice_idx]        # [H, W]

        # 列 0: 原图 (灰度)
        axes[row, 0].imshow(orig_slice, cmap='gray', vmin=0, vmax=1)
        axes[row, 0].set_ylabel(mod_name, fontsize=12, fontweight='bold')
        axes[row, 0].set_xticks([])
        axes[row, 0].set_yticks([])

        # 列 1: 噪声 (蓝白红)
        noise_rgb = colorize_noise(noise_slice, eps)
        axes[row, 1].imshow(noise_rgb)
        axes[row, 1].set_xticks([])
        axes[row, 1].set_yticks([])

        # 列 2: 加噪声图 (灰度)
        axes[row, 2].imshow(noisy_slice, cmap='gray', vmin=0, vmax=1)
        axes[row, 2].set_xticks([])
        axes[row, 2].set_yticks([])

        # 列 3: 差异图 (热力图)
        axes[row, 3].imshow(diff_slice, cmap='hot', vmin=0, vmax=1)
        axes[row, 3].set_xticks([])
        axes[row, 3].set_yticks([])

    # 行 5: 分割结果
    label_slice = label_np[slice_idx]  # [H, W]
    pred_slice = pred_np[slice_idx]    # [H, W]

    # 列 0: 原始 label
    label_rgb = label_to_rgb(label_slice)
    axes[4, 0].imshow(label_rgb)
    axes[4, 0].set_ylabel("Seg", fontsize=12, fontweight='bold')
    axes[4, 0].set_xlabel("Ground Truth", fontsize=10)
    axes[4, 0].set_xticks([])
    axes[4, 0].set_yticks([])

    # 列 1: 空白 (或可放其他信息)
    axes[4, 1].axis('off')
    axes[4, 1].text(0.5, 0.5, f"Case: {case_id}\nSlice: {slice_idx}/{D}",
                    ha='center', va='center', fontsize=12, transform=axes[4, 1].transAxes)

    # 列 2: 模型预测 (原图输入)
    pred_rgb = label_to_rgb(pred_slice)
    axes[4, 2].imshow(pred_rgb)
    axes[4, 2].set_xlabel("Pred (Clean)", fontsize=10)
    axes[4, 2].set_xticks([])
    axes[4, 2].set_yticks([])

    # 列 3: 空白 (用于图例)
    axes[4, 3].axis('off')
    # 添加图例
    legend_labels = ["Background", "NCR/NET", "ED", "ET"]
    for i, (label_name, color) in enumerate(zip(legend_labels, SEG_COLORS)):
        axes[4, 3].add_patch(plt.Rectangle((0.1, 0.8 - i*0.2), 0.2, 0.15,
                                            facecolor=color/255, edgecolor='black'))
        axes[4, 3].text(0.35, 0.875 - i*0.2, label_name, fontsize=10,
                        va='center', transform=axes[4, 3].transAxes)

    # 设置列标题
    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontsize=12, fontweight='bold')

    # 总标题
    fig.suptitle(f"UNet Noise Visualization - {case_id} (Slice {slice_idx})",
                 fontsize=14, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f"[Saved] {output_path}")


def visualize_with_noisy_pred(
    image: torch.Tensor,
    label: torch.Tensor,
    noise: torch.Tensor | None,
    pred_clean: torch.Tensor,
    pred_noisy: torch.Tensor,
    slice_idx: int,
    output_path: str,
    case_id: str = "",
    eps: float = 8/255,
):
    """
    可视化单个样本（包含加噪后的预测）

    Args:
        image: [C, D, H, W] 原始图像
        label: [D, H, W] 原始标签
        noise: [C, D, H, W] 噪声
        pred_clean: [D, H, W] 干净图像的预测
        pred_noisy: [D, H, W] 加噪图像的预测
        slice_idx: 切片索引
        output_path: 输出路径
        case_id: 样本 ID
        eps: 噪声范围
    """
    C, D, H, W = image.shape

    if slice_idx < 0:
        slice_idx = D // 2
    slice_idx = min(max(0, slice_idx), D - 1)

    if noise is None:
        noise = torch.zeros_like(image)

    image_np = image.numpy()
    noise_np = noise.numpy()
    label_np = label.numpy()
    pred_clean_np = pred_clean.numpy()
    pred_noisy_np = pred_noisy.numpy()

    noisy_np = np.clip(image_np + noise_np, 0, 1)
    diff_np = np.clip(np.abs(noise_np) * 10, 0, 1)

    # 创建图形: 6 行 x 4 列
    fig, axes = plt.subplots(6, 4, figsize=(16, 24))

    col_titles = ["Original", "Noise (BWR)", "Noisy", "Diff (|δ|x10)"]

    # 行 1-4: 四个模态
    for row, mod_name in enumerate(MODALITY_NAMES):
        orig_slice = image_np[row, slice_idx]
        noise_slice = noise_np[row, slice_idx]
        noisy_slice = noisy_np[row, slice_idx]
        diff_slice = diff_np[row, slice_idx]

        axes[row, 0].imshow(orig_slice, cmap='gray', vmin=0, vmax=1)
        axes[row, 0].set_ylabel(mod_name, fontsize=12, fontweight='bold')
        axes[row, 0].set_xticks([])
        axes[row, 0].set_yticks([])

        noise_rgb = colorize_noise(noise_slice, eps)
        axes[row, 1].imshow(noise_rgb)
        axes[row, 1].set_xticks([])
        axes[row, 1].set_yticks([])

        axes[row, 2].imshow(noisy_slice, cmap='gray', vmin=0, vmax=1)
        axes[row, 2].set_xticks([])
        axes[row, 2].set_yticks([])

        axes[row, 3].imshow(diff_slice, cmap='hot', vmin=0, vmax=1)
        axes[row, 3].set_xticks([])
        axes[row, 3].set_yticks([])

    # 行 5: Ground Truth 和 Pred Clean
    label_slice = label_np[slice_idx]
    pred_clean_slice = pred_clean_np[slice_idx]

    axes[4, 0].imshow(label_to_rgb(label_slice))
    axes[4, 0].set_ylabel("Label", fontsize=12, fontweight='bold')
    axes[4, 0].set_xlabel("Ground Truth", fontsize=10)
    axes[4, 0].set_xticks([])
    axes[4, 0].set_yticks([])

    axes[4, 1].axis('off')

    axes[4, 2].imshow(label_to_rgb(pred_clean_slice))
    axes[4, 2].set_xlabel("Pred (Clean)", fontsize=10)
    axes[4, 2].set_xticks([])
    axes[4, 2].set_yticks([])

    axes[4, 3].axis('off')

    # 行 6: Pred Noisy 和图例
    pred_noisy_slice = pred_noisy_np[slice_idx]

    axes[5, 0].axis('off')
    axes[5, 0].text(0.5, 0.5, f"Case: {case_id}\nSlice: {slice_idx}/{D}",
                    ha='center', va='center', fontsize=12, transform=axes[5, 0].transAxes)

    axes[5, 1].axis('off')

    axes[5, 2].imshow(label_to_rgb(pred_noisy_slice))
    axes[5, 2].set_ylabel("Pred", fontsize=12, fontweight='bold')
    axes[5, 2].set_xlabel("Pred (Noisy)", fontsize=10)
    axes[5, 2].set_xticks([])
    axes[5, 2].set_yticks([])

    # 图例
    axes[5, 3].axis('off')
    legend_labels = ["Background", "NCR/NET", "ED", "ET"]
    for i, (lbl, color) in enumerate(zip(legend_labels, SEG_COLORS)):
        axes[5, 3].add_patch(plt.Rectangle((0.1, 0.8 - i*0.2), 0.2, 0.15,
                                            facecolor=color/255, edgecolor='black'))
        axes[5, 3].text(0.35, 0.875 - i*0.2, lbl, fontsize=10,
                        va='center', transform=axes[5, 3].transAxes)

    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontsize=12, fontweight='bold')

    fig.suptitle(f"UNet Noise Visualization - {case_id} (Slice {slice_idx})",
                 fontsize=14, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f"[Saved] {output_path}")


def main():
    parser = argparse.ArgumentParser(description="UNet Noise 可视化")
    parser.add_argument("--model_path", type=str, required=True,
                        help="模型 checkpoint 路径")
    parser.add_argument("--noise_dir", type=str, required=True,
                        help="噪声目录 (包含 manifest.json)")
    parser.add_argument("--output_dir", type=str, default="outputs/visualize",
                        help="输出目录")
    parser.add_argument("--config", type=str, default=None,
                        help="配置文件路径")
    parser.add_argument("--dataset_config", type=str, default=None,
                        help="数据集配置文件路径，例如 configs/dataset/brats19.yaml")
    parser.add_argument("--split", type=str, default="train",
                        help="数据集分割 (train/val/test)")
    parser.add_argument("--sample_idx", type=int, default=0,
                        help="样本索引")
    parser.add_argument("--slice_idx", type=int, default=-1,
                        help="切片索引，-1 表示自动选择中间层")
    parser.add_argument("--eps", type=float, default=8/255,
                        help="噪声范围 epsilon")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="计算设备")
    parser.add_argument("--num_samples", type=int, default=1,
                        help="可视化样本数量")

    args = parser.parse_args()

    # 加载配置
    config = load_config(args.config, args.dataset_config)

    # 设备
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")

    # 加载模型
    print(f"[Loading Model] {args.model_path}")
    model = load_model(args.model_path, config, device)

    # 加载数据集
    print(f"[Loading Dataset] split={args.split}")
    dataset = load_dataset(config, split=args.split)
    print(f"[Dataset Size] {len(dataset)}")

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 可视化样本
    key_spec = get_config(config, "ue.key", {})

    for i in range(args.num_samples):
        idx = args.sample_idx + i
        if idx >= len(dataset):
            print(f"[Skip] Sample index {idx} out of range")
            break

        sample = dataset[idx]
        image = sample["image"]  # [C, D, H, W]
        label = sample["label"]  # [D, H, W]
        case_id = sample.get("case_id", f"sample_{idx}")

        # 提取 key 并加载噪声
        key = extract_key(sample, idx, key_spec)
        noise = load_noise(args.noise_dir, key)

        # 模型预测 (干净图像)
        with torch.no_grad():
            image_device = image.unsqueeze(0).to(device)  # [1, C, D, H, W]
            out_clean = model(image_device)
            if isinstance(out_clean, (tuple, list)):
                out_clean = out_clean[0]
            pred_clean = out_clean.argmax(dim=1).squeeze(0).cpu()  # [D, H, W]

        # 模型预测 (加噪图像)
        if noise is not None:
            noisy_image = (image + noise).clamp(0, 1)
            with torch.no_grad():
                noisy_device = noisy_image.unsqueeze(0).to(device)
                out_noisy = model(noisy_device)
                if isinstance(out_noisy, (tuple, list)):
                    out_noisy = out_noisy[0]
                pred_noisy = out_noisy.argmax(dim=1).squeeze(0).cpu()
        else:
            pred_noisy = pred_clean

        # 输出文件名
        output_path = os.path.join(args.output_dir, f"{idx:04d}_{case_id}_vis.png")

        # 可视化
        visualize_with_noisy_pred(
            image=image,
            label=label,
            noise=noise,
            pred_clean=pred_clean,
            pred_noisy=pred_noisy,
            slice_idx=args.slice_idx,
            output_path=output_path,
            case_id=case_id,
            eps=args.eps,
        )

    print(f"[Done] Visualized {min(args.num_samples, len(dataset) - args.sample_idx)} samples")


if __name__ == "__main__":
    main()