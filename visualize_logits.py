#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Logits 可视化脚本

可视化干净图像和加噪图像的模型logits对比：
- 加载已训练的victim模型
- 加载manifest中的噪声
- 比较 logits_clean 和 logits_noisy

使用示例：
python visualize_logits.py \
    --model_path outputs/flare21_seg/victim_unet/checkpoints/best_model.pth \
    --noise_dir outputs/flare21_ue/noise/epoch_0099 \
    --dataset_config configs/dataset/flare21.yaml \
    --dataset flare21 \
    --gpu 0
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

# 添加项目根目录到 path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.registry import get_dataset_builder, get_model
from src.core.ue_artifacts import UEShardsAccessor
from src.core.ue_keys import extract_key
from src.utils.config import get_config


# FLARE21 分割类别颜色
FLARE21_COLORS = np.array([
    [0, 0, 0],       # 0: 背景 - 黑色
    [255, 0, 0],     # 1: 肝脏 - 红色
    [0, 255, 0],     # 2: 肾脏 - 绿色
    [0, 0, 255],     # 3: 脾脏 - 蓝色
    [255, 255, 0],   # 4: 胰腺 - 黄色
], dtype=np.uint8)

FLARE21_LABELS = ["Background", "Liver", "Kidney", "Spleen", "Pancreas"]

# BraTS19 分割类别颜色
BRATS19_COLORS = np.array([
    [0, 0, 0],       # 0: 背景 - 黑色
    [255, 0, 0],     # 1: NCR/NET - 红色
    [0, 255, 0],     # 2: ED - 绿色
    [255, 255, 0],   # 3: ET - 黄色
], dtype=np.uint8)

BRATS19_LABELS = ["Background", "NCR/NET", "ED", "ET"]


def load_config(dataset: str, dataset_config: str = None) -> OmegaConf:
    """加载配置文件"""
    # 根据数据集选择默认配置
    if dataset == "flare21":
        model_config = {
            "name": "unet",
            "in_channels": 1,
            "num_classes": 5,
            "spatial_dims": 3,
            "channels": [32, 64, 128, 256, 512],
            "strides": [2, 2, 2, 2],
            "num_res_units": 2,
            "norm": "INSTANCE",
            "act": "RELU",
            "dropout": 0.0,
        }
    else:  # brats19
        model_config = {
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
        }

    base_config = {
        "model": model_config,
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

    # 加载数据集配置
    if dataset_config and os.path.exists(dataset_config):
        ds_cfg = OmegaConf.load(dataset_config)
        config.dataset = ds_cfg
    else:
        raise ValueError(
            f"未找到数据集配置，请通过 --dataset_config 指定，例如：\n"
            f"  --dataset_config configs/dataset/{dataset}.yaml"
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
    # 处理 builder 名称
    if ds_name == "flare21":
        ds_name = "flare21_seg"
    elif ds_name == "brats19":
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


def label_to_rgb(label_2d: np.ndarray, colors: np.ndarray) -> np.ndarray:
    """将标签转换为RGB图像"""
    h, w = label_2d.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)

    for cls_id, color in enumerate(colors):
        if cls_id >= len(colors):
            break
        mask = label_2d == cls_id
        rgb[mask] = color

    return rgb


def compute_fft_magnitude(data_2d: np.ndarray) -> np.ndarray:
    """计算2D FFT幅度谱（log scale，中心化）"""
    fft = np.fft.fft2(data_2d)
    fft_shift = np.fft.fftshift(fft)
    magnitude = np.abs(fft_shift)
    # log scale，避免log(0)
    magnitude_log = np.log10(magnitude + 1e-10)
    return magnitude_log


def visualize_logits(
    image: torch.Tensor,
    label: torch.Tensor,
    noise: torch.Tensor | None,
    logits_clean: torch.Tensor,
    logits_noisy: torch.Tensor,
    slice_idx: int,
    output_path: str,
    case_id: str = "",
    dataset: str = "flare21",
    eps: float = 8/255,
):
    """
    可视化logits对比（包含FFT频谱）

    Args:
        image: [C, D, H, W] 原始图像
        label: [D, H, W] 分割标签
        noise: [C, D, H, W] 噪声
        logits_clean: [num_classes, D, H, W] 干净图像的logits
        logits_noisy: [num_classes, D, H, W] 加噪图像的logits
        slice_idx: 切片索引
        output_path: 输出路径
        case_id: 样本ID
        dataset: 数据集类型
        eps: 噪声范围
    """
    C, D, H, W = image.shape
    num_classes = logits_clean.shape[0]

    # 选择颜色和标签
    if dataset == "flare21":
        colors = FLARE21_COLORS
        labels = FLARE21_LABELS
    else:
        colors = BRATS19_COLORS
        labels = BRATS19_LABELS

    # 自动选择切片
    if slice_idx < 0:
        label_np = label.numpy()
        label_sum = (label_np > 0).sum(axis=(1, 2))
        nonzero_slices = np.where(label_sum > 0)[0]
        if len(nonzero_slices) > 0:
            slice_idx = nonzero_slices[len(nonzero_slices) // 2]
        else:
            slice_idx = D // 2
    slice_idx = min(max(0, slice_idx), D - 1)

    # 转换为numpy
    if noise is None:
        noise = torch.zeros_like(image)

    image_np = image.numpy()
    noise_np = noise.numpy()
    label_np = label.numpy()
    logits_clean_np = logits_clean.numpy()
    logits_noisy_np = logits_noisy.numpy()

    # 计算预测
    pred_clean = logits_clean_np.argmax(axis=0)  # [D, H, W]
    pred_noisy = logits_noisy_np.argmax(axis=0)  # [D, H, W]

    # 计算logits差异
    logits_diff = logits_noisy_np - logits_clean_np  # [num_classes, D, H, W]

    # 创建图形
    # Row 1: 原图, 噪声, 加噪图, GT
    # Row 2: Pred Clean, Pred Noisy, Pred Diff Overlay
    # Row 3: Logits统计
    # Row 4: FFT总览（所有类别的FFT差异汇总）
    # Row 5+: 每个类别2行（logits + FFT）
    n_rows = 4 + num_classes * 2
    fig = plt.figure(figsize=(20, 4 * n_rows))
    gs = gridspec.GridSpec(n_rows, 5, figure=fig, hspace=0.3, wspace=0.2)

    # ============ Row 1: 原图、噪声、加噪图、GT ============
    # 取第一个通道
    orig_slice = image_np[0, slice_idx]
    noise_slice = noise_np[0, slice_idx]
    noisy_slice = np.clip(image_np[0, slice_idx] + noise_np[0, slice_idx], 0, 1)
    label_slice = label_np[slice_idx]

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(orig_slice, cmap='gray', vmin=0, vmax=1)
    ax1.set_title("Original Image", fontsize=10)
    ax1.axis('off')

    ax2 = fig.add_subplot(gs[0, 1])
    # 噪声可视化（蓝白红）
    normalized = np.clip(noise_slice / eps, -1, 1)
    noise_rgb = np.zeros((*noise_slice.shape, 3), dtype=np.float32)
    neg_mask = normalized < 0
    noise_rgb[neg_mask, 2] = 1.0
    noise_rgb[neg_mask, 0] = 1.0 + normalized[neg_mask]
    noise_rgb[neg_mask, 1] = 1.0 + normalized[neg_mask]
    pos_mask = normalized >= 0
    noise_rgb[pos_mask, 0] = 1.0
    noise_rgb[pos_mask, 1] = 1.0 - normalized[pos_mask]
    noise_rgb[pos_mask, 2] = 1.0 - normalized[pos_mask]
    ax2.imshow(noise_rgb)
    ax2.set_title(f"Noise (eps={eps:.4f})", fontsize=10)
    ax2.axis('off')

    ax3 = fig.add_subplot(gs[0, 2])
    ax3.imshow(noisy_slice, cmap='gray', vmin=0, vmax=1)
    ax3.set_title("Noisy Image", fontsize=10)
    ax3.axis('off')

    ax4 = fig.add_subplot(gs[0, 3])
    label_rgb = label_to_rgb(label_slice, colors)
    ax4.imshow(label_rgb)
    ax4.set_title("Ground Truth", fontsize=10)
    ax4.axis('off')

    # 信息面板
    ax5 = fig.add_subplot(gs[0, 4])
    ax5.axis('off')
    info_text = (
        f"Case: {case_id}\n"
        f"Slice: {slice_idx}/{D}\n"
        f"Shape: {H}x{W}\n"
        f"───────────────\n"
        f"Noise Stats:\n"
        f"  max: {noise_np.max():.6f}\n"
        f"  min: {noise_np.min():.6f}\n"
        f"  L_inf: {np.abs(noise_np).max():.6f}"
    )
    ax5.text(0.1, 0.9, info_text, fontsize=9, family='monospace',
             verticalalignment='top', transform=ax5.transAxes)

    # ============ Row 2: 预测结果对比 ============
    pred_clean_slice = pred_clean[slice_idx]
    pred_noisy_slice = pred_noisy[slice_idx]

    ax6 = fig.add_subplot(gs[1, 0])
    pred_clean_rgb = label_to_rgb(pred_clean_slice, colors)
    ax6.imshow(pred_clean_rgb)
    ax6.set_title("Pred (Clean)", fontsize=10)
    ax6.axis('off')

    ax7 = fig.add_subplot(gs[1, 1])
    pred_noisy_rgb = label_to_rgb(pred_noisy_slice, colors)
    ax7.imshow(pred_noisy_rgb)
    ax7.set_title("Pred (Noisy)", fontsize=10)
    ax7.axis('off')

    ax8 = fig.add_subplot(gs[1, 2])
    # 预测差异：红色表示不同
    diff_mask = pred_clean_slice != pred_noisy_slice
    diff_overlay = orig_slice[..., np.newaxis].repeat(3, axis=-1)
    diff_overlay[diff_mask] = [1.0, 0.0, 0.0]  # 红色标记差异
    ax8.imshow(diff_overlay)
    ax8.set_title(f"Pred Diff (red={diff_mask.sum()} pixels)", fontsize=10)
    ax8.axis('off')

    # 计算Dice差异
    ax9 = fig.add_subplot(gs[1, 3])
    ax9.axis('off')

    # 计算每类的IoU
    iou_text = "Per-class IoU (Clean vs Noisy):\n"
    for cls_id in range(num_classes):
        if cls_id < len(labels):
            clean_mask = pred_clean_slice == cls_id
            noisy_mask = pred_noisy_slice == cls_id
            intersection = (clean_mask & noisy_mask).sum()
            union = (clean_mask | noisy_mask).sum()
            if union > 0:
                iou = intersection / union
            else:
                iou = 1.0
            iou_text += f"  {labels[cls_id]}: {iou:.4f}\n"
    ax9.text(0.1, 0.9, iou_text, fontsize=9, family='monospace',
             verticalalignment='top', transform=ax9.transAxes)

    # 图例
    ax10 = fig.add_subplot(gs[1, 4])
    ax10.axis('off')
    legend_y = 0.9
    ax10.text(0.05, legend_y, "Classes:", fontsize=10, fontweight='bold',
              transform=ax10.transAxes)
    for i, label_name in enumerate(labels[:num_classes]):
        y_pos = legend_y - 0.15 * (i + 1)
        ax10.add_patch(plt.Rectangle((0.05, y_pos - 0.04), 0.15, 0.08,
                                     facecolor=colors[i]/255, edgecolor='black',
                                     transform=ax10.transAxes))
        ax10.text(0.25, y_pos, f"{i}: {label_name}", fontsize=9,
                  verticalalignment='center', transform=ax10.transAxes)

    # ============ Row 3: Logits 统计 ============
    ax11 = fig.add_subplot(gs[2, 0:2])
    # 绘制每个类别的平均logit值
    x = np.arange(num_classes)
    width = 0.35
    clean_means = [logits_clean_np[c, slice_idx].mean() for c in range(num_classes)]
    noisy_means = [logits_noisy_np[c, slice_idx].mean() for c in range(num_classes)]

    bars1 = ax11.bar(x - width/2, clean_means, width, label='Clean', color='blue', alpha=0.7)
    bars2 = ax11.bar(x + width/2, noisy_means, width, label='Noisy', color='red', alpha=0.7)
    ax11.set_xlabel('Class')
    ax11.set_ylabel('Mean Logit')
    ax11.set_title('Mean Logits per Class (this slice)', fontsize=10)
    ax11.set_xticks(x)
    ax11.set_xticklabels([labels[i] if i < len(labels) else f"C{i}" for i in range(num_classes)], rotation=45)
    ax11.legend()
    ax11.grid(True, alpha=0.3)

    ax12 = fig.add_subplot(gs[2, 2:4])
    # Logits差异分布直方图
    logits_diff_slice = logits_diff[:, slice_idx].flatten()
    ax12.hist(logits_diff_slice, bins=50, color='purple', alpha=0.7, edgecolor='black')
    ax12.axvline(x=0, color='red', linestyle='--', linewidth=2)
    ax12.set_xlabel('Logit Difference (Noisy - Clean)')
    ax12.set_ylabel('Count')
    ax12.set_title(f'Logits Difference Distribution\nmean={logits_diff_slice.mean():.4f}, std={logits_diff_slice.std():.4f}', fontsize=10)
    ax12.grid(True, alpha=0.3)

    ax13 = fig.add_subplot(gs[2, 4])
    ax13.axis('off')
    stats_text = (
        "Logits Divergence Stats:\n"
        f"───────────────────────\n"
        f"L1 diff: {np.abs(logits_diff).mean():.4f}\n"
        f"L2 diff: {np.sqrt((logits_diff ** 2).mean()):.4f}\n"
        f"Max diff: {np.abs(logits_diff).max():.4f}\n"
        f"───────────────────────\n"
        f"Pred mismatch:\n"
        f"  This slice: {diff_mask.sum()}\n"
        f"  All slices: {(pred_clean != pred_noisy).sum()}"
    )
    ax13.text(0.1, 0.9, stats_text, fontsize=9, family='monospace',
             verticalalignment='top', transform=ax13.transAxes)

    # ============ Row 4: FFT 总览 ============
    # 计算所有类别的FFT差异汇总
    fft_diff_sum = np.zeros((H, W))
    for c in range(num_classes):
        logit_diff_c = logits_diff[c, slice_idx]
        fft_diff_c = compute_fft_magnitude(logit_diff_c)
        fft_diff_sum += np.abs(fft_diff_c)
    fft_diff_avg = fft_diff_sum / num_classes

    ax_fft1 = fig.add_subplot(gs[3, 0])
    # 原图FFT
    orig_fft = compute_fft_magnitude(image_np[0, slice_idx])
    im_fft1 = ax_fft1.imshow(orig_fft, cmap='magma')
    ax_fft1.set_title("Original Image FFT", fontsize=10)
    ax_fft1.axis('off')
    plt.colorbar(im_fft1, ax=ax_fft1, fraction=0.046, pad=0.04)

    ax_fft2 = fig.add_subplot(gs[3, 1])
    # 噪声FFT
    noise_fft = compute_fft_magnitude(noise_np[0, slice_idx])
    im_fft2 = ax_fft2.imshow(noise_fft, cmap='magma')
    ax_fft2.set_title("Noise FFT", fontsize=10)
    ax_fft2.axis('off')
    plt.colorbar(im_fft2, ax=ax_fft2, fraction=0.046, pad=0.04)

    ax_fft3 = fig.add_subplot(gs[3, 2])
    # Logits差异FFT平均
    im_fft3 = ax_fft3.imshow(fft_diff_avg, cmap='magma')
    ax_fft3.set_title("Avg Logits Diff FFT", fontsize=10)
    ax_fft3.axis('off')
    plt.colorbar(im_fft3, ax=ax_fft3, fraction=0.046, pad=0.04)

    ax_fft4 = fig.add_subplot(gs[3, 3])
    # 径向功率谱对比
    # 计算径向平均
    center_y, center_x = H // 2, W // 2
    y_coords, x_coords = np.ogrid[:H, :W]
    r = np.sqrt((y_coords - center_y)**2 + (x_coords - center_x)**2).astype(int)
    r_max = min(center_y, center_x)

    # Clean logits FFT径向平均（取第一个前景类）
    fg_cls = 1 if num_classes > 1 else 0
    fft_clean_fg = compute_fft_magnitude(logits_clean_np[fg_cls, slice_idx])
    fft_noisy_fg = compute_fft_magnitude(logits_noisy_np[fg_cls, slice_idx])

    radial_clean = np.zeros(r_max)
    radial_noisy = np.zeros(r_max)
    for i in range(r_max):
        mask = r == i
        if mask.sum() > 0:
            radial_clean[i] = fft_clean_fg[mask].mean()
            radial_noisy[i] = fft_noisy_fg[mask].mean()

    ax_fft4.plot(radial_clean, 'b-', label='Clean', linewidth=1.5)
    ax_fft4.plot(radial_noisy, 'r-', label='Noisy', linewidth=1.5)
    ax_fft4.set_xlabel('Frequency (radius)')
    ax_fft4.set_ylabel('Log Magnitude')
    ax_fft4.set_title(f'Radial Power Spectrum ({labels[fg_cls]})', fontsize=10)
    ax_fft4.legend(fontsize=8)
    ax_fft4.grid(True, alpha=0.3)

    ax_fft5 = fig.add_subplot(gs[3, 4])
    ax_fft5.axis('off')
    fft_info = (
        "FFT Analysis:\n"
        "───────────────────────\n"
        "• Low freq = center\n"
        "• High freq = edge\n"
        "───────────────────────\n"
        "Logits FFT Diff:\n"
        f"  mean: {fft_diff_avg.mean():.3f}\n"
        f"  max:  {fft_diff_avg.max():.3f}\n"
        "───────────────────────\n"
        "噪声通过改变logits的\n"
        "频谱分布来影响预测"
    )
    ax_fft5.text(0.1, 0.95, fft_info, fontsize=9, family='monospace',
                verticalalignment='top', transform=ax_fft5.transAxes)

    # ============ Row 5+: 每个类别的Logits和FFT可视化 ============
    for cls_id in range(num_classes):
        row_logit = 4 + cls_id * 2      # Logits行
        row_fft = 4 + cls_id * 2 + 1    # FFT行
        class_name = labels[cls_id] if cls_id < len(labels) else f"Class {cls_id}"

        logit_clean = logits_clean_np[cls_id, slice_idx]
        logit_noisy = logits_noisy_np[cls_id, slice_idx]
        logit_diff = logits_diff[cls_id, slice_idx]

        # ---- Logits行 ----
        # 确定共同的colorbar范围
        vmin = min(logit_clean.min(), logit_noisy.min())
        vmax = max(logit_clean.max(), logit_noisy.max())

        # Clean logit
        ax_c = fig.add_subplot(gs[row_logit, 0])
        im_c = ax_c.imshow(logit_clean, cmap='viridis', vmin=vmin, vmax=vmax)
        ax_c.set_title(f"{class_name} - Logit Clean", fontsize=9)
        ax_c.axis('off')
        plt.colorbar(im_c, ax=ax_c, fraction=0.046, pad=0.04)

        # Noisy logit
        ax_n = fig.add_subplot(gs[row_logit, 1])
        im_n = ax_n.imshow(logit_noisy, cmap='viridis', vmin=vmin, vmax=vmax)
        ax_n.set_title(f"{class_name} - Logit Noisy", fontsize=9)
        ax_n.axis('off')
        plt.colorbar(im_n, ax=ax_n, fraction=0.046, pad=0.04)

        # Diff logit
        ax_d = fig.add_subplot(gs[row_logit, 2])
        diff_abs_max = max(abs(logit_diff.min()), abs(logit_diff.max()))
        if diff_abs_max < 1e-6:
            diff_abs_max = 1.0
        im_d = ax_d.imshow(logit_diff, cmap='RdBu_r', vmin=-diff_abs_max, vmax=diff_abs_max)
        ax_d.set_title(f"{class_name} - Diff (N-C)", fontsize=9)
        ax_d.axis('off')
        plt.colorbar(im_d, ax=ax_d, fraction=0.046, pad=0.04)

        # Softmax probabilities
        ax_p = fig.add_subplot(gs[row_logit, 3])
        prob_clean = F.softmax(torch.from_numpy(logits_clean_np[:, slice_idx]), dim=0).numpy()[cls_id]
        prob_noisy = F.softmax(torch.from_numpy(logits_noisy_np[:, slice_idx]), dim=0).numpy()[cls_id]
        prob_diff = prob_noisy - prob_clean
        prob_abs_max = max(abs(prob_diff.min()), abs(prob_diff.max()))
        if prob_abs_max < 1e-6:
            prob_abs_max = 1.0
        im_p = ax_p.imshow(prob_diff, cmap='RdBu_r', vmin=-prob_abs_max, vmax=prob_abs_max)
        ax_p.set_title(f"{class_name} - Prob Diff", fontsize=9)
        ax_p.axis('off')
        plt.colorbar(im_p, ax=ax_p, fraction=0.046, pad=0.04)

        # Stats for this class
        ax_s = fig.add_subplot(gs[row_logit, 4])
        ax_s.axis('off')
        cls_stats = (
            f"{class_name}:\n"
            f"───────────────\n"
            f"Logit Clean:\n"
            f"  mean: {logit_clean.mean():.3f}\n"
            f"  std:  {logit_clean.std():.3f}\n"
            f"Logit Noisy:\n"
            f"  mean: {logit_noisy.mean():.3f}\n"
            f"  std:  {logit_noisy.std():.3f}\n"
            f"Diff:\n"
            f"  mean: {logit_diff.mean():.4f}\n"
            f"  L1:   {np.abs(logit_diff).mean():.4f}"
        )
        ax_s.text(0.1, 0.95, cls_stats, fontsize=8, family='monospace',
                 verticalalignment='top', transform=ax_s.transAxes)

        # ---- FFT行 ----
        # 计算FFT
        fft_clean = compute_fft_magnitude(logit_clean)
        fft_noisy = compute_fft_magnitude(logit_noisy)
        fft_diff = fft_noisy - fft_clean

        # FFT共同范围
        fft_vmin = min(fft_clean.min(), fft_noisy.min())
        fft_vmax = max(fft_clean.max(), fft_noisy.max())

        # FFT Clean
        ax_fc = fig.add_subplot(gs[row_fft, 0])
        im_fc = ax_fc.imshow(fft_clean, cmap='magma', vmin=fft_vmin, vmax=fft_vmax)
        ax_fc.set_title(f"{class_name} - FFT Clean", fontsize=9)
        ax_fc.axis('off')
        plt.colorbar(im_fc, ax=ax_fc, fraction=0.046, pad=0.04)

        # FFT Noisy
        ax_fn = fig.add_subplot(gs[row_fft, 1])
        im_fn = ax_fn.imshow(fft_noisy, cmap='magma', vmin=fft_vmin, vmax=fft_vmax)
        ax_fn.set_title(f"{class_name} - FFT Noisy", fontsize=9)
        ax_fn.axis('off')
        plt.colorbar(im_fn, ax=ax_fn, fraction=0.046, pad=0.04)

        # FFT Diff
        ax_fd = fig.add_subplot(gs[row_fft, 2])
        fft_diff_abs_max = max(abs(fft_diff.min()), abs(fft_diff.max()))
        if fft_diff_abs_max < 1e-6:
            fft_diff_abs_max = 1.0
        im_fd = ax_fd.imshow(fft_diff, cmap='RdBu_r', vmin=-fft_diff_abs_max, vmax=fft_diff_abs_max)
        ax_fd.set_title(f"{class_name} - FFT Diff", fontsize=9)
        ax_fd.axis('off')
        plt.colorbar(im_fd, ax=ax_fd, fraction=0.046, pad=0.04)

        # 径向功率谱对比
        ax_fr = fig.add_subplot(gs[row_fft, 3])
        radial_clean_cls = np.zeros(r_max)
        radial_noisy_cls = np.zeros(r_max)
        for i in range(r_max):
            mask = r == i
            if mask.sum() > 0:
                radial_clean_cls[i] = fft_clean[mask].mean()
                radial_noisy_cls[i] = fft_noisy[mask].mean()

        ax_fr.plot(radial_clean_cls, 'b-', label='Clean', linewidth=1.5)
        ax_fr.plot(radial_noisy_cls, 'r-', label='Noisy', linewidth=1.5)
        ax_fr.fill_between(range(r_max), radial_clean_cls, radial_noisy_cls,
                           alpha=0.3, color='purple')
        ax_fr.set_xlabel('Frequency')
        ax_fr.set_ylabel('Log Mag')
        ax_fr.set_title(f"{class_name} - Radial Spectrum", fontsize=9)
        ax_fr.legend(fontsize=7)
        ax_fr.grid(True, alpha=0.3)

        # FFT Stats
        ax_fs = fig.add_subplot(gs[row_fft, 4])
        ax_fs.axis('off')
        fft_stats = (
            f"{class_name} FFT:\n"
            f"───────────────\n"
            f"FFT Clean:\n"
            f"  mean: {fft_clean.mean():.3f}\n"
            f"  max:  {fft_clean.max():.3f}\n"
            f"FFT Noisy:\n"
            f"  mean: {fft_noisy.mean():.3f}\n"
            f"  max:  {fft_noisy.max():.3f}\n"
            f"FFT Diff:\n"
            f"  L1:   {np.abs(fft_diff).mean():.4f}\n"
            f"  max:  {np.abs(fft_diff).max():.4f}"
        )
        ax_fs.text(0.1, 0.95, fft_stats, fontsize=8, family='monospace',
                  verticalalignment='top', transform=ax_fs.transAxes)

    # 总标题
    fig.suptitle(
        f"Logits Visualization - {case_id} (Slice {slice_idx}/{D})",
        fontsize=14, fontweight='bold', y=0.995
    )

    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)

    print(f"[Saved] {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Logits 可视化")
    parser.add_argument("--model_path", type=str, required=True,
                        help="模型 checkpoint 路径")
    parser.add_argument("--noise_dir", type=str, required=True,
                        help="噪声目录 (包含 manifest.json)")
    parser.add_argument("--output_dir", type=str, default="outputs/visualize_logits",
                        help="输出目录")
    parser.add_argument("--dataset_config", type=str, required=True,
                        help="数据集配置文件路径")
    parser.add_argument("--dataset", type=str, default="flare21",
                        choices=["flare21", "brats19"],
                        help="数据集类型")
    parser.add_argument("--split", type=str, default="train",
                        help="数据集分割 (train/val/test)")
    parser.add_argument("--sample_idx", type=int, default=0,
                        help="样本索引")
    parser.add_argument("--slice_idx", type=int, default=-1,
                        help="切片索引，-1 表示自动选择有标签的中间层")
    parser.add_argument("--eps", type=float, default=8/255,
                        help="噪声范围 epsilon")
    parser.add_argument("--num_samples", type=int, default=1,
                        help="可视化样本数量")
    parser.add_argument("--gpu", type=int, default=0,
                        help="GPU设备ID")

    args = parser.parse_args()

    # 设置设备
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")

    # 加载配置
    config = load_config(args.dataset, args.dataset_config)

    # 加载模型
    print(f"[Loading Model] {args.model_path}")
    model = load_model(args.model_path, config, device)
    print(f"[Model Loaded] {config.model.name}, in_channels={config.model.in_channels}, num_classes={config.model.num_classes}")

    # 加载数据集
    print(f"[Loading Dataset] {args.dataset}, split={args.split}")
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

        print(f"[Processing] {idx}: {case_id}, shape={tuple(image.shape)}")

        # 提取 key 并加载噪声
        key = extract_key(sample, idx, key_spec)
        noise = load_noise(args.noise_dir, key)

        if noise is None:
            print(f"[Warning] No noise found for {case_id}, using zero noise")
            noise = torch.zeros_like(image)

        # 模型推理
        with torch.no_grad():
            # 干净图像
            image_device = image.unsqueeze(0).to(device)  # [1, C, D, H, W]
            logits_clean = model(image_device)
            if isinstance(logits_clean, (tuple, list)):
                logits_clean = logits_clean[0]
            logits_clean = logits_clean.squeeze(0).cpu()  # [num_classes, D, H, W]

            # 加噪图像
            noisy_image = (image + noise).clamp(0, 1)
            noisy_device = noisy_image.unsqueeze(0).to(device)
            logits_noisy = model(noisy_device)
            if isinstance(logits_noisy, (tuple, list)):
                logits_noisy = logits_noisy[0]
            logits_noisy = logits_noisy.squeeze(0).cpu()  # [num_classes, D, H, W]

        # 输出文件名
        output_path = os.path.join(args.output_dir, f"{idx:04d}_{case_id}_logits_vis.png")

        # 可视化
        visualize_logits(
            image=image,
            label=label,
            noise=noise,
            logits_clean=logits_clean,
            logits_noisy=logits_noisy,
            slice_idx=args.slice_idx,
            output_path=output_path,
            case_id=case_id,
            dataset=args.dataset,
            eps=args.eps,
        )

    print(f"[Done] Visualized {min(args.num_samples, len(dataset) - args.sample_idx)} samples")


if __name__ == "__main__":
    main()
