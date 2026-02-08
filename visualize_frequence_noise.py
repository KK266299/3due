#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Frequence Noise 可视化脚本

针对CT图像（如FLARE21）的噪声和频谱约束可视化：
- 使用窗宽窗位 (Window Width/Level) 进行显示
- 可视化频谱约束掩码
- 可视化噪声在频域的分布

可视化内容:
- Row 1: 原图、噪声图、加噪图、噪声差异
- Row 2: 频域约束掩码（Z轴高通、XY低通）
- Row 3: 噪声频谱分布
- Row 4: 分割标签

使用示例：
python visualize_frequence_noise.py \
    --noise_dir outputs/flare21_ue/freq_roi_soft/20260116_112656/noise/epoch_0099 \
    --output_dir outputs/visualize_freq \
    --dataset_config configs/dataset/flare21.yaml \
    --window_width 400 \
    --window_level 40 \
    --sample_idx 0 \
    --slice_idx -1

ROI-aware 可视化（无图例、黑色边框，生成总图 + 小图）：
python visualize_frequence_noise.py \
    --noise_dir outputs/flare21_ue/freq_roi_soft/20260116_112656/noise/epoch_0099 \
    --output_dir outputs/visualize_freq \
    --dataset_config configs/dataset/flare21.yaml \
    --roi_vis \
    --sample_idx 0 \
    --slice_idx -1
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
from omegaconf import OmegaConf

# 添加项目根目录到 path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.registry import get_dataset_builder
from src.core.ue_artifacts import UEShardsAccessor
from src.core.ue_keys import extract_key
from src.utils.config import get_config


# FLARE21 分割类别颜色
# 0: 背景, 1: 肝脏, 2: 肾脏, 3: 脾脏, 4: 胰腺
SEG_COLORS = np.array([
    [0, 0, 0],       # 0: 背景 - 黑色
    [255, 0, 0],     # 1: 肝脏 - 红色
    [0, 255, 0],     # 2: 肾脏 - 绿色
    [0, 0, 255],     # 3: 脾脏 - 蓝色
    [255, 255, 0],   # 4: 胰腺 - 黄色
], dtype=np.uint8)

SEG_LABELS = ["Background", "Liver", "Kidney", "Spleen", "Pancreas"]


def load_config(dataset_config: str = None) -> OmegaConf:
    """加载配置文件"""
    base_config = {
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
            "未找到数据集配置，请通过 --dataset_config 指定，例如：\n"
            "  --dataset_config configs/dataset/flare21.yaml"
        )

    return config


def load_dataset(config: OmegaConf, split: str = "train"):
    """加载数据集"""
    ds_name = config.dataset.name
    # 处理 builder 名称
    if ds_name == "flare21":
        ds_name = "flare21_seg"
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


def apply_window_nnunet(
    image: np.ndarray,
    window_width: float,
    window_level: float,
    global_mean: float = 121.07,
    global_std: float = 49.66,
    z_clip: float = 5.0,
) -> np.ndarray:
    """
    对nnU-Net归一化的图像应用CT窗宽窗位

    nnU-Net预处理流程:
    1. Clip 到 percentile 范围
    2. Z-score: (x - mean) / std
    3. Clip z-score 到 [-z_clip, z_clip]
    4. 映射到 [0, 1]: (z + z_clip) / (2 * z_clip)

    逆变换:
    1. [0, 1] -> z-score: z = image * 2 * z_clip - z_clip
    2. z-score -> 近似HU: hu = z * std + mean

    Args:
        image: 归一化到[0,1]的图像
        window_width: 窗宽 (HU)
        window_level: 窗位 (HU)
        global_mean: 全局均值
        global_std: 全局标准差
        z_clip: z-score clip范围

    Returns:
        windowed: 归一化到[0,1]的窗口化图像
    """
    # 逆变换: [0, 1] -> z-score -> 近似HU
    z_score = image * (2.0 * z_clip) - z_clip  # [0,1] -> [-5,5]
    hu_approx = z_score * global_std + global_mean  # z-score -> HU

    # 应用窗宽窗位
    lower = window_level - window_width / 2
    upper = window_level + window_width / 2

    windowed = np.clip(hu_approx, lower, upper)
    windowed = (windowed - lower) / (upper - lower)

    return windowed


def apply_window_direct(image: np.ndarray, window_width: float, window_level: float) -> np.ndarray:
    """
    对ct_window归一化的图像应用CT窗宽窗位（直接HU映射）

    ct_window预处理: 直接将HU clip到[WL-WW/2, WL+WW/2]然后映射到[0,1]

    Args:
        image: 归一化到[0,1]的图像（已经是窗口化的）
        window_width: 原始窗宽 (HU)
        window_level: 原始窗位 (HU)

    Returns:
        image: 直接返回（已经是窗口化的）
    """
    # 对于ct_window模式，数据已经是窗口化的，直接返回
    return image


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


def label_to_rgb(label_2d: np.ndarray, num_classes: int = 5) -> np.ndarray:
    """
    将标签转换为 RGB 图像

    Args:
        label_2d: [H, W] 标签切片
        num_classes: 类别数量
    Returns:
        [H, W, 3] uint8 RGB 图像
    """
    h, w = label_2d.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)

    for cls_id in range(min(num_classes, len(SEG_COLORS))):
        mask = label_2d == cls_id
        rgb[mask] = SEG_COLORS[cls_id]

    return rgb


def build_frequency_mask(
    D: int, H: int, W: int,
    z_cutoff_low: float = 0.1,
    xy_cutoff_high: float = 0.3,
    z_sigma: float = 0.05,
    xy_sigma: float = 0.1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    构建频域约束掩码（用于可视化）

    Frequence方法使用:
    - Z轴: HIGH-PASS (去除低频，保留高频 -> 切片间多样性)
    - XY平面: LOW-PASS (去除高频，保留低频 -> 平滑)

    Returns:
        M_z: Z轴高通掩码 [D]
        M_xy: XY平面低通掩码 [H, W]
        M_combined: 组合掩码 [D, H, W]
    """
    # Z轴频率
    fz = np.abs(np.fft.fftfreq(D))  # [D]

    # XY平面频率
    fy = np.abs(np.fft.fftfreq(H))  # [H]
    fx = np.abs(np.fft.fftfreq(W))  # [W]
    yy, xx = np.meshgrid(fy, fx, indexing="ij")
    fxy = np.sqrt(yy**2 + xx**2)  # [H, W]

    # Z轴 HIGH-PASS: 保留 |f_z| > z_cutoff_low 的分量
    M_z = np.where(
        fz >= z_cutoff_low,
        np.ones_like(fz),
        np.exp(-((z_cutoff_low - fz) ** 2) / (2 * z_sigma ** 2))
    )

    # XY平面 LOW-PASS: 保留 |f_xy| < xy_cutoff_high 的分量
    M_xy = np.where(
        fxy <= xy_cutoff_high,
        np.ones_like(fxy),
        np.exp(-((fxy - xy_cutoff_high) ** 2) / (2 * xy_sigma ** 2))
    )

    # 组合掩码
    M_combined = M_z[:, np.newaxis, np.newaxis] * M_xy[np.newaxis, :, :]

    return M_z, M_xy, M_combined


def compute_noise_spectrum(noise: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    计算噪声的频谱

    Args:
        noise: [C, D, H, W] 噪声

    Returns:
        spectrum_z: Z轴方向的平均功率谱 [D]
        spectrum_xy: XY平面的平均功率谱 [H, W]
    """
    # 取第一个通道
    noise_3d = noise[0]  # [D, H, W]

    # 3D FFT
    noise_fft = np.fft.fftn(noise_3d)
    noise_fft = np.fft.fftshift(noise_fft)
    power = np.abs(noise_fft) ** 2

    # Z轴方向平均
    spectrum_xy = np.mean(power, axis=0)  # [H, W]

    # XY平面平均
    spectrum_z = np.mean(power, axis=(1, 2))  # [D]

    # 归一化到log scale
    spectrum_xy = np.log10(spectrum_xy + 1e-10)
    spectrum_z = np.log10(spectrum_z + 1e-10)

    return spectrum_z, spectrum_xy


def _add_black_border(ax, linewidth=2):
    """为 axes 添加黑色边框，移除刻度和标签"""
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('black')
        spine.set_linewidth(linewidth)


def _auto_select_slice(label_np: np.ndarray, D: int) -> int:
    """自动选择有标签的中间切片"""
    label_sum = (label_np > 0).sum(axis=(1, 2))  # [D]
    nonzero_slices = np.where(label_sum > 0)[0]
    if len(nonzero_slices) > 0:
        return int(nonzero_slices[len(nonzero_slices) // 2])
    return D // 2


def visualize_roi_noise_total(
    image: torch.Tensor,
    label: torch.Tensor,
    noise: torch.Tensor | None,
    slice_idx: int,
    output_path: str,
    eps: float = 8/255,
    window_width: float = 400,
    window_level: float = 40,
    num_classes: int = 5,
    norm_mode: str = "nnunet",
    global_mean: float = 121.07,
    global_std: float = 49.66,
    z_clip: float = 5.0,
):
    """
    ROI-aware 噪声综合可视化（总图）

    布局: 2 行 × 4 列
      Row 1: 原图, 噪声(蓝白红), 加噪图, |Noise|×20
      Row 2: ROI 掩码, 标签叠加, ROI 内噪声, 噪声能量

    无图例，无标题，所有面板带黑色边框。
    """
    C, D, H, W = image.shape

    # 自动选择切片
    label_np = label.numpy()
    if slice_idx < 0:
        slice_idx = _auto_select_slice(label_np, D)
    slice_idx = min(max(0, slice_idx), D - 1)

    if noise is None:
        noise = torch.zeros_like(image)

    image_np = image.numpy()
    noise_np = noise.numpy()
    noisy_np = np.clip(image_np + noise_np, 0, 1)

    # 取切片
    orig_slice = image_np[0, slice_idx]
    noise_slice = noise_np[0, slice_idx]
    noisy_slice = noisy_np[0, slice_idx]
    label_slice = label_np[slice_idx]

    # 应用窗宽窗位
    if norm_mode == "nnunet":
        orig_windowed = apply_window_nnunet(
            orig_slice, window_width, window_level,
            global_mean=global_mean, global_std=global_std, z_clip=z_clip,
        )
        noisy_windowed = apply_window_nnunet(
            noisy_slice, window_width, window_level,
            global_mean=global_mean, global_std=global_std, z_clip=z_clip,
        )
    else:
        orig_windowed = apply_window_direct(orig_slice, window_width, window_level)
        noisy_windowed = apply_window_direct(noisy_slice, window_width, window_level)

    # ROI 掩码
    roi_mask = (label_slice > 0).astype(np.float32)

    # ---------- 创建 2×4 图形 ----------
    fig, axes = plt.subplots(2, 4, figsize=(16, 8), facecolor='white')
    fig.subplots_adjust(wspace=0.04, hspace=0.04)

    # Row 1, Col 0: 原图
    axes[0, 0].imshow(orig_windowed, cmap='gray', vmin=0, vmax=1)
    _add_black_border(axes[0, 0])

    # Row 1, Col 1: 噪声（蓝白红）
    noise_rgb = colorize_noise(noise_slice, eps)
    axes[0, 1].imshow(noise_rgb)
    _add_black_border(axes[0, 1])

    # Row 1, Col 2: 加噪图
    axes[0, 2].imshow(noisy_windowed, cmap='gray', vmin=0, vmax=1)
    _add_black_border(axes[0, 2])

    # Row 1, Col 3: |Noise| × 20
    diff = np.abs(noise_slice) * 20
    axes[0, 3].imshow(diff, cmap='hot', vmin=0, vmax=1)
    _add_black_border(axes[0, 3])

    # Row 2, Col 0: ROI 掩码
    axes[1, 0].imshow(roi_mask, cmap='gray', vmin=0, vmax=1)
    _add_black_border(axes[1, 0])

    # Row 2, Col 1: 标签叠加原图
    label_rgb = label_to_rgb(label_slice, num_classes)
    overlay = orig_windowed[..., np.newaxis].repeat(3, axis=-1)
    fg_mask = label_slice > 0
    alpha = 0.5
    overlay[fg_mask] = (
        overlay[fg_mask] * (1 - alpha) + label_rgb[fg_mask] / 255.0 * alpha
    )
    axes[1, 1].imshow(overlay)
    _add_black_border(axes[1, 1])

    # Row 2, Col 2: ROI 内噪声
    noise_roi = noise_slice * roi_mask
    noise_roi_rgb = colorize_noise(noise_roi, eps)
    axes[1, 2].imshow(noise_roi_rgb)
    _add_black_border(axes[1, 2])

    # Row 2, Col 3: 噪声能量空间分布
    noise_energy = np.mean(noise_np ** 2, axis=(0, 1))  # [H, W]
    axes[1, 3].imshow(noise_energy, cmap='hot')
    _add_black_border(axes[1, 3])

    plt.savefig(output_path, dpi=150, bbox_inches='tight',
                facecolor='white', pad_inches=0.1)
    plt.close(fig)
    print(f"[Saved] ROI total: {output_path}")


def visualize_roi_noise_small(
    image: torch.Tensor,
    label: torch.Tensor,
    noise: torch.Tensor | None,
    slice_idx: int,
    output_path: str,
    eps: float = 8/255,
):
    """
    小噪声图: 仅显示 ROI 噪声（蓝白红），无图例，黑色边框。
    """
    C, D, H, W = image.shape

    label_np = label.numpy()
    if slice_idx < 0:
        slice_idx = _auto_select_slice(label_np, D)
    slice_idx = min(max(0, slice_idx), D - 1)

    if noise is None:
        noise = torch.zeros_like(image)

    noise_np = noise.numpy()
    noise_slice = noise_np[0, slice_idx]

    # ROI 掩码
    roi_mask = (label_np[slice_idx] > 0).astype(np.float32)
    noise_roi = noise_slice * roi_mask

    noise_rgb = colorize_noise(noise_roi, eps)

    fig, ax = plt.subplots(1, 1, figsize=(4, 4), facecolor='white')
    ax.imshow(noise_rgb, aspect='equal')
    _add_black_border(ax, linewidth=3)

    plt.savefig(output_path, dpi=150, bbox_inches='tight',
                facecolor='white', pad_inches=0.02)
    plt.close(fig)
    print(f"[Saved] ROI small: {output_path}")


def visualize_frequence_noise(
    image: torch.Tensor,
    label: torch.Tensor,
    noise: torch.Tensor | None,
    slice_idx: int,
    output_path: str,
    case_id: str = "",
    eps: float = 8/255,
    window_width: float = 400,
    window_level: float = 40,
    z_cutoff_low: float = 0.1,
    xy_cutoff_high: float = 0.3,
    num_classes: int = 5,
    norm_mode: str = "nnunet",
    global_mean: float = 121.07,
    global_std: float = 49.66,
    z_clip: float = 5.0,
):
    """
    可视化Frequence方法的噪声和频谱约束

    Args:
        image: [C, D, H, W] 原始图像 (归一化到[0,1])
        label: [D, H, W] 分割标签
        noise: [C, D, H, W] 噪声
        slice_idx: 切片索引，-1表示自动选择
        output_path: 输出路径
        case_id: 样本ID
        eps: 噪声范围
        window_width: CT窗宽
        window_level: CT窗位
        z_cutoff_low: Z轴高通截止频率
        xy_cutoff_high: XY低通截止频率
        num_classes: 分割类别数
        norm_mode: 归一化模式 ("nnunet" 或 "ct_window")
        global_mean: nnunet模式下的全局均值
        global_std: nnunet模式下的全局标准差
        z_clip: nnunet模式下的z-score clip范围
    """
    C, D, H, W = image.shape

    # 自动选择切片
    if slice_idx < 0:
        # 找到有标签的中间切片
        label_np = label.numpy()
        label_sum = (label_np > 0).sum(axis=(1, 2))  # [D]
        nonzero_slices = np.where(label_sum > 0)[0]
        if len(nonzero_slices) > 0:
            slice_idx = nonzero_slices[len(nonzero_slices) // 2]
        else:
            slice_idx = D // 2
    slice_idx = min(max(0, slice_idx), D - 1)

    # 如果没有噪声，创建零噪声
    if noise is None:
        noise = torch.zeros_like(image)

    # 转换为 numpy
    image_np = image.numpy()  # [C, D, H, W]
    noise_np = noise.numpy()  # [C, D, H, W]
    label_np = label.numpy()  # [D, H, W]

    # 计算加噪声图像
    noisy_np = np.clip(image_np + noise_np, 0, 1)

    # 构建频域掩码
    M_z, M_xy, M_combined = build_frequency_mask(
        D, H, W,
        z_cutoff_low=z_cutoff_low,
        xy_cutoff_high=xy_cutoff_high,
    )

    # 计算噪声频谱
    spectrum_z, spectrum_xy = compute_noise_spectrum(noise_np)

    # 创建图形
    fig = plt.figure(figsize=(20, 16))
    gs = gridspec.GridSpec(4, 5, figure=fig, hspace=0.3, wspace=0.2)

    # ============ Row 1: 原图、噪声、加噪图、差异 ============
    # 取第一个通道（CT单通道）
    orig_slice = image_np[0, slice_idx]       # [H, W]
    noise_slice = noise_np[0, slice_idx]      # [H, W]
    noisy_slice = noisy_np[0, slice_idx]      # [H, W]

    # 应用窗宽窗位（根据归一化模式选择）
    if norm_mode == "nnunet":
        orig_windowed = apply_window_nnunet(
            orig_slice, window_width, window_level,
            global_mean=global_mean, global_std=global_std, z_clip=z_clip
        )
        noisy_windowed = apply_window_nnunet(
            noisy_slice, window_width, window_level,
            global_mean=global_mean, global_std=global_std, z_clip=z_clip
        )
    else:
        # ct_window模式：数据已经是窗口化的
        orig_windowed = apply_window_direct(orig_slice, window_width, window_level)
        noisy_windowed = apply_window_direct(noisy_slice, window_width, window_level)

    # 1.1 原图
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(orig_windowed, cmap='gray', vmin=0, vmax=1)
    ax1.set_title(f"Original\n(WW={window_width}, WL={window_level})", fontsize=10)
    ax1.axis('off')

    # 1.2 噪声（蓝白红）
    ax2 = fig.add_subplot(gs[0, 1])
    noise_rgb = colorize_noise(noise_slice, eps)
    ax2.imshow(noise_rgb)
    ax2.set_title(f"Noise (eps={eps:.4f})", fontsize=10)
    ax2.axis('off')

    # 1.3 加噪图
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.imshow(noisy_windowed, cmap='gray', vmin=0, vmax=1)
    ax3.set_title("Noisy Image", fontsize=10)
    ax3.axis('off')

    # 1.4 噪声差异（放大）
    ax4 = fig.add_subplot(gs[0, 3])
    diff = np.abs(noise_slice) * 20  # 放大20倍
    ax4.imshow(diff, cmap='hot', vmin=0, vmax=1)
    ax4.set_title("|Noise| x 20", fontsize=10)
    ax4.axis('off')

    # 1.5 噪声统计信息
    ax5 = fig.add_subplot(gs[0, 4])
    ax5.axis('off')
    stats_text = (
        f"Case: {case_id}\n"
        f"Slice: {slice_idx}/{D}\n"
        f"Shape: {H}x{W}\n"
        f"───────────────\n"
        f"Noise Stats:\n"
        f"  max: {noise_np.max():.6f}\n"
        f"  min: {noise_np.min():.6f}\n"
        f"  mean: {noise_np.mean():.6f}\n"
        f"  std: {noise_np.std():.6f}\n"
        f"  L_inf: {np.abs(noise_np).max():.6f}\n"
        f"───────────────\n"
        f"Epsilon: {eps:.6f}\n"
        f"  ({eps*255:.1f}/255)"
    )
    ax5.text(0.1, 0.9, stats_text, fontsize=9, family='monospace',
             verticalalignment='top', transform=ax5.transAxes)

    # ============ Row 2: 频域约束掩码 ============
    # 2.1 Z轴高通掩码（1D）
    ax6 = fig.add_subplot(gs[1, 0])
    freq_z = np.fft.fftfreq(D)
    ax6.plot(freq_z, M_z, 'b-', linewidth=2)
    ax6.axvline(x=z_cutoff_low, color='r', linestyle='--', label=f'cutoff={z_cutoff_low}')
    ax6.axvline(x=-z_cutoff_low, color='r', linestyle='--')
    ax6.set_title("Z-axis HIGH-PASS Mask", fontsize=10)
    ax6.set_xlabel("Frequency (normalized)")
    ax6.set_ylabel("Magnitude")
    ax6.set_xlim(-0.5, 0.5)
    ax6.legend(fontsize=8)
    ax6.grid(True, alpha=0.3)

    # 2.2 XY平面低通掩码（2D）
    ax7 = fig.add_subplot(gs[1, 1])
    M_xy_shifted = np.fft.fftshift(M_xy)
    im7 = ax7.imshow(M_xy_shifted, cmap='viridis', vmin=0, vmax=1)
    ax7.set_title(f"XY LOW-PASS Mask\n(cutoff={xy_cutoff_high})", fontsize=10)
    ax7.axis('off')
    plt.colorbar(im7, ax=ax7, fraction=0.046, pad=0.04)

    # 2.3 组合掩码（中间切片）
    ax8 = fig.add_subplot(gs[1, 2])
    M_mid = M_combined[D // 2]
    M_mid_shifted = np.fft.fftshift(M_mid)
    im8 = ax8.imshow(M_mid_shifted, cmap='viridis', vmin=0, vmax=1)
    ax8.set_title("Combined Mask (mid-slice)", fontsize=10)
    ax8.axis('off')
    plt.colorbar(im8, ax=ax8, fraction=0.046, pad=0.04)

    # 2.4 频域约束说明
    ax9 = fig.add_subplot(gs[1, 3:5])
    ax9.axis('off')
    constraint_text = (
        "Frequency Domain Constraints:\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "Z-axis (slice direction): HIGH-PASS\n"
        f"  • Remove |f_z| < {z_cutoff_low} (low frequencies)\n"
        "  • Effect: Maximize inter-slice diversity\n"
        "  • Adjacent slices have DIFFERENT noise patterns\n"
        "\n"
        "XY-plane (in-plane): LOW-PASS\n"
        f"  • Remove |f_xy| > {xy_cutoff_high} (high frequencies)\n"
        "  • Effect: Ensure intra-slice smoothness\n"
        "  • Noise appears smooth, not grainy\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    )
    ax9.text(0.05, 0.95, constraint_text, fontsize=10, family='monospace',
             verticalalignment='top', transform=ax9.transAxes,
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    # ============ Row 3: 噪声频谱分布 ============
    # 3.1 Z轴频谱
    ax10 = fig.add_subplot(gs[2, 0])
    ax10.plot(freq_z, spectrum_z, 'g-', linewidth=1.5, label='Power')
    ax10.axvline(x=z_cutoff_low, color='r', linestyle='--', alpha=0.7)
    ax10.axvline(x=-z_cutoff_low, color='r', linestyle='--', alpha=0.7)
    ax10.set_title("Noise Z-axis Spectrum", fontsize=10)
    ax10.set_xlabel("Frequency")
    ax10.set_ylabel("Log Power")
    ax10.set_xlim(-0.5, 0.5)
    ax10.grid(True, alpha=0.3)

    # 3.2 XY平面频谱
    ax11 = fig.add_subplot(gs[2, 1])
    spectrum_xy_shifted = np.fft.fftshift(spectrum_xy)
    im11 = ax11.imshow(spectrum_xy_shifted, cmap='magma')
    ax11.set_title("Noise XY Spectrum (log)", fontsize=10)
    ax11.axis('off')
    plt.colorbar(im11, ax=ax11, fraction=0.046, pad=0.04)

    # 3.3 噪声3D频谱的2D投影
    ax12 = fig.add_subplot(gs[2, 2])
    noise_fft_3d = np.fft.fftn(noise_np[0])
    noise_fft_3d = np.fft.fftshift(noise_fft_3d)
    power_3d = np.abs(noise_fft_3d) ** 2
    power_proj = np.log10(np.mean(power_3d, axis=0) + 1e-10)
    im12 = ax12.imshow(power_proj, cmap='magma')
    ax12.set_title("3D Spectrum (XY projection)", fontsize=10)
    ax12.axis('off')
    plt.colorbar(im12, ax=ax12, fraction=0.046, pad=0.04)

    # 3.4-3.5 多切片噪声对比
    ax13 = fig.add_subplot(gs[2, 3])
    ax14 = fig.add_subplot(gs[2, 4])

    # 选择相邻切片对比
    slice1 = max(0, slice_idx - 5)
    slice2 = min(D - 1, slice_idx + 5)

    noise_s1 = colorize_noise(noise_np[0, slice1], eps)
    noise_s2 = colorize_noise(noise_np[0, slice2], eps)

    ax13.imshow(noise_s1)
    ax13.set_title(f"Noise @ slice {slice1}", fontsize=10)
    ax13.axis('off')

    ax14.imshow(noise_s2)
    ax14.set_title(f"Noise @ slice {slice2}", fontsize=10)
    ax14.axis('off')

    # ============ Row 4: 分割标签和图例 ============
    # 4.1 原始标签
    ax15 = fig.add_subplot(gs[3, 0])
    label_slice = label_np[slice_idx]
    label_rgb = label_to_rgb(label_slice, num_classes)
    ax15.imshow(label_rgb)
    ax15.set_title("Ground Truth", fontsize=10)
    ax15.axis('off')

    # 4.2 标签叠加原图
    ax16 = fig.add_subplot(gs[3, 1])
    overlay = orig_windowed[..., np.newaxis].repeat(3, axis=-1)
    label_mask = label_slice > 0
    alpha = 0.5
    overlay[label_mask] = overlay[label_mask] * (1 - alpha) + label_rgb[label_mask] / 255.0 * alpha
    ax16.imshow(overlay)
    ax16.set_title("Label Overlay", fontsize=10)
    ax16.axis('off')

    # 4.3 噪声能量空间分布
    ax17 = fig.add_subplot(gs[3, 2])
    noise_energy = np.mean(noise_np ** 2, axis=(0, 1))  # [H, W]
    im17 = ax17.imshow(noise_energy, cmap='hot')
    ax17.set_title("Noise Energy (spatial)", fontsize=10)
    ax17.axis('off')
    plt.colorbar(im17, ax=ax17, fraction=0.046, pad=0.04)

    # 4.4-4.5 图例
    ax18 = fig.add_subplot(gs[3, 3:5])
    ax18.axis('off')

    # 分割类别图例
    legend_y = 0.9
    ax18.text(0.05, legend_y, "Segmentation Classes:", fontsize=10, fontweight='bold',
              transform=ax18.transAxes)
    for i, (label_name, color) in enumerate(zip(SEG_LABELS[:num_classes], SEG_COLORS[:num_classes])):
        y_pos = legend_y - 0.12 * (i + 1)
        ax18.add_patch(plt.Rectangle((0.05, y_pos - 0.03), 0.08, 0.06,
                                     facecolor=color/255, edgecolor='black',
                                     transform=ax18.transAxes))
        ax18.text(0.15, y_pos, f"{i}: {label_name}", fontsize=9,
                  verticalalignment='center', transform=ax18.transAxes)

    # 噪声颜色图例
    ax18.text(0.55, legend_y, "Noise Colormap:", fontsize=10, fontweight='bold',
              transform=ax18.transAxes)
    ax18.text(0.55, legend_y - 0.12, "Blue: Negative noise", fontsize=9, color='blue',
              transform=ax18.transAxes)
    ax18.text(0.55, legend_y - 0.24, "White: Zero noise", fontsize=9, color='gray',
              transform=ax18.transAxes)
    ax18.text(0.55, legend_y - 0.36, "Red: Positive noise", fontsize=9, color='red',
              transform=ax18.transAxes)

    # 总标题
    fig.suptitle(
        f"Frequence Noise Visualization - {case_id} (Slice {slice_idx}/{D})\n"
        f"Window: WW={window_width}, WL={window_level}",
        fontsize=14, fontweight='bold', y=0.98
    )

    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)

    print(f"[Saved] {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Frequence Noise 可视化")
    parser.add_argument("--noise_dir", type=str, required=True,
                        help="噪声目录 (包含 manifest.json)")
    parser.add_argument("--output_dir", type=str, default="outputs/visualize_freq",
                        help="输出目录")
    parser.add_argument("--dataset_config", type=str, required=True,
                        help="数据集配置文件路径，例如 configs/dataset/flare21.yaml")
    parser.add_argument("--split", type=str, default="train",
                        help="数据集分割 (train/val/test)")
    parser.add_argument("--sample_idx", type=int, default=0,
                        help="样本索引")
    parser.add_argument("--slice_idx", type=int, default=-1,
                        help="切片索引，-1 表示自动选择有标签的中间层")
    parser.add_argument("--eps", type=float, default=4/255,
                        help="噪声范围 epsilon")
    parser.add_argument("--num_samples", type=int, default=1,
                        help="可视化样本数量")

    # CT窗宽窗位参数
    parser.add_argument("--window_width", type=float, default=400,
                        help="CT窗宽 (HU)")
    parser.add_argument("--window_level", type=float, default=40,
                        help="CT窗位 (HU)")

    # 频域约束参数（用于可视化）
    parser.add_argument("--z_cutoff_low", type=float, default=0.1,
                        help="Z轴高通截止频率")
    parser.add_argument("--xy_cutoff_high", type=float, default=0.3,
                        help="XY低通截止频率")

    # 分割类别数
    parser.add_argument("--num_classes", type=int, default=5,
                        help="分割类别数量")

    # nnU-Net归一化参数
    parser.add_argument("--norm_mode", type=str, default="nnunet",
                        choices=["nnunet", "ct_window"],
                        help="归一化模式: nnunet (z-score) 或 ct_window (直接窗口)")
    parser.add_argument("--global_mean", type=float, default=121.07,
                        help="nnunet模式下的全局均值 (FLARE21: 121.07)")
    parser.add_argument("--global_std", type=float, default=49.66,
                        help="nnunet模式下的全局标准差 (FLARE21: 49.66)")
    parser.add_argument("--z_clip", type=float, default=5.0,
                        help="nnunet模式下的z-score clip范围")

    # ROI 可视化模式
    parser.add_argument("--roi_vis", action="store_true",
                        help="生成 ROI-aware 噪声可视化（总图 + 小图，无图例，黑色边框）")

    args = parser.parse_args()

    # 加载配置
    config = load_config(args.dataset_config)

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

        print(f"[Processing] {idx}: {case_id}, shape={tuple(image.shape)}")

        # 提取 key 并加载噪声
        key = extract_key(sample, idx, key_spec)
        noise = load_noise(args.noise_dir, key)

        if noise is None:
            print(f"[Warning] No noise found for {case_id}, using zero noise")

        # 输出文件名
        output_path = os.path.join(args.output_dir, f"{idx:04d}_{case_id}_freq_vis.png")

        if args.roi_vis:
            # ---- ROI-aware 可视化: 总图 + 小图, 无图例, 黑色边框 ----
            roi_total_path = os.path.join(
                args.output_dir, f"{idx:04d}_{case_id}_roi_total.png"
            )
            roi_small_path = os.path.join(
                args.output_dir, f"{idx:04d}_{case_id}_roi_small.png"
            )

            visualize_roi_noise_total(
                image=image,
                label=label,
                noise=noise,
                slice_idx=args.slice_idx,
                output_path=roi_total_path,
                eps=args.eps,
                window_width=args.window_width,
                window_level=args.window_level,
                num_classes=args.num_classes,
                norm_mode=args.norm_mode,
                global_mean=args.global_mean,
                global_std=args.global_std,
                z_clip=args.z_clip,
            )

            visualize_roi_noise_small(
                image=image,
                label=label,
                noise=noise,
                slice_idx=args.slice_idx,
                output_path=roi_small_path,
                eps=args.eps,
            )
        else:
            # ---- 原始频域可视化 ----
            visualize_frequence_noise(
                image=image,
                label=label,
                noise=noise,
                slice_idx=args.slice_idx,
                output_path=output_path,
                case_id=case_id,
                eps=args.eps,
                window_width=args.window_width,
                window_level=args.window_level,
                z_cutoff_low=args.z_cutoff_low,
                xy_cutoff_high=args.xy_cutoff_high,
                num_classes=args.num_classes,
                norm_mode=args.norm_mode,
                global_mean=args.global_mean,
                global_std=args.global_std,
                z_clip=args.z_clip,
            )

    print(f"[Done] Visualized {min(args.num_samples, len(dataset) - args.sample_idx)} samples")


if __name__ == "__main__":
    main()