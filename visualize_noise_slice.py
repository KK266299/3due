#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Inter-Slice Noise Visualization for BraTS19.

可视化方案 A + B + F:
  A. 多切片并列对比图: 连续切片噪声 + 相邻差异
  B. 层间相关性曲线: Pearson 相关系数沿 Z 轴变化
  F. 分割影响可视化: 原图/噪声/GT/Pred(Clean)/Pred(Noisy)/差异

使用示例:
python visualize_noise_slice.py \
    --noise_dir /path/to/noise/epoch_0099 \
    --output_dir outputs/visualize_slice \
    --dataset_config configs/dataset/brats19.yaml \
    --model_path /path/to/best_model.pth \
    --sample_idx 0 \
    --num_samples 3
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.registry import get_dataset_builder, get_model
from src.core.ue_artifacts import UEShardsAccessor
from src.core.ue_keys import extract_key
from src.utils.config import get_config


# ======================== Constants ======================== #

MODALITY_NAMES = ["T1", "T1ce", "T2", "FLAIR"]

# BraTS19: 0=BG, 1=NCR/NET, 2=ED, 3=ET
SEG_COLORS = np.array([
    [0, 0, 0],       # 0: Background
    [255, 0, 0],     # 1: NCR/NET
    [0, 255, 0],     # 2: ED
    [255, 255, 0],   # 3: ET
], dtype=np.uint8)

SEG_LABELS = ["Background", "NCR/NET", "ED", "ET"]


# ======================== Helpers ======================== #

def load_config(dataset_config: str) -> OmegaConf:
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
    if dataset_config and os.path.exists(dataset_config):
        ds_cfg = OmegaConf.load(dataset_config)
        config.dataset = ds_cfg
    else:
        raise ValueError(f"Dataset config not found: {dataset_config}")
    return config


def load_dataset(config: OmegaConf, split: str = "train"):
    ds_name = config.dataset.name
    if ds_name == "brats19":
        ds_name = "brats19_seg"
    builder_cls = get_dataset_builder(ds_name)
    builder = builder_cls(config)
    return builder.get_dataset(split=split)


def load_noise(noise_dir: str, key: str) -> Optional[torch.Tensor]:
    # Accept both directory and direct manifest path
    if noise_dir.endswith("manifest.json") and os.path.isfile(noise_dir):
        manifest_path = noise_dir
    else:
        manifest_path = os.path.join(noise_dir, "manifest.json")
    if not os.path.exists(manifest_path):
        print(f"[Warning] Manifest not found: {manifest_path}")
        return None
    accessor = UEShardsAccessor.from_manifest(manifest_path)
    try:
        return accessor.get(key, perturb_type="samplewise")
    except KeyError:
        print(f"[Warning] Noise not found for key={key}")
        return None


def load_model(model_path: str, config: OmegaConf, device: torch.device) -> torch.nn.Module:
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


def run_inference(model: torch.nn.Module, image: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Return argmax prediction [D, H, W]."""
    with torch.no_grad():
        x = image.unsqueeze(0).to(device)
        out = model(x)
        if isinstance(out, (tuple, list)):
            out = out[0]
        pred = out.argmax(dim=1).squeeze(0).cpu()
    return pred


def colorize_noise(noise_2d: np.ndarray, eps: float = 8/255) -> np.ndarray:
    """[H, W] noise -> [H, W, 3] uint8 BWR colormap."""
    normalized = np.clip(noise_2d / max(eps, 1e-12), -1, 1)
    rgb = np.zeros((*noise_2d.shape, 3), dtype=np.float32)
    neg = normalized < 0
    rgb[neg, 2] = 1.0
    rgb[neg, 0] = 1.0 + normalized[neg]
    rgb[neg, 1] = 1.0 + normalized[neg]
    pos = normalized >= 0
    rgb[pos, 0] = 1.0
    rgb[pos, 1] = 1.0 - normalized[pos]
    rgb[pos, 2] = 1.0 - normalized[pos]
    return (rgb * 255).astype(np.uint8)


def label_to_rgb(label_2d: np.ndarray, num_classes: int = 4) -> np.ndarray:
    """[H, W] label -> [H, W, 3] uint8 RGB."""
    h, w = label_2d.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_id in range(min(num_classes, len(SEG_COLORS))):
        mask = label_2d == cls_id
        rgb[mask] = SEG_COLORS[cls_id]
    return rgb


def auto_select_slice(label_np: np.ndarray, D: int) -> int:
    """Select a slice with maximum foreground area."""
    label_sum = (label_np > 0).sum(axis=(1, 2))  # [D]
    nonzero = np.where(label_sum > 0)[0]
    if len(nonzero) > 0:
        # Pick the slice with the most foreground
        best_idx = nonzero[np.argmax(label_sum[nonzero])]
        return int(best_idx)
    return D // 2


def compute_inter_slice_correlation(noise_np: np.ndarray, channel: int = 0) -> List[float]:
    """
    Compute Pearson correlation between adjacent slices.

    Args:
        noise_np: [C, D, H, W]
        channel: which channel to use

    Returns:
        List of length D-1, correlation[i] = corr(noise[z_i], noise[z_{i+1}])
    """
    noise_3d = noise_np[channel]  # [D, H, W]
    D = noise_3d.shape[0]
    correlations = []
    for z in range(D - 1):
        s1 = noise_3d[z].flatten()
        s2 = noise_3d[z + 1].flatten()
        std1, std2 = np.std(s1), np.std(s2)
        if std1 > 1e-8 and std2 > 1e-8:
            corr = np.corrcoef(s1, s2)[0, 1]
        else:
            corr = 0.0
        correlations.append(float(corr))
    return correlations


# ======================== Scheme A ======================== #

def visualize_scheme_a(
    noise_np: np.ndarray,
    image_np: np.ndarray,
    label_np: np.ndarray,
    center_z: int,
    output_path: str,
    case_id: str = "",
    eps: float = 8/255,
    n_slices: int = 5,
    channel: int = 0,
):
    """
    Scheme A: Multi-slice comparison.

    Row 1: Original images (consecutive slices)
    Row 2: Noise patterns (BWR) for each slice
    Row 3: Inter-slice noise difference |δ[z+1] - δ[z]|
    """
    C, D, H, W = noise_np.shape

    # Compute start index, centered around center_z
    half = n_slices // 2
    start = max(0, min(center_z - half, D - n_slices))
    slices = list(range(start, min(start + n_slices, D)))
    n = len(slices)

    fig, axes = plt.subplots(3, n, figsize=(3.5 * n, 10.5), facecolor='white')
    fig.subplots_adjust(wspace=0.05, hspace=0.15)

    # Row 1: Original images (channel 0)
    for i, z in enumerate(slices):
        axes[0, i].imshow(image_np[channel, z], cmap='gray', vmin=0, vmax=1)
        axes[0, i].set_title(f'z={z}', fontsize=10)
        axes[0, i].axis('off')
    axes[0, 0].set_ylabel('Original', fontsize=12, fontweight='bold')

    # Row 2: Noise (BWR)
    for i, z in enumerate(slices):
        noise_rgb = colorize_noise(noise_np[channel, z], eps)
        axes[1, i].imshow(noise_rgb)
        axes[1, i].axis('off')
    axes[1, 0].set_ylabel('Noise (BWR)', fontsize=12, fontweight='bold')

    # Row 3: Adjacent difference |δ[z+1] - δ[z]|
    for i in range(n):
        if i < n - 1:
            z0, z1 = slices[i], slices[i + 1]
            diff = np.abs(noise_np[channel, z1] - noise_np[channel, z0])
            # Normalize: max possible diff is 2*eps
            diff_norm = diff / (2 * eps + 1e-12)
            axes[2, i].imshow(diff_norm, cmap='hot', vmin=0, vmax=1)
            axes[2, i].set_title(f'|δ[{z1}]-δ[{z0}]|', fontsize=8)
        else:
            axes[2, i].axis('off')
        axes[2, i].axis('off')
    axes[2, 0].set_ylabel('Slice Diff', fontsize=12, fontweight='bold')

    fig.suptitle(
        f'Scheme A: Multi-Slice Noise Comparison - {case_id}\n'
        f'Slices {slices[0]}~{slices[-1]} (channel {MODALITY_NAMES[channel]})',
        fontsize=13, fontweight='bold', y=0.98,
    )

    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"[Saved] Scheme A: {output_path}")


# ======================== Scheme B ======================== #

def visualize_scheme_b(
    noise_np: np.ndarray,
    output_path: str,
    case_id: str = "",
    channel: int = 0,
):
    """
    Scheme B: Inter-slice correlation curve.

    Main plot: Pearson correlation between adjacent slices along Z axis.
    Sub-plots: XZ and YZ cross-sections of noise.
    """
    C, D, H, W = noise_np.shape
    correlations = compute_inter_slice_correlation(noise_np, channel)

    fig = plt.figure(figsize=(16, 8), facecolor='white')
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)

    # --- Main: correlation curve ---
    ax_main = fig.add_subplot(gs[0, :2])
    x = np.arange(len(correlations))
    ax_main.plot(x, correlations, 'b-', linewidth=1.2, alpha=0.8, label='Adjacent corr.')

    mean_corr = np.mean(correlations)
    ax_main.axhline(y=mean_corr, color='r', linestyle='--', linewidth=1.5,
                     label=f'Mean = {mean_corr:.4f}')
    ax_main.axhline(y=0, color='gray', linestyle=':', linewidth=0.8)

    # Shade ±1 std
    std_corr = np.std(correlations)
    ax_main.fill_between(x, mean_corr - std_corr, mean_corr + std_corr,
                         alpha=0.15, color='red', label=f'±1σ = {std_corr:.4f}')

    ax_main.set_xlabel('Slice Index (z)', fontsize=11)
    ax_main.set_ylabel('Pearson Correlation', fontsize=11)
    ax_main.set_title(
        f'Inter-Slice Noise Correlation ({MODALITY_NAMES[channel]})',
        fontsize=12, fontweight='bold',
    )
    ax_main.set_xlim(0, len(correlations) - 1)
    ax_main.set_ylim(-1.05, 1.05)
    ax_main.legend(fontsize=9, loc='upper right')
    ax_main.grid(True, alpha=0.3)

    # --- Stats box ---
    ax_stats = fig.add_subplot(gs[0, 2])
    ax_stats.axis('off')
    stats_text = (
        f"Statistics\n"
        f"{'─' * 28}\n"
        f"Case:     {case_id}\n"
        f"Channel:  {MODALITY_NAMES[channel]}\n"
        f"Depth:    {D} slices\n"
        f"{'─' * 28}\n"
        f"Mean corr:  {mean_corr:+.4f}\n"
        f"Std  corr:   {std_corr:.4f}\n"
        f"Min  corr:  {np.min(correlations):+.4f}\n"
        f"Max  corr:  {np.max(correlations):+.4f}\n"
        f"Median:     {np.median(correlations):+.4f}\n"
        f"{'─' * 28}\n"
        f"L∞ noise:   {np.abs(noise_np).max():.6f}\n"
        f"L2 noise:   {np.sqrt(np.mean(noise_np**2)):.6f}\n"
    )
    ax_stats.text(0.05, 0.95, stats_text, fontsize=10, family='monospace',
                  verticalalignment='top', transform=ax_stats.transAxes,
                  bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    # --- XZ cross-section ---
    ax_xz = fig.add_subplot(gs[1, 0])
    noise_ch = noise_np[channel]  # [D, H, W]
    vmax = np.abs(noise_ch).max()
    ax_xz.imshow(noise_ch[:, H // 2, :], cmap='RdBu_r', vmin=-vmax, vmax=vmax, aspect='auto')
    ax_xz.set_title('XZ Cross-Section (y=H/2)', fontsize=10)
    ax_xz.set_xlabel('x')
    ax_xz.set_ylabel('z (depth)')

    # --- YZ cross-section ---
    ax_yz = fig.add_subplot(gs[1, 1])
    ax_yz.imshow(noise_ch[:, :, W // 2], cmap='RdBu_r', vmin=-vmax, vmax=vmax, aspect='auto')
    ax_yz.set_title('YZ Cross-Section (x=W/2)', fontsize=10)
    ax_yz.set_xlabel('y')
    ax_yz.set_ylabel('z (depth)')

    # --- Histogram of correlations ---
    ax_hist = fig.add_subplot(gs[1, 2])
    ax_hist.hist(correlations, bins=30, color='steelblue', edgecolor='white', alpha=0.8)
    ax_hist.axvline(x=mean_corr, color='r', linestyle='--', linewidth=1.5)
    ax_hist.set_title('Correlation Distribution', fontsize=10)
    ax_hist.set_xlabel('Pearson Correlation')
    ax_hist.set_ylabel('Count')
    ax_hist.grid(True, alpha=0.3)

    fig.suptitle(
        f'Scheme B: Inter-Slice Correlation Analysis - {case_id}',
        fontsize=13, fontweight='bold', y=1.01,
    )

    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"[Saved] Scheme B: {output_path}")


# ======================== Scheme F ======================== #

def visualize_scheme_f(
    image: torch.Tensor,
    label: torch.Tensor,
    noise: Optional[torch.Tensor],
    pred_clean: torch.Tensor,
    pred_noisy: torch.Tensor,
    slice_idx: int,
    output_path: str,
    case_id: str = "",
    eps: float = 8/255,
):
    """
    Scheme F: Segmentation impact visualization.

    Row 1 (4 modalities): Original | Noise (BWR) | Noisy | |δ|×10
    Row 2: GT | Pred(Clean) | Pred(Noisy) | Pred Diff
    """
    C, D, H, W = image.shape

    if noise is None:
        noise = torch.zeros_like(image)

    image_np = image.numpy()
    noise_np = noise.numpy()
    label_np = label.numpy()
    pred_clean_np = pred_clean.numpy()
    pred_noisy_np = pred_noisy.numpy()

    noisy_np = np.clip(image_np + noise_np, 0, 1)
    label_slice = label_np[slice_idx]
    pred_c_slice = pred_clean_np[slice_idx]
    pred_n_slice = pred_noisy_np[slice_idx]

    # Layout: (C+1) rows x 4 cols
    # Rows 0..C-1: each modality
    # Row C: segmentation results
    n_rows = C + 1
    fig, axes = plt.subplots(n_rows, 4, figsize=(16, 4 * n_rows), facecolor='white')
    fig.subplots_adjust(wspace=0.05, hspace=0.12)

    col_titles = ["Original", "Noise (BWR)", "Noisy", "|δ| × 10"]

    # Rows 0..C-1: modalities
    for row in range(C):
        mod_name = MODALITY_NAMES[row] if row < len(MODALITY_NAMES) else f"Ch{row}"
        orig_s = image_np[row, slice_idx]
        noise_s = noise_np[row, slice_idx]
        noisy_s = noisy_np[row, slice_idx]
        diff_s = np.clip(np.abs(noise_s) * 10, 0, 1)

        axes[row, 0].imshow(orig_s, cmap='gray', vmin=0, vmax=1)
        axes[row, 0].set_ylabel(mod_name, fontsize=12, fontweight='bold')

        noise_rgb = colorize_noise(noise_s, eps)
        axes[row, 1].imshow(noise_rgb)

        axes[row, 2].imshow(noisy_s, cmap='gray', vmin=0, vmax=1)

        axes[row, 3].imshow(diff_s, cmap='hot', vmin=0, vmax=1)

        for col in range(4):
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])

    # Last row: segmentation
    seg_row = C

    # GT
    gt_rgb = label_to_rgb(label_slice)
    axes[seg_row, 0].imshow(gt_rgb)
    axes[seg_row, 0].set_ylabel("Seg", fontsize=12, fontweight='bold')
    axes[seg_row, 0].set_xlabel("Ground Truth", fontsize=9)

    # Pred clean
    pred_c_rgb = label_to_rgb(pred_c_slice)
    axes[seg_row, 1].imshow(pred_c_rgb)
    axes[seg_row, 1].set_xlabel("Pred (Clean)", fontsize=9)

    # Pred noisy
    pred_n_rgb = label_to_rgb(pred_n_slice)
    axes[seg_row, 2].imshow(pred_n_rgb)
    axes[seg_row, 2].set_xlabel("Pred (Noisy)", fontsize=9)

    # Pred difference
    pred_diff = (pred_c_slice != pred_n_slice).astype(np.float32)
    axes[seg_row, 3].imshow(pred_diff, cmap='hot', vmin=0, vmax=1)
    axes[seg_row, 3].set_xlabel("Pred Difference", fontsize=9)

    for col in range(4):
        axes[seg_row, col].set_xticks([])
        axes[seg_row, col].set_yticks([])

    # Column titles
    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontsize=12, fontweight='bold')

    # Compute Dice degradation for subtitle
    def dice_score(pred, gt, num_classes=4):
        dices = []
        for c in range(1, num_classes):  # skip background
            p = (pred == c)
            g = (gt == c)
            intersection = np.logical_and(p, g).sum()
            union = p.sum() + g.sum()
            if union == 0:
                continue
            dices.append(2.0 * intersection / union)
        return np.mean(dices) if dices else 0.0

    dice_clean = dice_score(pred_c_slice, label_slice)
    dice_noisy = dice_score(pred_n_slice, label_slice)

    fig.suptitle(
        f'Scheme F: Segmentation Impact - {case_id} (Slice {slice_idx}/{D})\n'
        f'Dice: Clean={dice_clean:.4f} → Noisy={dice_noisy:.4f} '
        f'(Δ={dice_noisy - dice_clean:+.4f})',
        fontsize=13, fontweight='bold', y=0.99,
    )

    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"[Saved] Scheme F: {output_path}")


# ======================== Main ======================== #

def main():
    parser = argparse.ArgumentParser(description="Inter-Slice Noise Visualization (BraTS19)")
    parser.add_argument("--noise_dir", type=str, required=True,
                        help="Noise directory (contains manifest.json)")
    parser.add_argument("--output_dir", type=str, default="outputs/visualize_slice",
                        help="Output directory")
    parser.add_argument("--dataset_config", type=str, default="configs/dataset/brats19.yaml",
                        help="Dataset config path")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Segmentation model checkpoint (required for Scheme F)")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--sample_idx", type=int, default=0,
                        help="Starting sample index")
    parser.add_argument("--num_samples", type=int, default=3,
                        help="Number of samples to visualize")
    parser.add_argument("--slice_idx", type=int, default=-1,
                        help="Slice index, -1 = auto (max foreground)")
    parser.add_argument("--eps", type=float, default=8/255,
                        help="Noise epsilon for colormap range")
    parser.add_argument("--n_slices", type=int, default=5,
                        help="Number of consecutive slices for Scheme A")
    parser.add_argument("--channel", type=int, default=0,
                        help="Channel index for Scheme A/B (0=T1, 1=T1ce, 2=T2, 3=FLAIR)")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--schemes", type=str, default="a,b,f",
                        help="Comma-separated schemes to run: a,b,f")

    args = parser.parse_args()

    # Parse schemes
    schemes = set(s.strip().lower() for s in args.schemes.split(","))

    # Load config & dataset
    config = load_config(args.dataset_config)
    print(f"[Loading Dataset] split={args.split}")
    dataset = load_dataset(config, split=args.split)
    print(f"[Dataset] {len(dataset)} samples")

    # Load model (for scheme F)
    seg_model = None
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if "f" in schemes and args.model_path:
        print(f"[Loading Model] {args.model_path}")
        seg_model = load_model(args.model_path, config, device)
        print(f"[Model] loaded on {device}")
    elif "f" in schemes:
        print("[Warning] Scheme F requested but no --model_path provided, skipping F")
        schemes.discard("f")

    os.makedirs(args.output_dir, exist_ok=True)

    key_spec = get_config(config, "ue.key", {})

    for i in range(args.num_samples):
        idx = args.sample_idx + i
        if idx >= len(dataset):
            print(f"[Skip] index {idx} out of range")
            break

        sample = dataset[idx]
        image = sample["image"]   # [C, D, H, W]
        label = sample["label"]   # [D, H, W]
        case_id = sample.get("case_id", f"sample_{idx}")
        C, D, H, W = image.shape

        print(f"\n[Processing] {idx}: {case_id}, shape={tuple(image.shape)}")

        # Load noise
        key = extract_key(sample, idx, key_spec)
        noise = load_noise(args.noise_dir, key)

        if noise is None:
            print(f"  [Warning] No noise found, using zeros")
            noise = torch.zeros_like(image)

        image_np = image.numpy()
        noise_np = noise.numpy()
        label_np = label.numpy()

        # Auto-select slice
        if args.slice_idx < 0:
            center_z = auto_select_slice(label_np, D)
        else:
            center_z = min(max(0, args.slice_idx), D - 1)

        prefix = f"{idx:04d}_{case_id}"

        # --- Scheme A ---
        if "a" in schemes:
            path_a = os.path.join(args.output_dir, f"{prefix}_A_multi_slice.png")
            visualize_scheme_a(
                noise_np=noise_np,
                image_np=image_np,
                label_np=label_np,
                center_z=center_z,
                output_path=path_a,
                case_id=case_id,
                eps=args.eps,
                n_slices=args.n_slices,
                channel=args.channel,
            )

        # --- Scheme B ---
        if "b" in schemes:
            path_b = os.path.join(args.output_dir, f"{prefix}_B_correlation.png")
            visualize_scheme_b(
                noise_np=noise_np,
                output_path=path_b,
                case_id=case_id,
                channel=args.channel,
            )

        # --- Scheme F ---
        if "f" in schemes and seg_model is not None:
            pred_clean = run_inference(seg_model, image, device)
            noisy_image = (image + noise).clamp(0, 1)
            pred_noisy = run_inference(seg_model, noisy_image, device)

            path_f = os.path.join(args.output_dir, f"{prefix}_F_seg_impact.png")
            visualize_scheme_f(
                image=image,
                label=label,
                noise=noise,
                pred_clean=pred_clean,
                pred_noisy=pred_noisy,
                slice_idx=center_z,
                output_path=path_f,
                case_id=case_id,
                eps=args.eps,
            )

    print(f"\n[Done] Output: {args.output_dir}")


if __name__ == "__main__":
    main()
