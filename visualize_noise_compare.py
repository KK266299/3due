#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Multi-Method Noise Comparison Visualization for BraTS19.

在同一张图里对比多个方法 (e.g. Ours vs EM) 的:
  A. 多切片噪声模式对比
  B. 层间相关性曲线对比
  F. 分割影响对比

使用示例:
python visualize_noise_compare.py \
    --method Ours \
        /home/.../nofreq_learnable.../noise/epoch_0099/manifest.json \
        /home/.../victim_nofreq_learnable.../best_model.pth \
    --method EM \
        /home/.../minmin.../noise/epoch_0099/manifest.json \
        /home/.../victim_minmin.../best_model.pth \
    --dataset_config configs/dataset/brats19.yaml \
    --output_dir outputs/visualize_compare \
    --num_samples 3
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import numpy as np
import torch
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.registry import get_dataset_builder, get_model
from src.core.ue_artifacts import UEShardsAccessor
from src.core.ue_keys import extract_key
from src.utils.config import get_config
from src.utils.ssim import ssim as ssim_fn
from src.utils.eval_metrics import compute_noise_jacobian_metrics


# ======================== Constants ======================== #

MODALITY_NAMES = ["T1", "T1ce", "T2", "FLAIR"]

SEG_COLORS = np.array([
    [0, 0, 0],       # 0: Background
    [255, 0, 0],     # 1: NCR/NET
    [0, 255, 0],     # 2: ED
    [255, 255, 0],   # 3: ET
], dtype=np.uint8)

# Distinct colors for methods
METHOD_COLORS = [
    "#2196F3",  # blue   (method 1)
    "#F44336",  # red    (method 2)
    "#4CAF50",  # green  (method 3)
    "#FF9800",  # orange (method 4)
    "#9C27B0",  # purple (method 5)
    "#00BCD4",  # cyan   (method 6)
]

METHOD_CMAPS = ["RdBu_r", "PiYG_r", "PuOr_r", "BrBG_r", "coolwarm", "seismic"]

# Noise colorbar range
NOISE_VMIN = -4 / 255
NOISE_VMAX = 4 / 255
NOISE_CMAP = "bwr"


def _add_noise_colorbar(fig, axes_list, location="right", shrink=0.6, pad=0.02,
                        label="Noise Intensity", fontsize=9):
    """Add a shared colorbar for noise images with range [-4/255, 4/255]."""
    norm = Normalize(vmin=NOISE_VMIN, vmax=NOISE_VMAX)
    sm = ScalarMappable(norm=norm, cmap=NOISE_CMAP)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes_list, location=location,
                        shrink=shrink, pad=pad, aspect=30)
    cbar.set_ticks([NOISE_VMIN, NOISE_VMIN / 2, 0, NOISE_VMAX / 2, NOISE_VMAX])
    cbar.set_ticklabels(["-4/255", "-2/255", "0", "2/255", "4/255"])
    if label:
        cbar.set_label(label, fontsize=fontsize)
    return cbar


# ======================== Helpers (reuse from single) ======================== #

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


def load_noise(noise_path: str, key: str) -> Optional[torch.Tensor]:
    if noise_path.endswith("manifest.json") and os.path.isfile(noise_path):
        manifest_path = noise_path
    else:
        manifest_path = os.path.join(noise_path, "manifest.json")
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
    with torch.no_grad():
        x = image.unsqueeze(0).to(device)
        out = model(x)
        if isinstance(out, (tuple, list)):
            out = out[0]
        pred = out.argmax(dim=1).squeeze(0).cpu()
    return pred


def label_to_rgb(label_2d: np.ndarray, num_classes: int = 4) -> np.ndarray:
    h, w = label_2d.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_id in range(min(num_classes, len(SEG_COLORS))):
        rgb[label_2d == cls_id] = SEG_COLORS[cls_id]
    return rgb


def auto_select_slice(label_np: np.ndarray, D: int) -> int:
    label_sum = (label_np > 0).sum(axis=(1, 2))
    nonzero = np.where(label_sum > 0)[0]
    if len(nonzero) > 0:
        return int(nonzero[np.argmax(label_sum[nonzero])])
    return D // 2


def compute_inter_slice_metrics(
    noise: torch.Tensor,
    channel: int = 0,
) -> Dict[str, List[float]]:
    """
    Compute multiple inter-slice similarity metrics along depth.

    Uses existing project metrics:
      - Pearson correlation (numpy)
      - SSIM (src.utils.ssim)
      - L2 distance (SliceConsistencyLoss._l2_consistency style)
      - Cosine similarity
      - Total Variation along depth (src.utils.eval_metrics style)

    Args:
        noise: [C, D, H, W] tensor
        channel: which channel to compute on

    Returns:
        Dict of metric_name -> List[float] of length D-1
    """
    noise_ch = noise[channel]  # [D, H, W]
    D, H, W = noise_ch.shape

    pearson = []
    ssim_vals = []
    l2_dist = []
    cosine_sim = []

    for z in range(D - 1):
        s1 = noise_ch[z]  # [H, W]
        s2 = noise_ch[z + 1]

        # --- Pearson correlation (numpy, fast) ---
        f1, f2 = s1.numpy().flatten(), s2.numpy().flatten()
        if np.std(f1) > 1e-8 and np.std(f2) > 1e-8:
            pearson.append(float(np.corrcoef(f1, f2)[0, 1]))
        else:
            pearson.append(0.0)

        # --- SSIM (reuse project's ssim function) ---
        # ssim expects [B, C, H, W]; data_range = noise amplitude range
        x1 = s1.unsqueeze(0).unsqueeze(0).float()  # [1,1,H,W]
        x2 = s2.unsqueeze(0).unsqueeze(0).float()
        vrange = max(float(noise_ch.max() - noise_ch.min()), 1e-8)
        try:
            sv = ssim_fn(x1, x2, data_range=vrange, size_average=True,
                         win_size=min(7, min(H, W) - 1 if min(H, W) % 2 == 0 else min(H, W)))
            ssim_vals.append(float(sv.item()))
        except Exception:
            ssim_vals.append(0.0)

        # --- L2 distance (same as SliceConsistencyLoss._l2_consistency) ---
        diff_sq = (s2 - s1).pow(2).mean()
        l2_dist.append(float(diff_sq.sqrt().item()))

        # --- Cosine similarity ---
        v1, v2 = s1.flatten(), s2.flatten()
        dot = torch.dot(v1, v2)
        n1, n2 = v1.norm(), v2.norm()
        if n1 > 1e-8 and n2 > 1e-8:
            cosine_sim.append(float((dot / (n1 * n2)).item()))
        else:
            cosine_sim.append(0.0)

    # --- Total Variation along depth (per-slice TV) ---
    # This is |δ[z+1] - δ[z]| summed over spatial, for each z pair
    # (from compute_noise_jacobian_metrics convention)
    depth_tv = []
    for z in range(D - 1):
        diff_abs = (noise_ch[z + 1] - noise_ch[z]).abs().mean()
        depth_tv.append(float(diff_abs.item()))

    return {
        "Pearson Corr.": pearson,
        "SSIM": ssim_vals,
        "L2 Dist.": l2_dist,
        "Cosine Sim.": cosine_sim,
        "Depth TV": depth_tv,
    }


def dice_score(pred: np.ndarray, gt: np.ndarray, num_classes: int = 4) -> float:
    dices = []
    for c in range(1, num_classes):
        p, g = (pred == c), (gt == c)
        inter = np.logical_and(p, g).sum()
        union = p.sum() + g.sum()
        if union > 0:
            dices.append(2.0 * inter / union)
    return float(np.mean(dices)) if dices else 0.0


# ======================== Data structure for a method ======================== #

class MethodData:
    """Holds loaded noise/model/predictions for one method."""

    def __init__(self, name: str, noise_path: str, model_path: str):
        self.name = name
        self.noise_path = noise_path
        self.model_path = model_path
        self.model: Optional[torch.nn.Module] = None

    def load_model(self, config: OmegaConf, device: torch.device):
        print(f"  [Loading Model] {self.name}: {self.model_path}")
        self.model = load_model(self.model_path, config, device)

    def get_noise(self, key: str) -> Optional[torch.Tensor]:
        return load_noise(self.noise_path, key)


# ======================== Compare A ======================== #

def compare_scheme_a(
    methods: List[MethodData],
    noises: List[np.ndarray],
    image_np: np.ndarray,
    center_z: int,
    output_path: str,
    case_id: str = "",
    eps: float = 8/255,
    n_slices: int = 5,
    channel: int = 0,
):
    """
    Comparison Scheme A: side-by-side multi-slice noise.

    Row 0:          Original images
    Row 1..M:       Noise (BWR) per method
    Row M+1..2M:    Inter-slice diff per method
    """
    C, D, H, W = image_np.shape
    M = len(methods)

    half = n_slices // 2
    start = max(0, min(center_z - half, D - n_slices))
    slices = list(range(start, min(start + n_slices, D)))
    n = len(slices)

    n_rows = 1 + 2 * M  # original + (noise + diff) per method
    fig, axes = plt.subplots(n_rows, n, figsize=(3.5 * n, 3.2 * n_rows), facecolor="white")
    fig.subplots_adjust(wspace=0.04, hspace=0.12)

    # Row 0: Original
    for i, z in enumerate(slices):
        axes[0, i].imshow(image_np[channel, z], cmap="gray", vmin=0, vmax=1)
        axes[0, i].set_title(f"z={z}", fontsize=10)
        axes[0, i].axis("off")
    axes[0, 0].set_ylabel("Original", fontsize=11, fontweight="bold")

    noise_axes = []  # collect noise axes for shared colorbar
    for m_idx, (method, noise_np) in enumerate(zip(methods, noises)):
        row_noise = 1 + m_idx * 2
        row_diff = 2 + m_idx * 2

        # Noise row — use diverging colormap with fixed range
        for i, z in enumerate(slices):
            axes[row_noise, i].imshow(
                noise_np[channel, z], cmap=NOISE_CMAP,
                vmin=NOISE_VMIN, vmax=NOISE_VMAX)
            axes[row_noise, i].axis("off")
            noise_axes.append(axes[row_noise, i])
        axes[row_noise, 0].set_ylabel(
            f"{method.name}\nNoise", fontsize=10, fontweight="bold"
        )

        # Diff row
        for i in range(n):
            if i < n - 1:
                z0, z1 = slices[i], slices[i + 1]
                diff = np.abs(noise_np[channel, z1] - noise_np[channel, z0])
                diff_norm = diff / (2 * eps + 1e-12)
                axes[row_diff, i].imshow(diff_norm, cmap="hot", vmin=0, vmax=1)
                axes[row_diff, i].set_title(f"|Δ|", fontsize=7)
            axes[row_diff, i].axis("off")
        axes[row_diff, 0].set_ylabel(
            f"{method.name}\nSlice Diff", fontsize=10, fontweight="bold"
        )

    # Add shared colorbar for noise rows
    _add_noise_colorbar(fig, noise_axes, shrink=0.8, pad=0.03)

    fig.suptitle(
        f"Compare A: Multi-Slice Noise — {case_id}\n"
        f"Slices {slices[0]}~{slices[-1]} ({MODALITY_NAMES[channel]}), "
        f"Methods: {' vs '.join(m.name for m in methods)}",
        fontsize=13, fontweight="bold", y=0.99,
    )

    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[Saved] Compare A: {output_path}")


# ======================== Compare B ======================== #

def compare_scheme_b(
    methods: List[MethodData],
    noises: List[torch.Tensor],
    output_path: str,
    case_id: str = "",
    channel: int = 0,
):
    """
    Comparison Scheme B: multi-metric inter-slice similarity comparison.

    Uses existing project metrics:
      - Pearson Correlation
      - SSIM (src.utils.ssim)
      - L2 Distance (SliceConsistencyLoss style)
      - Cosine Similarity
      - Depth Total Variation

    Layout:
      Row 0: Pearson Correlation curves + box
      Row 1: SSIM curves + box
      Row 2: L2 Distance curves + box
      Row 3: Cosine Similarity curves + box
      Right column: summary stats table
    """
    M = len(methods)

    # Compute all metrics for each method
    print(f"  [Scheme B] Computing inter-slice metrics ...")
    all_metrics: List[Dict[str, List[float]]] = []
    for m_idx, noise_t in enumerate(noises):
        metrics = compute_inter_slice_metrics(noise_t, channel)
        all_metrics.append(metrics)
        print(f"    {methods[m_idx].name}: Pearson μ={np.mean(metrics['Pearson Corr.']):+.4f}, "
              f"SSIM μ={np.mean(metrics['SSIM']):.4f}, "
              f"L2 μ={np.mean(metrics['L2 Dist.']):.6f}, "
              f"Cosine μ={np.mean(metrics['Cosine Sim.']):+.4f}")

    # Select metrics for curve plots (similarity-type and distance-type)
    curve_metrics = ["Pearson Corr.", "SSIM", "Cosine Sim.", "L2 Dist.", "Depth TV"]
    # Which metrics are "higher = more similar" vs "higher = more different"
    similarity_type = {"Pearson Corr.", "SSIM", "Cosine Sim."}

    n_curves = len(curve_metrics)
    fig = plt.figure(figsize=(22, 3.5 * n_curves), facecolor="white")
    gs = gridspec.GridSpec(
        n_curves, 3, figure=fig,
        width_ratios=[4, 1.2, 2],
        hspace=0.35, wspace=0.25,
    )

    for row, metric_name in enumerate(curve_metrics):
        is_sim = metric_name in similarity_type

        # --- Left: Curves ---
        ax_curve = fig.add_subplot(gs[row, 0])
        all_vals = []
        for m_idx, method in enumerate(methods):
            vals = all_metrics[m_idx][metric_name]
            all_vals.append(vals)
            color = METHOD_COLORS[m_idx % len(METHOD_COLORS)]
            x = np.arange(len(vals))
            mean_v = np.mean(vals)
            ax_curve.plot(x, vals, color=color, linewidth=1.0, alpha=0.8,
                          label=f"{method.name} (μ={mean_v:.4f})")
            ax_curve.axhline(y=mean_v, color=color, linestyle="--",
                             linewidth=0.8, alpha=0.4)

        if is_sim:
            ax_curve.axhline(y=0, color="gray", linestyle=":", linewidth=0.6)
        ax_curve.set_xlabel("Slice Index (z)", fontsize=9)
        ax_curve.set_ylabel(metric_name, fontsize=10)
        ax_curve.set_title(metric_name, fontsize=11, fontweight="bold")
        ax_curve.legend(fontsize=8, loc="upper right")
        ax_curve.grid(True, alpha=0.25)
        if is_sim:
            ax_curve.set_ylim(-1.05, 1.05)

        # --- Middle: Box plot ---
        ax_box = fig.add_subplot(gs[row, 1])
        bp = ax_box.boxplot(
            all_vals,
            labels=[m.name for m in methods],
            patch_artist=True,
            widths=0.55,
        )
        for m_idx, patch in enumerate(bp["boxes"]):
            color = METHOD_COLORS[m_idx % len(METHOD_COLORS)]
            patch.set_facecolor(color)
            patch.set_alpha(0.3)
        ax_box.set_title("Distrib.", fontsize=9, fontweight="bold")
        ax_box.grid(True, alpha=0.25, axis="y")
        if is_sim:
            ax_box.axhline(y=0, color="gray", linestyle=":", linewidth=0.6)

    # --- Right column: comprehensive stats table ---
    ax_stats = fig.add_subplot(gs[:, 2])
    ax_stats.axis("off")

    lines = []
    lines.append("Inter-Slice Similarity Summary")
    lines.append("═" * 52)
    lines.append(f"Case: {case_id}")
    lines.append(f"Channel: {MODALITY_NAMES[channel]}")
    lines.append(f"Depth: {noises[0].shape[1]} slices")
    lines.append("")

    for metric_name in curve_metrics:
        is_sim = metric_name in similarity_type
        lines.append(f"── {metric_name} ──")
        lines.append(f"{'Method':<10} {'Mean':>8} {'Std':>8} {'Med':>8}")
        lines.append("─" * 38)
        for m_idx, method in enumerate(methods):
            vals = np.array(all_metrics[m_idx][metric_name])
            lines.append(
                f"{method.name:<10} {np.mean(vals):+8.4f} {np.std(vals):8.4f} "
                f"{np.median(vals):+8.4f}"
            )
        lines.append("")

    # Global noise stats (reuse compute_noise_jacobian_metrics)
    lines.append("── Global Noise Stats ──")
    lines.append(f"{'Method':<10} {'L∞':>9} {'L2(rms)':>9} {'TV':>9} {'∇mean':>9}")
    lines.append("─" * 42)
    for m_idx, method in enumerate(methods):
        noise_5d = noises[m_idx].unsqueeze(0)  # [1, C, D, H, W]
        jm = compute_noise_jacobian_metrics(noise_5d)
        lines.append(
            f"{method.name:<10} {jm['noise_linf']:9.6f} {jm['noise_l2']:9.6f} "
            f"{jm['noise_tv']:9.6f} {jm['noise_grad_mean']:9.6f}"
        )

    stats_text = "\n".join(lines)
    ax_stats.text(
        0.02, 0.98, stats_text, fontsize=8.5, family="monospace",
        verticalalignment="top", transform=ax_stats.transAxes,
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8),
    )

    fig.suptitle(
        f"Compare B: Inter-Slice Similarity — {' vs '.join(m.name for m in methods)}\n"
        f"Metrics: Pearson | SSIM | Cosine | L2 | TV  (channel: {MODALITY_NAMES[channel]})",
        fontsize=13, fontweight="bold", y=1.0,
    )

    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[Saved] Compare B: {output_path}")


# ======================== Compare F ======================== #

def compare_scheme_f(
    methods: List[MethodData],
    noises: List[torch.Tensor],
    preds_clean: List[torch.Tensor],
    preds_noisy: List[torch.Tensor],
    image: torch.Tensor,
    label: torch.Tensor,
    slice_idx: int,
    output_path: str,
    case_id: str = "",
    eps: float = 8/255,
    vis_channel: int = 0,
):
    """
    Comparison Scheme F: segmentation impact across methods.

    Layout (cols):
      Original | GT | Method1 Noise | M1 Pred(noisy) | M1 Diff | Method2 Noise | M2 Pred(noisy) | M2 Diff
    Row 0: single modality view
    Row 1: segmentation view
    """
    M = len(methods)
    image_np = image.numpy()
    label_np = label.numpy()
    C, D, H, W = image_np.shape

    label_slice = label_np[slice_idx]
    gt_rgb = label_to_rgb(label_slice)
    orig_slice = image_np[vis_channel, slice_idx]

    # 2 rows x (2 + 3*M) cols
    # Col 0: Original, Col 1: GT
    # Per method: Noise(BWR), Pred(Noisy), Pred Diff
    n_cols = 2 + 3 * M
    fig, axes = plt.subplots(2, n_cols, figsize=(3.5 * n_cols, 7.5), facecolor="white")
    fig.subplots_adjust(wspace=0.06, hspace=0.10)

    # --- Col 0: Original ---
    axes[0, 0].imshow(orig_slice, cmap="gray", vmin=0, vmax=1)
    axes[0, 0].set_title(f"Original\n({MODALITY_NAMES[vis_channel]})", fontsize=10,
                         fontweight="bold")
    axes[1, 0].imshow(orig_slice, cmap="gray", vmin=0, vmax=1)
    axes[1, 0].set_title("", fontsize=10)

    # --- Col 1: GT ---
    axes[0, 1].imshow(gt_rgb)
    axes[0, 1].set_title("Ground Truth", fontsize=10, fontweight="bold")
    # Clean pred from first method as reference
    pred_c0 = preds_clean[0].numpy()[slice_idx]
    axes[1, 1].imshow(label_to_rgb(pred_c0))
    axes[1, 1].set_title("Pred (Clean)", fontsize=10, fontweight="bold")

    # Dice strings for title
    dice_strs = []

    for m_idx, method in enumerate(methods):
        noise_np = noises[m_idx].numpy() if noises[m_idx] is not None else np.zeros_like(image_np)
        pred_c_np = preds_clean[m_idx].numpy()
        pred_n_np = preds_noisy[m_idx].numpy()

        col_base = 2 + m_idx * 3
        color = METHOD_COLORS[m_idx % len(METHOD_COLORS)]

        noise_slice = noise_np[vis_channel, slice_idx]
        pred_c_slice = pred_c_np[slice_idx]
        pred_n_slice = pred_n_np[slice_idx]

        # Row 0: Noise (diverging cmap), Noisy image, |δ|×10
        axes[0, col_base].imshow(
            noise_slice, cmap=NOISE_CMAP, vmin=NOISE_VMIN, vmax=NOISE_VMAX)
        axes[0, col_base].set_title(f"{method.name}\nNoise", fontsize=10,
                                    fontweight="bold", color=color)

        noisy_slice = np.clip(orig_slice + noise_slice, 0, 1)
        axes[0, col_base + 1].imshow(noisy_slice, cmap="gray", vmin=0, vmax=1)
        axes[0, col_base + 1].set_title("Noisy Image", fontsize=9)

        diff_abs = np.clip(np.abs(noise_slice) * 10, 0, 1)
        axes[0, col_base + 2].imshow(diff_abs, cmap="hot", vmin=0, vmax=1)
        axes[0, col_base + 2].set_title("|δ|×10", fontsize=9)

        # Row 1: Pred(Noisy), Pred Diff overlay, Dice
        axes[1, col_base].imshow(label_to_rgb(pred_n_slice))
        axes[1, col_base].set_title(f"Pred (Noisy)", fontsize=9)

        pred_diff = (pred_c_slice != pred_n_slice).astype(np.float32)
        axes[1, col_base + 1].imshow(pred_diff, cmap="hot", vmin=0, vmax=1)
        axes[1, col_base + 1].set_title("Pred Diff", fontsize=9)

        # Per-class Dice text
        d_clean = dice_score(pred_c_slice, label_slice)
        d_noisy = dice_score(pred_n_slice, label_slice)
        delta = d_noisy - d_clean
        n_changed = int(pred_diff.sum())
        pct_changed = 100.0 * n_changed / (H * W)

        info_text = (
            f"{method.name}\n"
            f"Dice clean: {d_clean:.4f}\n"
            f"Dice noisy: {d_noisy:.4f}\n"
            f"ΔDice: {delta:+.4f}\n"
            f"Changed: {n_changed} ({pct_changed:.1f}%)"
        )
        axes[1, col_base + 2].axis("off")
        axes[1, col_base + 2].text(
            0.05, 0.95, info_text, fontsize=9, family="monospace",
            verticalalignment="top", transform=axes[1, col_base + 2].transAxes,
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.85),
            color=color,
        )

        dice_strs.append(f"{method.name}: {d_clean:.3f}→{d_noisy:.3f} (Δ{delta:+.3f})")

    # Turn off all axes — remove black borders
    noise_axes_f = []
    for ax_row in axes:
        for ax in ax_row:
            ax.axis("off")

    # Collect noise axes for shared colorbar
    for m_idx in range(M):
        col_base = 2 + m_idx * 3
        noise_axes_f.append(axes[0, col_base])
    _add_noise_colorbar(fig, noise_axes_f, shrink=0.8, pad=0.03)

    fig.suptitle(
        f"Compare F: Segmentation Impact — {case_id} (z={slice_idx})\n"
        f"Dice: {' | '.join(dice_strs)}",
        fontsize=12, fontweight="bold", y=1.01,
    )

    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[Saved] Compare F: {output_path}")


# ======================== Compare U (Unified) ======================== #

def compare_unified(
    methods: List[MethodData],
    noises: List[torch.Tensor],
    pred_noisy: torch.Tensor,
    image: torch.Tensor,
    slice_z: int,
    output_path: str,
    case_id: str = "",
    eps: float = 8/255,
    vis_channel: int = 0,
):
    """
    Unified paper figure with three separated panels: (a) + (b) + (c).

    (a) Tight image grid (no borders), 2 rows (z, z+1):
        Cols: Original | Noise per method | Noisy(Ours)

    (b) Vertical Pearson correlation curve (with plot frame):
        Title at top, xlabel at bottom, y-axis = slice z (inverted).

    (c) Clean + Seg (no borders), 2 rows:
        Top = clean image at z, Bottom = victim seg on clean at z.
    """
    M = len(methods)
    image_np = image.numpy()
    pred_np = pred_noisy.numpy()
    C, D, H, W = image_np.shape

    z0 = slice_z
    z1 = min(slice_z + 1, D - 1)

    # (a) cols: Original + M noise + Noisy(Ours)
    n_a = 1 + M + 1

    # --- Figure sizing ---
    cell = 2.0
    cbar_w = 0.35  # width for vertical colorbar
    b_w = 1.6
    c_w = 1.2
    gap = 0.4  # gap between panels
    fig_w = cell * n_a + cbar_w + gap + b_w + gap + c_w
    fig_h = cell * 2 + 0.5

    fig = plt.figure(figsize=(fig_w, fig_h), facecolor="white")

    # Use three separate GridSpecs for proper gaps between panels
    left_edge = 0.0
    a_right = (cell * n_a) / fig_w
    cbar_left = a_right + 0.005
    cbar_right = (cell * n_a + cbar_w) / fig_w
    b_left = cbar_right + gap / fig_w
    b_right = b_left + b_w / fig_w
    c_left = b_right + gap / fig_w
    c_right = 1.0

    gs_a = gridspec.GridSpec(
        2, n_a, figure=fig,
        left=left_edge, right=a_right, bottom=0.02, top=0.88,
        wspace=0.005, hspace=0.005,
    )
    gs_b = gridspec.GridSpec(
        1, 1, figure=fig,
        left=b_left, right=b_right, bottom=0.08, top=0.88,
    )
    gs_c = gridspec.GridSpec(
        2, 1, figure=fig,
        left=c_left, right=c_right, bottom=0.02, top=0.88,
        wspace=0.005, hspace=0.005,
    )

    # ============ (a) Image grid — no borders, no text ============
    a_axes = {}
    for row in range(2):
        for col in range(n_a):
            ax = fig.add_subplot(gs_a[row, col])
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            a_axes[(row, col)] = ax

    ours_noise_np = noises[-1].numpy()

    for row, z in enumerate([z0, z1]):
        a_axes[(row, 0)].imshow(
            image_np[vis_channel, z], cmap="gray", vmin=0, vmax=1)

        for m_idx in range(M):
            col = 1 + m_idx
            noise_np_m = noises[m_idx].numpy()
            a_axes[(row, col)].imshow(
                noise_np_m[vis_channel, z], cmap=NOISE_CMAP,
                vmin=NOISE_VMIN, vmax=NOISE_VMAX)

        noisy_slice = np.clip(
            image_np[vis_channel, z] + ours_noise_np[vis_channel, z], 0, 1)
        a_axes[(row, 1 + M)].imshow(noisy_slice, cmap="gray", vmin=0, vmax=1)

    # (no column titles, row labels, or panel labels)

    # ============ (b) Pearson correlation — keep plot frame ============
    ax_b = fig.add_subplot(gs_b[0, 0])

    for m_idx in range(M):
        noise_ch = noises[m_idx][vis_channel].numpy()
        Dz = noise_ch.shape[0]
        corrs = []
        for zz in range(Dz - 1):
            s1 = noise_ch[zz].flatten()
            s2 = noise_ch[zz + 1].flatten()
            if np.std(s1) > 1e-8 and np.std(s2) > 1e-8:
                corrs.append(float(np.corrcoef(s1, s2)[0, 1]))
            else:
                corrs.append(0.0)

        color = METHOD_COLORS[m_idx % len(METHOD_COLORS)]
        z_indices = np.arange(len(corrs))
        ax_b.plot(corrs, z_indices, color=color, linewidth=1.0, alpha=0.85)

    ax_b.invert_yaxis()
    ax_b.axvline(x=0, color="gray", linestyle=":", linewidth=0.5)
    ax_b.set_xlim(-1.05, 1.05)

    ax_b.set_xticklabels([])
    ax_b.set_yticklabels([])
    ax_b.tick_params(length=0)
    ax_b.grid(True, alpha=0.2)

    ax_b.axhline(y=z0, color="black", linestyle="--", linewidth=0.5, alpha=0.4)
    ax_b.axhline(y=z1, color="black", linestyle="--", linewidth=0.5, alpha=0.4)

    # ============ (c) Clean + Seg — no borders, no text ============
    ax_c0 = fig.add_subplot(gs_c[0, 0])
    ax_c0.imshow(image_np[vis_channel, z0], cmap="gray", vmin=0, vmax=1)
    ax_c0.set_xticks([]); ax_c0.set_yticks([])
    for sp in ax_c0.spines.values():
        sp.set_visible(False)

    ax_c1 = fig.add_subplot(gs_c[1, 0])
    ax_c1.imshow(label_to_rgb(pred_np[z0]))
    ax_c1.set_xticks([]); ax_c1.set_yticks([])
    for sp in ax_c1.spines.values():
        sp.set_visible(False)

    # Vertical colorbar on the right of panel (a), dedicated axes
    cbar_ax = fig.add_axes([cbar_left, 0.08, 0.012, 0.75])
    norm = Normalize(vmin=NOISE_VMIN, vmax=NOISE_VMAX)
    sm = ScalarMappable(norm=norm, cmap=NOISE_CMAP)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_ticks([NOISE_VMIN, 0, NOISE_VMAX])
    cbar.set_ticklabels(["-4/255", "0", "4/255"])
    cbar.ax.tick_params(labelsize=6)

    plt.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white",
                pad_inches=0.03)
    plt.close(fig)
    print(f"[Saved] Unified: {output_path}")


# ======================== CLI ======================== #

class MethodAction(argparse.Action):
    """Custom argparse action: --method NAME NOISE_PATH MODEL_PATH"""
    def __call__(self, parser, namespace, values, option_string=None):
        methods = getattr(namespace, self.dest, None) or []
        if len(values) != 3:
            parser.error(
                f"--method requires 3 arguments: NAME NOISE_PATH MODEL_PATH, got {len(values)}"
            )
        methods.append({"name": values[0], "noise": values[1], "model": values[2]})
        setattr(namespace, self.dest, methods)


def main():
    parser = argparse.ArgumentParser(
        description="Multi-Method Noise Comparison (BraTS19)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--method", nargs=3, action=MethodAction, dest="methods",
        metavar=("NAME", "NOISE_PATH", "MODEL_PATH"),
        help="Method spec: name, noise manifest path, model checkpoint path. "
             "Repeat for each method.",
    )
    parser.add_argument("--output_dir", type=str, default="outputs/visualize_compare")
    parser.add_argument("--dataset_config", type=str, default="configs/dataset/brats19.yaml")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--sample_idx", type=int, default=0)
    parser.add_argument("--num_samples", type=int, default=3)
    parser.add_argument("--slice_idx", type=int, default=-1,
                        help="-1 = auto (max foreground)")
    parser.add_argument("--eps", type=float, default=8/255)
    parser.add_argument("--n_slices", type=int, default=5)
    parser.add_argument("--channel", type=int, default=0,
                        help="Modality for A/B (0=T1, 1=T1ce, 2=T2, 3=FLAIR)")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--schemes", type=str, default="a,b,f",
                        help="Comma-separated: a,b,f,u (u=unified)")

    args = parser.parse_args()

    if not args.methods or len(args.methods) < 2:
        parser.error("At least 2 --method entries are required for comparison.")

    schemes = set(s.strip().lower() for s in args.schemes.split(","))

    # Load config & dataset
    config = load_config(args.dataset_config)
    print(f"[Loading Dataset] split={args.split}")
    dataset = load_dataset(config, split=args.split)
    print(f"[Dataset] {len(dataset)} samples")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Build method objects & load models
    need_all_models = "f" in schemes
    method_objs: List[MethodData] = []
    for i, md in enumerate(args.methods):
        m = MethodData(name=md["name"], noise_path=md["noise"],
                       model_path=md["model"])
        if need_all_models:
            m.load_model(config, device)
        method_objs.append(m)
    # For scheme u, only need the last method's model (Ours) for segmentation
    if "u" in schemes and not need_all_models:
        method_objs[-1].load_model(config, device)

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

        print(f"\n{'='*60}")
        print(f"[Sample {idx}] {case_id}, shape={tuple(image.shape)}")

        key = extract_key(sample, idx, key_spec)
        image_np = image.numpy()
        label_np = label.numpy()

        # Auto-select slice
        if args.slice_idx < 0:
            center_z = auto_select_slice(label_np, D)
        else:
            center_z = min(max(0, args.slice_idx), D - 1)
        print(f"  Center slice: z={center_z}")

        # Load all noises and clamp to [-4/255, 4/255]
        noise_list: List[torch.Tensor] = []
        noise_np_list: List[np.ndarray] = []
        for m in method_objs:
            noise = m.get_noise(key)
            if noise is None:
                print(f"  [Warning] {m.name}: no noise, using zeros")
                noise = torch.zeros_like(image)
            linf_raw = noise.abs().max().item()
            noise = noise.clamp(NOISE_VMIN, NOISE_VMAX)
            noise_list.append(noise)
            noise_np_list.append(noise.numpy())
            print(f"  {m.name}: L∞(raw)={linf_raw:.6f}, "
                  f"L∞(clamped)={noise.abs().max():.6f}, "
                  f"L2(rms)={torch.sqrt(torch.mean(noise**2)):.6f}")

        prefix = f"{idx:04d}_{case_id}"

        # --- Compare A ---
        if "a" in schemes:
            path_a = os.path.join(args.output_dir, f"{prefix}_cmpA_multi_slice.png")
            compare_scheme_a(
                methods=method_objs,
                noises=noise_np_list,
                image_np=image_np,
                center_z=center_z,
                output_path=path_a,
                case_id=case_id,
                eps=args.eps,
                n_slices=args.n_slices,
                channel=args.channel,
            )

        # --- Compare B ---
        if "b" in schemes:
            path_b = os.path.join(args.output_dir, f"{prefix}_cmpB_similarity.png")
            compare_scheme_b(
                methods=method_objs,
                noises=noise_list,
                output_path=path_b,
                case_id=case_id,
                channel=args.channel,
            )

        # --- Compare F ---
        if "f" in schemes:
            preds_clean = []
            preds_noisy = []
            for m_idx, m in enumerate(method_objs):
                pc = run_inference(m.model, image, device)
                noisy_img = (image + noise_list[m_idx]).clamp(0, 1)
                pn = run_inference(m.model, noisy_img, device)
                preds_clean.append(pc)
                preds_noisy.append(pn)

            path_f = os.path.join(args.output_dir, f"{prefix}_cmpF_seg_impact.png")
            compare_scheme_f(
                methods=method_objs,
                noises=noise_list,
                preds_clean=preds_clean,
                preds_noisy=preds_noisy,
                image=image,
                label=label,
                slice_idx=center_z,
                output_path=path_f,
                case_id=case_id,
                eps=args.eps,
                vis_channel=args.channel,
            )

        # --- Compare U (unified) ---
        if "u" in schemes:
            # Clean image through victim model -> should segment poorly
            ours_m = method_objs[-1]
            pred_ours = run_inference(ours_m.model, image, device)

            path_u = os.path.join(args.output_dir, f"{prefix}_unified.png")
            compare_unified(
                methods=method_objs,
                noises=noise_list,
                pred_noisy=pred_ours,
                image=image,
                slice_z=center_z,
                output_path=path_u,
                case_id=case_id,
                eps=args.eps,
                vis_channel=args.channel,
            )

    print(f"\n[Done] Output: {args.output_dir}")


if __name__ == "__main__":
    main()