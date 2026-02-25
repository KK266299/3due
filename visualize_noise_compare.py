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

SEG_COLORS = np.array([
    [0, 0, 0],       # 0: Background
    [255, 0, 0],     # 1: NCR/NET
    [0, 255, 0],     # 2: ED
    [255, 255, 0],   # 3: ET
], dtype=np.uint8)

# Distinct colors for methods
METHOD_COLORS = [
    "#2196F3",  # blue  (method 1)
    "#F44336",  # red   (method 2)
    "#4CAF50",  # green (method 3)
    "#FF9800",  # orange(method 4)
]

METHOD_CMAPS = ["RdBu_r", "PiYG_r", "PuOr_r", "BrBG_r"]


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


def colorize_noise(noise_2d: np.ndarray, eps: float = 8/255) -> np.ndarray:
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


def compute_inter_slice_correlation(noise_np: np.ndarray, channel: int = 0) -> List[float]:
    noise_3d = noise_np[channel]
    D = noise_3d.shape[0]
    correlations = []
    for z in range(D - 1):
        s1 = noise_3d[z].flatten()
        s2 = noise_3d[z + 1].flatten()
        if np.std(s1) > 1e-8 and np.std(s2) > 1e-8:
            corr = float(np.corrcoef(s1, s2)[0, 1])
        else:
            corr = 0.0
        correlations.append(corr)
    return correlations


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

    for m_idx, (method, noise_np) in enumerate(zip(methods, noises)):
        row_noise = 1 + m_idx * 2
        row_diff = 2 + m_idx * 2

        # Noise row
        for i, z in enumerate(slices):
            noise_rgb = colorize_noise(noise_np[channel, z], eps)
            axes[row_noise, i].imshow(noise_rgb)
            axes[row_noise, i].axis("off")
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
    noises: List[np.ndarray],
    output_path: str,
    case_id: str = "",
    channel: int = 0,
):
    """
    Comparison Scheme B: overlay inter-slice correlation curves.

    Left:  Correlation curves (all methods overlaid)
    Right: Statistics table + box plot
    """
    M = len(methods)
    all_corrs = []
    for noise_np in noises:
        all_corrs.append(compute_inter_slice_correlation(noise_np, channel))

    fig = plt.figure(figsize=(18, 7), facecolor="white")
    gs = gridspec.GridSpec(1, 3, figure=fig, width_ratios=[3, 1, 1.5], wspace=0.25)

    # --- Left: Correlation curves ---
    ax_corr = fig.add_subplot(gs[0, 0])
    for m_idx, (method, corrs) in enumerate(zip(methods, all_corrs)):
        color = METHOD_COLORS[m_idx % len(METHOD_COLORS)]
        x = np.arange(len(corrs))
        ax_corr.plot(x, corrs, color=color, linewidth=1.2, alpha=0.85,
                     label=f"{method.name} (μ={np.mean(corrs):.3f})")
        # Mean line
        ax_corr.axhline(y=np.mean(corrs), color=color, linestyle="--",
                        linewidth=1.0, alpha=0.5)

    ax_corr.axhline(y=0, color="gray", linestyle=":", linewidth=0.8)
    ax_corr.set_xlabel("Slice Index (z)", fontsize=11)
    ax_corr.set_ylabel("Pearson Correlation", fontsize=11)
    ax_corr.set_title(
        f"Inter-Slice Correlation ({MODALITY_NAMES[channel]}) — {case_id}",
        fontsize=12, fontweight="bold",
    )
    ax_corr.set_ylim(-1.05, 1.05)
    ax_corr.legend(fontsize=9, loc="upper right")
    ax_corr.grid(True, alpha=0.3)

    # --- Middle: Box plot ---
    ax_box = fig.add_subplot(gs[0, 1])
    bp = ax_box.boxplot(
        all_corrs,
        labels=[m.name for m in methods],
        patch_artist=True,
        widths=0.6,
    )
    for m_idx, patch in enumerate(bp["boxes"]):
        color = METHOD_COLORS[m_idx % len(METHOD_COLORS)]
        patch.set_facecolor(color)
        patch.set_alpha(0.35)
    ax_box.set_ylabel("Correlation", fontsize=10)
    ax_box.set_title("Distribution", fontsize=11, fontweight="bold")
    ax_box.axhline(y=0, color="gray", linestyle=":", linewidth=0.8)
    ax_box.grid(True, alpha=0.3, axis="y")

    # --- Right: Stats table ---
    ax_stats = fig.add_subplot(gs[0, 2])
    ax_stats.axis("off")

    header = f"{'Method':<12} {'Mean':>7} {'Std':>7} {'Min':>7} {'Max':>7}\n"
    sep = "─" * 48 + "\n"
    rows = header + sep
    for m_idx, (method, corrs) in enumerate(zip(methods, all_corrs)):
        c = np.array(corrs)
        rows += (
            f"{method.name:<12} {np.mean(c):+.4f} {np.std(c):.4f} "
            f"{np.min(c):+.4f} {np.max(c):+.4f}\n"
        )

    # Add noise stats
    rows += sep + f"{'Method':<12} {'L∞':>10} {'L2(rms)':>10}\n" + sep
    for m_idx, (method, noise_np) in enumerate(zip(methods, noises)):
        linf = np.abs(noise_np).max()
        l2 = np.sqrt(np.mean(noise_np ** 2))
        rows += f"{method.name:<12} {linf:10.6f} {l2:10.6f}\n"

    ax_stats.text(
        0.02, 0.95, rows, fontsize=9.5, family="monospace",
        verticalalignment="top", transform=ax_stats.transAxes,
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8),
    )

    fig.suptitle(
        f"Compare B: Correlation Analysis — {' vs '.join(m.name for m in methods)}",
        fontsize=13, fontweight="bold", y=1.02,
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

        # Row 0: Noise BWR, Noisy image, |δ|×10
        noise_rgb = colorize_noise(noise_slice, eps)
        axes[0, col_base].imshow(noise_rgb)
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

    # Turn off all axes
    for ax_row in axes:
        for ax in ax_row:
            ax.set_xticks([])
            ax.set_yticks([])

    fig.suptitle(
        f"Compare F: Segmentation Impact — {case_id} (z={slice_idx})\n"
        f"Dice: {' | '.join(dice_strs)}",
        fontsize=12, fontweight="bold", y=1.01,
    )

    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[Saved] Compare F: {output_path}")


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
                        help="Comma-separated: a,b,f")

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
    method_objs: List[MethodData] = []
    for md in args.methods:
        m = MethodData(name=md["name"], noise_path=md["noise"],
                       model_path=md["model"])
        if "f" in schemes:
            m.load_model(config, device)
        method_objs.append(m)

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

        # Load all noises
        noise_list: List[torch.Tensor] = []
        noise_np_list: List[np.ndarray] = []
        for m in method_objs:
            noise = m.get_noise(key)
            if noise is None:
                print(f"  [Warning] {m.name}: no noise, using zeros")
                noise = torch.zeros_like(image)
            noise_list.append(noise)
            noise_np_list.append(noise.numpy())
            print(f"  {m.name}: L∞={noise.abs().max():.6f}, "
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
            path_b = os.path.join(args.output_dir, f"{prefix}_cmpB_correlation.png")
            compare_scheme_b(
                methods=method_objs,
                noises=noise_np_list,
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

    print(f"\n[Done] Output: {args.output_dir}")


if __name__ == "__main__":
    main()
