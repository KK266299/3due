#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
计算分割模型在干净测试集上的 HD95 和 Dice 指标。

支持的数据集：
  - brats19_seg: BraTS19 脑肿瘤分割 (4ch, 4class, regions: ET/TC/WT)
  - flare21_seg: FLARE21 腹部器官分割 (1ch, 5class, regions: Liver/Kidney/Spleen/Pancreas)
  - kits19_seg:  KiTS19 肾脏肿瘤分割 (1ch, 3class, regions: Kidney/Tumor)

使用示例：
  python cal_metrics.py \
    --dataset brats19_seg \
    --model_path /path/to/best_model.pth \
    --device cuda:0

  python cal_metrics.py \
    --dataset flare21_seg \
    --model_path /path/to/best_model.pth \
    --output_csv results/flare21_metrics.csv
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

# 添加项目根目录到 path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.registry import get_dataset_builder, get_model
from src.utils.config import get_config

# ─── MONAI metrics ───────────────────────────────────────────────
from monai.metrics import DiceMetric, HausdorffDistanceMetric


# ======================================================================
#  数据集配置表
# ======================================================================

DATASET_CONFIGS: Dict[str, dict] = {
    "brats19_seg": {
        "dataset_yaml": "brats19",          # configs/dataset/ 下的文件名
        "in_channels": 4,
        "num_classes": 4,
        "region_names": ["ET", "TC", "WT"],
        "class_indices": {"bg": 0, "ncr": 1, "edema": 2, "enh": 3},
    },
    "flare21_seg": {
        "dataset_yaml": "flare21",
        "in_channels": 1,
        "num_classes": 5,
        "region_names": ["Liver", "Kidney", "Spleen", "Pancreas"],
        "class_indices": {"bg": 0, "liver": 1, "kidney": 2, "spleen": 3, "pancreas": 4},
    },
    "kits19_seg": {
        "dataset_yaml": "kits19",
        "in_channels": 1,
        "num_classes": 3,
        "region_names": ["Kidney", "Tumor"],
        "class_indices": {"bg": 0, "kidney": 1, "tumor": 2},
    },
}


# ======================================================================
#  Region mask 构建（与 evaluation 策略一致）
# ======================================================================

def build_region_masks(y_id: torch.Tensor, dataset: str) -> torch.Tensor:
    """
    将 label id map 转换为 region-based binary masks。

    Args:
        y_id: [B, D, H, W] LongTensor
        dataset: 数据集名称

    Returns:
        y_reg: [B, R, D, H, W] float32, R = 区域数量
    """
    ci = DATASET_CONFIGS[dataset]["class_indices"]

    if dataset == "brats19_seg":
        enh = ci["enh"]
        ncr = ci["ncr"]
        bg = ci["bg"]
        y_et = y_id.eq(enh)                        # ET
        y_tc = y_id.eq(ncr) | y_id.eq(enh)         # TC
        y_wt = y_id.ne(bg)                          # WT
        return torch.stack([y_et.float(), y_tc.float(), y_wt.float()], dim=1)

    elif dataset == "flare21_seg":
        return torch.stack([
            (y_id == ci["liver"]).float(),
            (y_id == ci["kidney"]).float(),
            (y_id == ci["spleen"]).float(),
            (y_id == ci["pancreas"]).float(),
        ], dim=1)

    elif dataset == "kits19_seg":
        kidney = ci["kidney"]
        tumor = ci["tumor"]
        y_kidney = (y_id == kidney) | (y_id == tumor)   # kidney 包含 tumor
        y_tumor = (y_id == tumor)
        return torch.stack([y_kidney.float(), y_tumor.float()], dim=1)

    else:
        raise ValueError(f"Unsupported dataset: {dataset}")


# ======================================================================
#  配置 / 模型 / 数据集加载
# ======================================================================

def build_config(dataset: str) -> OmegaConf:
    """根据数据集名称构建配置。"""
    ds_info = DATASET_CONFIGS[dataset]
    ds_yaml_name = ds_info["dataset_yaml"]

    # 加载数据集 YAML
    ds_yaml_path = Path(__file__).resolve().parent / "configs" / "dataset" / f"{ds_yaml_name}.yaml"
    if not ds_yaml_path.exists():
        raise FileNotFoundError(f"Dataset config not found: {ds_yaml_path}")
    ds_cfg = OmegaConf.load(str(ds_yaml_path))

    config = OmegaConf.create({
        "model": {
            "name": "unet",
            "in_channels": ds_info["in_channels"],
            "num_classes": ds_info["num_classes"],
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
        "dataset": ds_cfg,
    })
    return config


def load_model(model_path: str, config: OmegaConf, device: torch.device) -> torch.nn.Module:
    """加载模型 checkpoint。"""
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


def load_dataset(config: OmegaConf, dataset: str, split: str = "test"):
    """加载数据集。"""
    ds_name = config.dataset.name
    # brats19 -> brats19_seg
    if ds_name == "brats19":
        ds_name = "brats19_seg"
    elif ds_name == "flare21":
        ds_name = "flare21_seg"
    elif ds_name == "kits19":
        ds_name = "kits19_seg"

    builder_cls = get_dataset_builder(ds_name)
    builder = builder_cls(config)
    return builder.get_dataset(split=split)


# ======================================================================
#  主逻辑
# ======================================================================

def compute_metrics(
    model: torch.nn.Module,
    dataset,
    dataset_name: str,
    device: torch.device,
) -> Tuple[Dict[str, float], List[Dict[str, object]]]:
    """
    在测试集上逐样本推理，计算 HD95 和 Dice。

    Returns:
        summary: 汇总指标 dict
        per_sample: 每个样本的指标列表
    """
    ds_info = DATASET_CONFIGS[dataset_name]
    region_names = ds_info["region_names"]
    n_regions = len(region_names)

    # MONAI metrics（per-sample, per-region）
    dice_metric = DiceMetric(
        include_background=True,
        reduction="none",
        get_not_nans=True,
    )
    hd95_metric = HausdorffDistanceMetric(
        include_background=True,
        percentile=95,
        reduction="none",
        get_not_nans=True,
    )

    per_sample_results: List[Dict[str, object]] = []

    pbar = tqdm(range(len(dataset)), desc=f"Inference ({dataset_name})")
    for idx in pbar:
        sample = dataset[idx]
        image = sample["image"]     # [C, D, H, W]
        label = sample["label"]     # [D, H, W]
        case_id = sample.get("case_id", f"sample_{idx}")

        # 推理
        with torch.no_grad():
            x = image.unsqueeze(0).to(device)          # [1, C, D, H, W]
            logits = model(x)                           # [1, num_classes, D, H, W]
            pred_id = logits.argmax(dim=1)              # [1, D, H, W]

        # 构建 region masks
        y_id = label.unsqueeze(0).long()                # [1, D, H, W]
        y_reg = build_region_masks(y_id, dataset_name)  # [1, R, D, H, W]
        p_reg = build_region_masks(pred_id.cpu(), dataset_name)  # [1, R, D, H, W]

        # 累积 metrics
        dice_metric(y_pred=p_reg, y=y_reg)
        hd95_metric(y_pred=p_reg, y=y_reg)

        # 单样本结果
        row = {"case_id": case_id, "index": idx}
        # 取最后累积的一条（index = idx）
        # MONAI 在 __call__ 时会 append，aggregate 时返回所有累积的结果
        # 这里暂时存 case_id，汇总后再填
        per_sample_results.append(row)

    # ---- Aggregate Dice ----
    dice_vals, dice_nans = dice_metric.aggregate()
    dice_vals = dice_vals.view(-1, n_regions)
    dice_nans = dice_nans.view(-1, n_regions)

    # ---- Aggregate HD95 ----
    hd95_vals, hd95_nans = hd95_metric.aggregate()
    hd95_vals = hd95_vals.view(-1, n_regions)
    hd95_nans = hd95_nans.view(-1, n_regions)

    # 填充 per-sample 数据
    for i, row in enumerate(per_sample_results):
        for r, rn in enumerate(region_names):
            d_val = float(dice_vals[i, r].item()) if dice_nans[i, r] > 0 else float("nan")
            h_val = float(hd95_vals[i, r].item()) if hd95_nans[i, r] > 0 else float("nan")
            row[f"dice_{rn}"] = d_val
            row[f"hd95_{rn}"] = h_val

    # ---- 汇总指标 ----
    summary: Dict[str, float] = {}

    for r, rn in enumerate(region_names):
        # Dice
        valid_mask = dice_nans[:, r] > 0
        if valid_mask.any():
            summary[f"dice_{rn}"] = float(dice_vals[valid_mask, r].mean().item())
        else:
            summary[f"dice_{rn}"] = 0.0

        # HD95
        hd_valid_mask = hd95_nans[:, r] > 0
        if hd_valid_mask.any():
            hd_values = hd95_vals[hd_valid_mask, r]
            # 过滤 inf 值（当 pred 或 gt 为空时 MONAI 可能返回 inf）
            finite_mask = torch.isfinite(hd_values)
            if finite_mask.any():
                summary[f"hd95_{rn}"] = float(hd_values[finite_mask].mean().item())
            else:
                summary[f"hd95_{rn}"] = float("inf")
        else:
            summary[f"hd95_{rn}"] = float("inf")

    # 平均 Dice / HD95
    valid_dice = [v for k, v in summary.items() if k.startswith("dice_") and v > 0]
    summary["dice_avg"] = float(np.mean(valid_dice)) if valid_dice else 0.0

    valid_hd = [v for k, v in summary.items()
                if k.startswith("hd95_") and np.isfinite(v)]
    summary["hd95_avg"] = float(np.mean(valid_hd)) if valid_hd else float("inf")

    return summary, per_sample_results


def main():
    parser = argparse.ArgumentParser(
        description="计算分割模型在干净测试集上的 HD95 和 Dice 指标"
    )
    parser.add_argument(
        "--dataset", type=str, required=True,
        choices=list(DATASET_CONFIGS.keys()),
        help="数据集名称 (brats19_seg / flare21_seg / kits19_seg)",
    )
    parser.add_argument(
        "--model_path", type=str, required=True,
        help="模型 checkpoint 路径 (best_model.pth)",
    )
    parser.add_argument(
        "--output_csv", type=str, default=None,
        help="输出 CSV 文件路径（默认自动生成）",
    )
    parser.add_argument(
        "--device", type=str, default="cuda:0",
        help="计算设备 (default: cuda:0)",
    )
    parser.add_argument(
        "--split", type=str, default="test",
        help="数据集 split (default: test)",
    )

    args = parser.parse_args()

    # ---- 设备 ----
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")

    # ---- 配置 ----
    config = build_config(args.dataset)
    print(f"[Dataset] {args.dataset}")
    print(f"[Model Path] {args.model_path}")

    # ---- 加载模型 ----
    model = load_model(args.model_path, config, device)
    print("[Model] Loaded successfully")

    # ---- 加载数据集 ----
    test_dataset = load_dataset(config, args.dataset, split=args.split)
    print(f"[Dataset] {args.split} split: {len(test_dataset)} samples")

    # ---- 计算指标 ----
    summary, per_sample = compute_metrics(model, test_dataset, args.dataset, device)

    # ---- 打印汇总 ----
    ds_info = DATASET_CONFIGS[args.dataset]
    region_names = ds_info["region_names"]

    print("\n" + "=" * 60)
    print(f"  Results: {args.dataset} ({args.split} split)")
    print("=" * 60)

    print(f"\n{'Region':<15} {'Dice':>10} {'HD95':>10}")
    print("-" * 37)
    for rn in region_names:
        d = summary.get(f"dice_{rn}", 0.0)
        h = summary.get(f"hd95_{rn}", float("inf"))
        h_str = f"{h:.4f}" if np.isfinite(h) else "inf"
        print(f"{rn:<15} {d:>10.4f} {h_str:>10}")

    print("-" * 37)
    avg_d = summary.get("dice_avg", 0.0)
    avg_h = summary.get("hd95_avg", float("inf"))
    avg_h_str = f"{avg_h:.4f}" if np.isfinite(avg_h) else "inf"
    print(f"{'Average':<15} {avg_d:>10.4f} {avg_h_str:>10}")
    print()

    # ---- 保存 CSV ----
    if args.output_csv is None:
        # 自动生成输出路径
        model_dir = str(Path(args.model_path).parent)
        output_csv = os.path.join(model_dir, f"metrics_{args.dataset}_{args.split}.csv")
    else:
        output_csv = args.output_csv

    os.makedirs(os.path.dirname(output_csv) if os.path.dirname(output_csv) else ".", exist_ok=True)

    # 写 per-sample CSV
    if per_sample:
        fieldnames = list(per_sample[0].keys())
        with open(output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in per_sample:
                writer.writerow(row)
        print(f"[Saved] Per-sample results: {output_csv}")

    # 写 summary CSV
    summary_csv = output_csv.replace(".csv", "_summary.csv")
    with open(summary_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)
    print(f"[Saved] Summary: {summary_csv}")

    print("[Done]")


if __name__ == "__main__":
    main()
