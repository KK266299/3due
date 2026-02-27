#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
计算分割模型在干净测试集上的 Dice / IoU / HD95 / ASD 指标。

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

import h5py
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

# 添加项目根目录到 path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import src.models      # noqa: F401  — 触发 @register_model 装饰器
import src.datasets    # noqa: F401  — 触发 @register_dataset_builder 装饰器

from src.registry import get_dataset_builder, get_model
from src.utils.config import get_config

# ─── MONAI metrics ───────────────────────────────────────────────
from monai.metrics import DiceMetric, HausdorffDistanceMetric, MeanIoU, SurfaceDistanceMetric


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
    在测试集上逐样本推理，计算 Dice / IoU / HD95 / ASD。

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
    iou_metric = MeanIoU(
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
    asd_metric = SurfaceDistanceMetric(
        include_background=True,
        symmetric=True,
        reduction="none",
        get_not_nans=True,
    )

    per_sample_results: List[Dict[str, object]] = []

    # 追踪每个样本各区域是否非空，用于修正 HD95/ASD 的 NaN 惩罚
    gt_nonempty_list: List[torch.Tensor] = []
    pred_nonempty_list: List[torch.Tensor] = []
    volume_diags: List[float] = []

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

        # 从 h5 读取 spacing，计算有效体素间距 (mm)
        spacing = None
        h5_path = sample.get("h5_path", None)
        if h5_path is not None:
            with h5py.File(h5_path, "r") as f:
                orig_sp = f.attrs.get("orig_spacing", None)
                orig_sh = f.attrs.get("orig_shape", None)
                final_sh = f.attrs.get("final_shape", f.attrs.get("target_shape", None))
            if orig_sp is not None and orig_sh is not None and final_sh is not None:
                orig_sp = np.array(orig_sp, dtype=np.float64)
                orig_sh = np.array(orig_sh, dtype=np.float64)
                final_sh = np.array(final_sh, dtype=np.float64)
                # h5 存储顺序 (H, W, D)，tensor 空间顺序 (D, H, W)
                eff_sp = orig_sp * orig_sh / final_sh          # (H, W, D)
                spacing = torch.tensor(
                    [eff_sp[2], eff_sp[0], eff_sp[1]],         # -> (D, H, W)
                    dtype=torch.float32,
                )

        # 追踪各区域是否非空（用于后续 NaN 惩罚）
        gt_ne = y_reg[0].flatten(1).any(dim=1)     # [R] bool
        pred_ne = p_reg[0].flatten(1).any(dim=1)   # [R] bool
        gt_nonempty_list.append(gt_ne)
        pred_nonempty_list.append(pred_ne)

        # 体积对角线 (mm)，用作 NaN 惩罚距离
        D, H, W = y_reg.shape[2:]
        if spacing is not None:
            sp_d, sp_h, sp_w = spacing[0].item(), spacing[1].item(), spacing[2].item()
            volume_diags.append(float(((D*sp_d)**2 + (H*sp_h)**2 + (W*sp_w)**2) ** 0.5))
        else:
            volume_diags.append(float((D**2 + H**2 + W**2) ** 0.5))

        # 累积 metrics
        dice_metric(y_pred=p_reg, y=y_reg)
        iou_metric(y_pred=p_reg, y=y_reg)
        hd95_metric(y_pred=p_reg, y=y_reg, spacing=spacing)
        asd_metric(y_pred=p_reg, y=y_reg, spacing=spacing)

        # 单样本结果（汇总后再填数值）
        row = {"case_id": case_id, "index": idx}
        per_sample_results.append(row)

    # ---- Aggregate ----
    dice_vals, dice_nans = dice_metric.aggregate()
    dice_vals = dice_vals.view(-1, n_regions)
    dice_nans = dice_nans.view(-1, n_regions)

    iou_vals, iou_nans = iou_metric.aggregate()
    iou_vals = iou_vals.view(-1, n_regions)
    iou_nans = iou_nans.view(-1, n_regions)

    hd95_vals, hd95_nans = hd95_metric.aggregate()
    hd95_vals = hd95_vals.view(-1, n_regions)
    hd95_nans = hd95_nans.view(-1, n_regions)

    asd_vals, asd_nans = asd_metric.aggregate()
    asd_vals = asd_vals.view(-1, n_regions)
    asd_nans = asd_nans.view(-1, n_regions)

    # ---- 修正 NaN：单侧为空时用体积对角线 (mm) 作为惩罚值 ----
    # 当 GT 有某区域但 Pred 完全为空（或反之）时，MONAI 对 HD95/ASD 返回 NaN，
    # 导致这些"完全漏检"样本被排除在平均之外，使 HD95 均值虚低。
    # 修正：用该样本体积的对角线长度作为惩罚距离。
    gt_nonempty = torch.stack(gt_nonempty_list)      # [N, R]
    pred_nonempty = torch.stack(pred_nonempty_list)   # [N, R]
    one_sided_empty = gt_nonempty ^ pred_nonempty     # 恰好一侧非空

    for i in range(hd95_vals.shape[0]):
        for r in range(n_regions):
            if one_sided_empty[i, r]:
                penalty = volume_diags[i]
                if hd95_nans[i, r] == 0:   # 原本是 NaN
                    hd95_vals[i, r] = penalty
                    hd95_nans[i, r] = 1
                if asd_nans[i, r] == 0:
                    asd_vals[i, r] = penalty
                    asd_nans[i, r] = 1

    # 填充 per-sample 数据
    for i, row in enumerate(per_sample_results):
        for r, rn in enumerate(region_names):
            row[f"dice_{rn}"] = float(dice_vals[i, r].item()) if dice_nans[i, r] > 0 else float("nan")
            row[f"iou_{rn}"]  = float(iou_vals[i, r].item())  if iou_nans[i, r]  > 0 else float("nan")
            row[f"hd95_{rn}"] = float(hd95_vals[i, r].item()) if hd95_nans[i, r] > 0 else float("nan")
            row[f"asd_{rn}"]  = float(asd_vals[i, r].item())  if asd_nans[i, r]  > 0 else float("nan")

    # ---- 汇总指标 ----
    summary: Dict[str, float] = {}

    # 辅助函数：对距离类指标（HD95/ASD）安全取均值，过滤 inf
    def _safe_dist_mean(vals: torch.Tensor, nans: torch.Tensor, r: int) -> float:
        valid_mask = nans[:, r] > 0
        if not valid_mask.any():
            return float("inf")
        v = vals[valid_mask, r]
        finite = torch.isfinite(v)
        if not finite.any():
            return float("inf")
        return float(v[finite].mean().item())

    for r, rn in enumerate(region_names):
        # Dice
        valid_mask = dice_nans[:, r] > 0
        summary[f"dice_{rn}"] = float(dice_vals[valid_mask, r].mean().item()) if valid_mask.any() else 0.0

        # IoU
        valid_mask = iou_nans[:, r] > 0
        summary[f"iou_{rn}"] = float(iou_vals[valid_mask, r].mean().item()) if valid_mask.any() else 0.0

        # HD95
        summary[f"hd95_{rn}"] = _safe_dist_mean(hd95_vals, hd95_nans, r)

        # ASD
        summary[f"asd_{rn}"] = _safe_dist_mean(asd_vals, asd_nans, r)

    # 平均值
    valid_dice = [v for k, v in summary.items() if k.startswith("dice_") and v > 0]
    summary["dice_avg"] = float(np.mean(valid_dice)) if valid_dice else 0.0

    valid_iou = [v for k, v in summary.items() if k.startswith("iou_") and v > 0]
    summary["iou_avg"] = float(np.mean(valid_iou)) if valid_iou else 0.0

    valid_hd = [v for k, v in summary.items() if k.startswith("hd95_") and np.isfinite(v)]
    summary["hd95_avg"] = float(np.mean(valid_hd)) if valid_hd else float("inf")

    valid_asd = [v for k, v in summary.items() if k.startswith("asd_") and np.isfinite(v)]
    summary["asd_avg"] = float(np.mean(valid_asd)) if valid_asd else float("inf")

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

    print("\n" + "=" * 70)
    print(f"  Results: {args.dataset} ({args.split} split)")
    print("=" * 70)

    def _fmt(v: float) -> str:
        return f"{v:.4f}" if np.isfinite(v) else "inf"

    print(f"\n{'Region':<15} {'Dice':>10} {'IoU':>10} {'HD95(mm)':>10} {'ASD(mm)':>10}")
    print("-" * 57)
    for rn in region_names:
        d  = summary.get(f"dice_{rn}", 0.0)
        io = summary.get(f"iou_{rn}", 0.0)
        h  = summary.get(f"hd95_{rn}", float("inf"))
        a  = summary.get(f"asd_{rn}", float("inf"))
        print(f"{rn:<15} {_fmt(d):>10} {_fmt(io):>10} {_fmt(h):>10} {_fmt(a):>10}")

    print("-" * 57)
    print(f"{'Average':<15} {_fmt(summary.get('dice_avg', 0.0)):>10} "
          f"{_fmt(summary.get('iou_avg', 0.0)):>10} "
          f"{_fmt(summary.get('hd95_avg', float('inf'))):>10} "
          f"{_fmt(summary.get('asd_avg', float('inf'))):>10}")
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
