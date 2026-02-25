#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
汇总所有方法在同一数据集上的 Dice / HD95 结果。

三种用法：

1) 指定汇总目录，自动递归查找所有 _summary.csv：
   python collect_results.py --dir /path/to/outputs/flare21_seg

2) 自动从 run_cal_metrics.sh 中已跑完的 summary CSV 收集：
   python collect_results.py --dataset brats19_seg

3) 手动指定多个 summary CSV：
   python collect_results.py \
       --csv path/to/clean_summary.csv \
       --csv path/to/victim_summary.csv \
       --output results/brats19_compare.csv
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np


# ======================================================================
#  模型路径 → 实验名
# ======================================================================

# 与 run_cal_metrics.sh 中的 MODEL_LIST 保持同步
_BRATS19_MODELS = [
    "/home/dengzhipeng/data/project/3d_ue/outputs/brats19_seg/clean/20251220_144653/checkpoints/checkpoints/best_model.pth",
    "/home/dengzhipeng/data/project/3d_ue/outputs/brats19_seg/victim_random_noise_gaussian/20251228_082846/checkpoints/checkpoints/best_model.pth",
    "/home/dengzhipeng/data/project/3d_ue/outputs/brats19_seg/victim_minmin_4_255/20251220_061401/checkpoints/checkpoints/best_model.pth",
    "/home/dengzhipeng/data/project/3d_ue/outputs/brats19_seg/victim_sep/20251222_124616/checkpoints/checkpoints/best_model.pth",
    "/home/dengzhipeng/data/project/3d_ue/outputs/brats19_seg/victim_nofreq_learnable_zdiv02_logits005/20260207_135201/checkpoints/checkpoints/best_model.pth",
]

DATASET_MODELS = {
    "brats19_seg": _BRATS19_MODELS,
    # "flare21_seg": [...],
    # "kits19_seg":  [...],
}

DATASET_REGIONS = {
    "brats19_seg": ["ET", "TC", "WT"],
    "flare21_seg": ["Liver", "Kidney", "Spleen", "Pancreas"],
    "kits19_seg":  ["Kidney", "Tumor"],
}

# 只保留 Dice 和 HD95
METRICS = ["dice", "hd95"]


def extract_method_name(model_path: str) -> str:
    """
    从模型路径提取实验名。

    例如:
      .../brats19_seg/clean/2025.../checkpoints/checkpoints/best_model.pth
      → "clean"

      .../brats19_seg/victim_minmin_4_255/2025.../...
      → "victim_minmin_4_255"
    """
    parts = Path(model_path).parts
    # 路径结构: .../<dataset>/<method>/<timestamp>/checkpoints/checkpoints/best_model.pth
    # method 在 dataset 后面
    for i, p in enumerate(parts):
        if p.endswith("_seg") and i + 1 < len(parts):
            return parts[i + 1]
    # fallback: 用父目录中的非日期部分
    for p in reversed(parts):
        if p not in ("checkpoints", "best_model.pth") and not re.match(r"^\d{8}_\d{6}$", p):
            return p
    return "unknown"


def extract_method_from_csv_path(csv_path: str) -> str:
    """
    从 summary CSV 路径提取实验名。

    向上遍历目录结构，找到类似实验名的部分。
    路径格式示例:
      .../flare21_seg/clean/20260101_120000/checkpoints/.../metrics_..._summary.csv
      → "clean"
      .../victim_minmin_unetpp/20260101_120000/.../metrics_..._summary.csv
      → "victim_minmin_unetpp"
    """
    parts = Path(csv_path).parts
    # 查找 _seg 后的部分作为 method name
    for i, p in enumerate(parts):
        if p.endswith("_seg") and i + 1 < len(parts):
            return parts[i + 1]

    # fallback: 找第一个非时间戳、非 checkpoints、非文件名的有意义部分
    skip = {"checkpoints", "results"}
    for p in reversed(parts[:-1]):  # 跳过文件名
        if p in skip:
            continue
        if re.match(r"^\d{8}_\d{6}$", p):
            continue
        if p.endswith("_seg"):
            continue
        if p.startswith("/") or p in ("home", "data", "project", "outputs"):
            continue
        return p
    return Path(csv_path).stem.replace("_summary", "")


def read_summary_csv(csv_path: str) -> Dict[str, float]:
    """读取 summary CSV 为 dict。"""
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            return {k: float(v) for k, v in row.items()}
    return {}


def find_summary_csv(model_path: str, dataset: str, split: str = "test") -> str:
    """根据模型路径找到对应的 summary CSV。"""
    model_dir = str(Path(model_path).parent)
    return os.path.join(model_dir, f"metrics_{dataset}_{split}_summary.csv")


def find_summary_csvs_in_dir(root_dir: str) -> List[str]:
    """递归查找目录下所有 _summary.csv 文件。"""
    results = []
    for dirpath, _, filenames in os.walk(root_dir):
        for fn in filenames:
            if fn.endswith("_summary.csv"):
                results.append(os.path.join(dirpath, fn))
    results.sort()
    return results


def fmt(v: float, is_pct: bool = False) -> str:
    """格式化数值。"""
    if not np.isfinite(v):
        return "inf"
    if is_pct:
        return f"{v * 100:.2f}"
    return f"{v:.4f}"


def main():
    parser = argparse.ArgumentParser(description="汇总多个方法的 Dice & HD95 指标")
    parser.add_argument("--dir", type=str, default=None,
                        help="汇总目录：递归查找其中所有 _summary.csv")
    parser.add_argument("--dataset", type=str, default=None,
                        help="数据集名称，自动查找已有的 summary CSV")
    parser.add_argument("--csv", type=str, action="append", default=None,
                        help="手动指定 summary CSV 路径（可多次使用）")
    parser.add_argument("--split", type=str, default="test",
                        help="数据集 split (default: test)")
    parser.add_argument("--output", type=str, default=None,
                        help="输出汇总 CSV 路径（默认自动生成）")
    parser.add_argument("--percent", action="store_true",
                        help="Dice 以百分比显示 (×100)")
    args = parser.parse_args()

    # ---- 收集 summary CSV ----
    entries: List[Dict] = []   # [{"method": str, "data": dict}, ...]

    if args.dir:
        # 目录模式：递归查找所有 _summary.csv
        root_dir = os.path.abspath(args.dir)
        if not os.path.isdir(root_dir):
            print(f"[ERROR] 目录不存在: {root_dir}")
            sys.exit(1)

        csv_files = find_summary_csvs_in_dir(root_dir)
        if not csv_files:
            print(f"[ERROR] 在 {root_dir} 下未找到任何 _summary.csv 文件")
            sys.exit(1)

        print(f"[INFO] 在 {root_dir} 下找到 {len(csv_files)} 个 summary CSV:")
        for cf in csv_files:
            method = extract_method_from_csv_path(cf)
            data = read_summary_csv(cf)
            entries.append({"method": method, "data": data, "csv_path": cf})
            print(f"  - {method}: {cf}")
        print()

    elif args.csv:
        # 手动模式
        for csv_path in args.csv:
            if not os.path.exists(csv_path):
                print(f"[WARN] CSV not found: {csv_path}, skipping")
                continue
            method = extract_method_from_csv_path(csv_path)
            data = read_summary_csv(csv_path)
            entries.append({"method": method, "data": data, "csv_path": csv_path})

    elif args.dataset:
        # 自动模式
        if args.dataset not in DATASET_MODELS:
            print(f"[ERROR] No model list for dataset '{args.dataset}'")
            print(f"  Available: {list(DATASET_MODELS.keys())}")
            sys.exit(1)

        for model_path in DATASET_MODELS[args.dataset]:
            csv_path = find_summary_csv(model_path, args.dataset, args.split)
            method = extract_method_name(model_path)
            if not os.path.exists(csv_path):
                print(f"[WARN] Summary not found for {method}: {csv_path}")
                entries.append({"method": method, "data": {}, "csv_path": csv_path})
                continue
            data = read_summary_csv(csv_path)
            entries.append({"method": method, "data": data, "csv_path": csv_path})
    else:
        parser.error("请指定 --dir、--dataset 或 --csv")

    if not entries:
        print("[ERROR] 没有找到任何结果")
        sys.exit(1)

    # ---- 确定 region names ----
    if args.dataset and args.dataset in DATASET_REGIONS:
        region_names = DATASET_REGIONS[args.dataset]
    else:
        # 从 data keys 推断
        sample_data = next((e["data"] for e in entries if e["data"]), {})
        region_names = []
        for k in sample_data:
            if k.startswith("dice_") and not k.endswith("_avg"):
                region_names.append(k.replace("dice_", ""))

    # ---- 构建表格列 ----
    is_pct = args.percent

    # 每个 metric 显示: 各 region + avg
    metric_cols = []
    for m in METRICS:
        for rn in region_names:
            metric_cols.append(f"{m}_{rn}")
        metric_cols.append(f"{m}_avg")

    col_w = 12
    method_w = 45
    total_w = method_w + len(metric_cols) * col_w

    # ---- 打印表格 ----
    ds_label = args.dataset or (os.path.basename(args.dir.rstrip("/")) if args.dir else "custom")
    print("=" * total_w)
    title = f"Results: {ds_label} ({args.split} split)"
    if is_pct:
        title += "  [Dice in %]"
    print(f"  {title}")
    print("=" * total_w)

    # 表头
    hdr = f"{'Method':<{method_w}}"
    for c in metric_cols:
        hdr += f"{c:>{col_w}}"
    print(hdr)
    print("-" * total_w)

    # 每行
    rows_for_csv = []
    for entry in entries:
        method = entry["method"]
        data = entry["data"]

        row_str = f"{method:<{method_w}}"
        row_csv = {"method": method}

        for c in metric_cols:
            m_name = c.split("_")[0]  # dice or hd95
            is_overlap = m_name in ("dice",)
            v = data.get(c, float("nan"))
            row_csv[c] = v
            row_str += f"{fmt(v, is_pct and is_overlap):>{col_w}}"

        print(row_str)
        rows_for_csv.append(row_csv)

    print("-" * total_w)
    print()

    # ---- 保存汇总 CSV ----
    if args.output is None:
        out_dir = "results"
        os.makedirs(out_dir, exist_ok=True)
        ds_tag = ds_label
        output_csv = os.path.join(out_dir, f"compare_{ds_tag}_{args.split}.csv")
    else:
        output_csv = args.output
        os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)

    if rows_for_csv:
        fieldnames = list(rows_for_csv[0].keys())
        with open(output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows_for_csv:
                writer.writerow(row)
        print(f"[Saved] {output_csv}")

    print("[Done]")


if __name__ == "__main__":
    main()
