#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
图像质量评估脚本 (Image Quality Assessment)

评估加噪后图像与原始图像之间的 PSNR 和 SSIM 指标。
支持 3D 医学图像 (BraTS19 等)，可按 volume 和 slice 两种粒度计算。

使用示例：
python image_qua.py \
    --noise_manifest /path/to/noise/epoch_0099/manifest.json \
    --output_dir outputs/image_quality \
    --dataset_config configs/dataset/brats19.yaml \
    --metrics psnr ssim \
    --use_pyiqa
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from omegaconf import OmegaConf

# 添加项目根目录到 path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.registry import get_dataset_builder
from src.core.ue_artifacts import UEShardsAccessor
from src.core.ue_keys import extract_key
from src.utils.config import get_config
from src.utils.eval_metrics import compute_psnr, compute_ssim

# 可选: pyiqa
try:
    import pyiqa
    HAS_PYIQA = True
except ImportError:
    HAS_PYIQA = False


# -------------------------------------------------------------------------
# 配置加载 (与 visualize_unet_noise.py 保持一致)
# -------------------------------------------------------------------------

def load_config(config_path: str = None, dataset_config: str = None) -> OmegaConf:
    """加载配置文件"""
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

    if config_path and os.path.exists(config_path):
        user_config = OmegaConf.load(config_path)
        config = OmegaConf.merge(config, user_config)

    if dataset_config and os.path.exists(dataset_config):
        ds_cfg = OmegaConf.load(dataset_config)
        config.dataset = ds_cfg
    elif not hasattr(config, 'dataset') or config.dataset is None:
        default_ds_path = Path(__file__).parent / "configs" / "dataset" / "brats19.yaml"
        if default_ds_path.exists():
            config.dataset = OmegaConf.load(str(default_ds_path))
        else:
            raise ValueError(
                "未找到数据集配置，请通过 --dataset_config 指定，例如：\n"
                "  --dataset_config configs/dataset/brats19.yaml"
            )

    return config


def load_dataset(config: OmegaConf, split: str = "train"):
    """加载数据集"""
    ds_name = config.dataset.name
    if ds_name == "brats19":
        ds_name = "brats19_seg"
    builder_cls = get_dataset_builder(ds_name)
    builder = builder_cls(config)
    return builder.get_dataset(split=split)


def load_noise(manifest_path: str, key: str) -> Optional[torch.Tensor]:
    """通过 manifest 加载指定 key 的噪声张量"""
    accessor = UEShardsAccessor.from_manifest(manifest_path)
    try:
        noise = accessor.get(key, perturb_type="samplewise")
        return noise
    except KeyError:
        return None


def load_noise_accessor(manifest_path: str) -> UEShardsAccessor:
    """加载 noise accessor，可复用"""
    return UEShardsAccessor.from_manifest(manifest_path)


# -------------------------------------------------------------------------
# 指标计算
# -------------------------------------------------------------------------

def compute_volume_psnr(original: torch.Tensor, noisy: torch.Tensor) -> float:
    """
    计算 3D volume 的 PSNR。

    Args:
        original: [C, D, H, W] 原始图像
        noisy:    [C, D, H, W] 加噪图像

    Returns:
        PSNR 值 (dB)
    """
    # 添加 batch 维度 -> [1, C, D, H, W]
    orig_batch = original.unsqueeze(0).float()
    noisy_batch = noisy.unsqueeze(0).float()
    psnr_val = compute_psnr(orig_batch, noisy_batch, data_range=1.0)
    return float(psnr_val.item())


def compute_volume_ssim(original: torch.Tensor, noisy: torch.Tensor) -> float:
    """
    计算 3D volume 的 SSIM (逐 slice 平均)。

    Args:
        original: [C, D, H, W] 原始图像
        noisy:    [C, D, H, W] 加噪图像

    Returns:
        SSIM 值
    """
    C, D, H, W = original.shape
    ssim_vals = []
    for d in range(D):
        # 取出单个 slice: [C, H, W] -> [1, C, H, W]
        orig_slice = original[:, d, :, :].unsqueeze(0).float()
        noisy_slice = noisy[:, d, :, :].unsqueeze(0).float()
        val = compute_ssim(orig_slice, noisy_slice, data_range=1.0, size_average=True)
        ssim_vals.append(float(val.item()))
    return float(np.mean(ssim_vals))


def compute_per_slice_metrics(
    original: torch.Tensor,
    noisy: torch.Tensor,
) -> List[Dict[str, float]]:
    """
    逐 slice 计算 PSNR 和 SSIM。

    Args:
        original: [C, D, H, W]
        noisy:    [C, D, H, W]

    Returns:
        List of dicts, 每个 slice 一个 dict: {"slice_idx", "psnr", "ssim"}
    """
    C, D, H, W = original.shape
    results = []
    for d in range(D):
        orig_slice = original[:, d, :, :].unsqueeze(0).float()
        noisy_slice = noisy[:, d, :, :].unsqueeze(0).float()

        psnr_val = compute_psnr(orig_slice, noisy_slice, data_range=1.0)
        ssim_val = compute_ssim(orig_slice, noisy_slice, data_range=1.0, size_average=True)

        results.append({
            "slice_idx": d,
            "psnr": float(psnr_val.item()),
            "ssim": float(ssim_val.item()),
        })
    return results


def compute_pyiqa_metrics(
    original: torch.Tensor,
    noisy: torch.Tensor,
    metric_names: List[str],
    device: str = "cpu",
) -> Dict[str, float]:
    """
    使用 pyiqa 库计算图像质量指标 (逐 slice 平均)。

    Args:
        original: [C, D, H, W]
        noisy:    [C, D, H, W]
        metric_names: 例如 ['psnr', 'ssim', 'lpips', 'dists']
        device: 计算设备

    Returns:
        Dict of metric_name -> value
    """
    if not HAS_PYIQA:
        print("[Warning] pyiqa 未安装，跳过 pyiqa 指标。安装方式: pip install pyiqa")
        return {}

    metrics = {}
    for name in metric_names:
        try:
            metrics[name] = pyiqa.create_metric(name, device=device)
        except Exception as e:
            print(f"[Warning] 无法创建 pyiqa 指标 '{name}': {e}")

    if not metrics:
        return {}

    C, D, H, W = original.shape
    accum = {name: 0.0 for name in metrics}
    count = 0

    for d in range(D):
        # [C, H, W] -> [1, C, H, W]
        orig_slice = original[:, d, :, :].unsqueeze(0).float().to(device)
        noisy_slice = noisy[:, d, :, :].unsqueeze(0).float().to(device)

        # pyiqa 对 LPIPS/DISTS 需要 3 通道
        if orig_slice.shape[1] == 1:
            orig_3ch = orig_slice.repeat(1, 3, 1, 1)
            noisy_3ch = noisy_slice.repeat(1, 3, 1, 1)
        elif orig_slice.shape[1] > 3:
            orig_3ch = orig_slice[:, :3, :, :]
            noisy_3ch = noisy_slice[:, :3, :, :]
        else:
            orig_3ch = orig_slice
            noisy_3ch = noisy_slice

        for name, metric_fn in metrics.items():
            try:
                # PSNR/SSIM 可以用原始通道; LPIPS 等需要 3 通道
                if name in ('lpips', 'lpips-vgg', 'dists'):
                    score = metric_fn(noisy_3ch, orig_3ch)
                else:
                    score = metric_fn(noisy_slice, orig_slice)
                accum[name] += float(score.mean().item())
            except Exception:
                pass

        count += 1

    results = {}
    for name in metrics:
        results[f"pyiqa_{name}"] = accum[name] / max(count, 1)

    return results


# -------------------------------------------------------------------------
# 结果输出
# -------------------------------------------------------------------------

def print_summary_table(results: List[Dict], metrics_cols: List[str]):
    """打印汇总表格到终端"""
    # 表头
    header = ["Case ID", "Num Slices"] + metrics_cols
    col_widths = [max(len(h), 12) for h in header]

    # 更新列宽
    for row in results:
        for i, h in enumerate(header):
            val = str(row.get(h, ""))
            col_widths[i] = max(col_widths[i], len(val))

    # 分隔线
    sep = "+" + "+".join("-" * (w + 2) for w in col_widths) + "+"

    # 打印
    print("\n" + "=" * 60)
    print("图像质量评估结果 (Image Quality Assessment Results)")
    print("=" * 60)

    print(sep)
    print("|" + "|".join(f" {h:^{w}} " for h, w in zip(header, col_widths)) + "|")
    print(sep)

    for row in results:
        vals = []
        for h in header:
            v = row.get(h, "")
            if isinstance(v, float):
                vals.append(f"{v:.4f}")
            else:
                vals.append(str(v))
        print("|" + "|".join(f" {v:^{w}} " for v, w in zip(vals, col_widths)) + "|")

    print(sep)

    # 均值行
    if len(results) > 1:
        avg_row = ["MEAN", "-"]
        for col in metrics_cols:
            vals = [r[col] for r in results if isinstance(r.get(col), (int, float))]
            if vals:
                avg_row.append(f"{np.mean(vals):.4f}")
            else:
                avg_row.append("-")
        print("|" + "|".join(f" {v:^{w}} " for v, w in zip(avg_row, col_widths)) + "|")

        std_row = ["STD", "-"]
        for col in metrics_cols:
            vals = [r[col] for r in results if isinstance(r.get(col), (int, float))]
            if vals:
                std_row.append(f"{np.std(vals):.4f}")
            else:
                std_row.append("-")
        print("|" + "|".join(f" {v:^{w}} " for v, w in zip(std_row, col_widths)) + "|")

        print(sep)

    print()


def save_csv(results: List[Dict], output_path: str, fieldnames: List[str]):
    """保存结果到 CSV 文件"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    print(f"[Saved CSV] {output_path}")


def save_json(results: List[Dict], summary: Dict, output_path: str):
    """保存结果到 JSON 文件"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    data = {
        "summary": summary,
        "per_sample": results,
    }
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"[Saved JSON] {output_path}")


# -------------------------------------------------------------------------
# 主函数
# -------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="评估加噪后图像与原始图像的 PSNR / SSIM 指标"
    )
    parser.add_argument(
        "--noise_manifest", type=str, required=True,
        help="噪声 manifest.json 路径"
    )
    parser.add_argument(
        "--output_dir", type=str, default="outputs/image_quality",
        help="输出目录"
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="完整配置文件路径"
    )
    parser.add_argument(
        "--dataset_config", type=str, default=None,
        help="数据集配置文件路径，例如 configs/dataset/brats19.yaml"
    )
    parser.add_argument(
        "--split", type=str, default="train",
        help="数据集分割 (train/val/test)"
    )
    parser.add_argument(
        "--metrics", nargs="+", default=["psnr", "ssim"],
        help="使用 pyiqa 计算的指标列表 (仅当 --use_pyiqa 启用时有效)"
    )
    parser.add_argument(
        "--use_pyiqa", action="store_true",
        help="是否使用 pyiqa 库计算指标 (默认使用内置实现)"
    )
    parser.add_argument(
        "--per_slice", action="store_true",
        help="是否输出逐 slice 的详细结果"
    )
    parser.add_argument(
        "--max_samples", type=int, default=-1,
        help="最多评估的样本数，-1 表示全部"
    )
    parser.add_argument(
        "--device", type=str, default="cpu",
        help="计算设备 (cpu 或 cuda:X)"
    )

    args = parser.parse_args()

    # 检查 manifest 存在
    if not os.path.exists(args.noise_manifest):
        print(f"[Error] Manifest 文件不存在: {args.noise_manifest}")
        sys.exit(1)

    # 加载配置 & 数据集
    print("[1/4] 加载配置和数据集...")
    config = load_config(args.config, args.dataset_config)
    dataset = load_dataset(config, split=args.split)
    print(f"  数据集大小: {len(dataset)}")

    # 加载 noise accessor
    print("[2/4] 加载噪声数据...")
    accessor = load_noise_accessor(args.noise_manifest)
    available_keys = accessor.keys()
    print(f"  噪声 keys 数量: {len(available_keys)}")

    # 确定评估样本数
    num_samples = len(dataset)
    if args.max_samples > 0:
        num_samples = min(num_samples, args.max_samples)

    # key 配置
    key_spec = get_config(config, "ue.key", {})

    # 汇总结果
    all_results = []
    slice_results = []

    metrics_cols = ["PSNR", "SSIM", "Noise_L2", "Noise_Linf"]

    # pyiqa 指标列
    pyiqa_cols = []
    if args.use_pyiqa and HAS_PYIQA:
        pyiqa_cols = [f"pyiqa_{m}" for m in args.metrics]
        metrics_cols.extend(pyiqa_cols)

    print(f"[3/4] 评估 {num_samples} 个样本...")
    skipped = 0

    for idx in range(num_samples):
        sample = dataset[idx]
        image = sample["image"]  # [C, D, H, W]
        case_id = sample.get("case_id", f"sample_{idx}")

        # 提取 key 并加载噪声
        key = extract_key(sample, idx, key_spec)
        try:
            noise = accessor.get(key, perturb_type="samplewise")
        except KeyError:
            skipped += 1
            continue

        # 加噪图像: clip(image + noise, 0, 1)
        noisy_image = (image + noise).clamp(0, 1)

        # 计算 volume 级指标
        psnr_val = compute_volume_psnr(image, noisy_image)
        ssim_val = compute_volume_ssim(image, noisy_image)

        # 噪声统计
        noise_l2 = float(noise.pow(2).mean().sqrt().item())
        noise_linf = float(noise.abs().max().item())

        row = {
            "Case ID": case_id,
            "Num Slices": image.shape[1],
            "PSNR": psnr_val,
            "SSIM": ssim_val,
            "Noise_L2": noise_l2,
            "Noise_Linf": noise_linf,
        }

        # pyiqa 指标
        if args.use_pyiqa and HAS_PYIQA:
            pyiqa_res = compute_pyiqa_metrics(
                image, noisy_image, args.metrics, device=args.device
            )
            row.update(pyiqa_res)

        all_results.append(row)

        # 逐 slice 结果
        if args.per_slice:
            per_slice = compute_per_slice_metrics(image, noisy_image)
            for s in per_slice:
                s["case_id"] = case_id
            slice_results.extend(per_slice)

        # 进度
        if (idx + 1) % 10 == 0 or (idx + 1) == num_samples:
            print(f"  [{idx + 1}/{num_samples}] {case_id}: PSNR={psnr_val:.2f} dB, SSIM={ssim_val:.4f}")

    if skipped > 0:
        print(f"  [Info] 跳过 {skipped} 个样本 (未找到对应噪声)")

    # 打印汇总表格
    print_summary_table(all_results, metrics_cols)

    # 计算汇总统计
    summary = {}
    for col in metrics_cols:
        vals = [r[col] for r in all_results if isinstance(r.get(col), (int, float))]
        if vals:
            summary[col] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
                "min": float(np.min(vals)),
                "max": float(np.max(vals)),
            }

    summary["num_samples"] = len(all_results)
    summary["num_skipped"] = skipped
    summary["noise_manifest"] = args.noise_manifest

    # 保存结果
    print("[4/4] 保存结果...")
    os.makedirs(args.output_dir, exist_ok=True)

    # CSV
    csv_fields = ["Case ID", "Num Slices"] + metrics_cols
    save_csv(all_results, os.path.join(args.output_dir, "results.csv"), csv_fields)

    # JSON
    save_json(all_results, summary, os.path.join(args.output_dir, "results.json"))

    # 逐 slice CSV
    if args.per_slice and slice_results:
        slice_fields = ["case_id", "slice_idx", "psnr", "ssim"]
        save_csv(slice_results, os.path.join(args.output_dir, "results_per_slice.csv"), slice_fields)

    # 打印汇总
    print("\n汇总统计:")
    print(f"  评估样本数: {len(all_results)}")
    for col in ["PSNR", "SSIM"]:
        if col in summary:
            s = summary[col]
            print(f"  {col}: mean={s['mean']:.4f}, std={s['std']:.4f}, "
                  f"min={s['min']:.4f}, max={s['max']:.4f}")

    print(f"\n结果已保存到: {args.output_dir}")


if __name__ == "__main__":
    main()
