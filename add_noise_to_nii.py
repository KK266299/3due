#!/usr/bin/env python3
"""
将噪声添加到原始H5图像，并导出为NIfTI文件。

用法示例：
    # BraTS19 单个被试
    python scripts/add_noise_to_nii.py \
        --noise_dir /path/to/noise_artifacts \
        --h5_path /path/to/BraTS19_xxx.h5 \
        --output_dir /path/to/output \
        --dataset brats19

    # KiTS19 单个被试
    python scripts/add_noise_to_nii.py \
        --noise_dir /path/to/noise_artifacts \
        --h5_path /path/to/KiTS19_xxx.h5 \
        --output_dir /path/to/output \
        --dataset kits19

    # 批量处理（使用CSV）
    python scripts/add_noise_to_nii.py \
        --noise_dir /path/to/noise_artifacts \
        --csv_path /path/to/train.csv \
        --output_dir /path/to/output \
        --dataset brats19
"""

from __future__ import annotations

import os
import sys
import json
import argparse
from typing import Optional, Dict, Any, List, Tuple

import h5py
import numpy as np
import torch
import nibabel as nib

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.ue_artifacts import UEShardsAccessor


# BraTS19 通道名称
BRATS19_CHANNEL_NAMES = ["t1", "t1ce", "t2", "flair"]


def load_h5_image(h5_path: str) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    读取H5文件中的图像数据。

    Args:
        h5_path: H5文件路径

    Returns:
        image: (C, H, W, D), float32, [0,1]
        metadata: 包含case_id等元数据的字典
    """
    with h5py.File(h5_path, "r") as f:
        image = f["image"][()]  # (C, H, W, D), float32, [0,1]

        metadata = {}
        for key in f.attrs.keys():
            metadata[key] = f.attrs[key]

        # 尝试获取case_id
        if "case_id" in f.attrs:
            metadata["case_id"] = str(f.attrs["case_id"])
        else:
            # 从文件名推断
            metadata["case_id"] = os.path.splitext(os.path.basename(h5_path))[0]

    return image, metadata


def get_noise_key(case_id: str, accessor: UEShardsAccessor) -> Any:
    """
    获取噪声的key。

    尝试多种可能的key格式，找到匹配的。
    """
    available_keys = accessor.keys()

    # 尝试直接匹配
    if case_id in available_keys:
        return case_id

    # 尝试整数索引
    for key in available_keys:
        if isinstance(key, int):
            continue
        if str(key) == case_id:
            return key

    # 打印可用keys帮助调试
    print(f"[DEBUG] Available keys (first 10): {available_keys[:10]}")
    print(f"[DEBUG] Looking for case_id: {case_id}")

    raise KeyError(f"Cannot find noise key for case_id: {case_id}")


def add_noise_to_image(
    image: np.ndarray,
    noise: torch.Tensor,
    clamp_range: Tuple[float, float] = (0.0, 1.0)
) -> np.ndarray:
    """
    将噪声添加到图像上。

    Args:
        image: (C, H, W, D), float32, [0,1] - H5中存储的格式
        noise: (C, D, H, W), float32 - 噪声（PyTorch格式）
        clamp_range: 输出范围

    Returns:
        noisy_image: (C, H, W, D), float32
    """
    # 将image从 (C, H, W, D) 转换为 (C, D, H, W) 以匹配噪声格式
    image_cdhw = np.transpose(image, (0, 3, 1, 2))  # (C, H, W, D) -> (C, D, H, W)

    # 将噪声转为numpy
    noise_np = noise.numpy()  # (C, D, H, W)

    # 检查形状
    if image_cdhw.shape != noise_np.shape:
        raise ValueError(
            f"Shape mismatch: image shape {image_cdhw.shape} vs noise shape {noise_np.shape}"
        )

    # 添加噪声
    noisy_image_cdhw = image_cdhw + noise_np

    # Clamp到有效范围
    noisy_image_cdhw = np.clip(noisy_image_cdhw, clamp_range[0], clamp_range[1])

    # 转回 (C, H, W, D) 格式
    noisy_image = np.transpose(noisy_image_cdhw, (0, 2, 3, 1))  # (C, D, H, W) -> (C, H, W, D)

    return noisy_image


def save_as_nifti(
    image: np.ndarray,
    output_path: str,
    affine: Optional[np.ndarray] = None
) -> None:
    """
    保存图像为NIfTI文件。

    Args:
        image: 3D数组 (H, W, D)
        output_path: 输出路径
        affine: 仿射矩阵，默认为单位矩阵
    """
    if affine is None:
        affine = np.eye(4)

    # 确保是float32
    image = image.astype(np.float32)

    # 创建NIfTI对象
    nii = nib.Nifti1Image(image, affine)

    # 保存
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    nib.save(nii, output_path)
    print(f"Saved: {output_path}")


def process_single_subject(
    h5_path: str,
    noise_accessor: UEShardsAccessor,
    output_dir: str,
    dataset: str,
    noise_key: Optional[Any] = None,
    save_original: bool = False,
    save_noise_only: bool = False,
) -> None:
    """
    处理单个被试。

    Args:
        h5_path: H5文件路径
        noise_accessor: 噪声访问器
        output_dir: 输出目录
        dataset: 数据集类型 ('brats19' 或 'kits19')
        noise_key: 可选，指定噪声的key
        save_original: 是否同时保存原始图像
        save_noise_only: 是否只保存噪声（用于调试）
    """
    # 加载图像
    image, metadata = load_h5_image(h5_path)
    case_id = metadata.get("case_id", os.path.splitext(os.path.basename(h5_path))[0])

    print(f"\n{'='*60}")
    print(f"Processing: {case_id}")
    print(f"  H5 path: {h5_path}")
    print(f"  Image shape (C, H, W, D): {image.shape}")
    print(f"  Image range: [{image.min():.4f}, {image.max():.4f}]")

    # 获取噪声
    if noise_key is None:
        noise_key = get_noise_key(case_id, noise_accessor)

    noise = noise_accessor.get(noise_key)  # (C, D, H, W), float32
    print(f"  Noise shape (C, D, H, W): {noise.shape}")
    print(f"  Noise range: [{noise.min():.4f}, {noise.max():.4f}]")

    # 添加噪声
    noisy_image = add_noise_to_image(image, noise)
    print(f"  Noisy image range: [{noisy_image.min():.4f}, {noisy_image.max():.4f}]")

    # 确定通道数和名称
    num_channels = image.shape[0]

    if dataset.lower() == "brats19":
        if num_channels != 4:
            print(f"  [WARNING] Expected 4 channels for BraTS19, got {num_channels}")
        channel_names = BRATS19_CHANNEL_NAMES[:num_channels]
    elif dataset.lower() == "kits19":
        if num_channels != 1:
            print(f"  [WARNING] Expected 1 channel for KiTS19, got {num_channels}")
        channel_names = ["ct"] if num_channels == 1 else [f"ch{i}" for i in range(num_channels)]
    else:
        channel_names = [f"ch{i}" for i in range(num_channels)]

    # 创建输出目录
    subject_output_dir = os.path.join(output_dir, case_id)
    os.makedirs(subject_output_dir, exist_ok=True)

    # 保存每个通道为独立的NIfTI文件
    for ch_idx, ch_name in enumerate(channel_names):
        # 加噪图像
        noisy_ch = noisy_image[ch_idx]  # (H, W, D)
        output_path = os.path.join(subject_output_dir, f"{case_id}_{ch_name}_noisy.nii.gz")
        save_as_nifti(noisy_ch, output_path)

        # 可选：保存原始图像
        if save_original:
            orig_ch = image[ch_idx]  # (H, W, D)
            orig_path = os.path.join(subject_output_dir, f"{case_id}_{ch_name}_original.nii.gz")
            save_as_nifti(orig_ch, orig_path)

        # 可选：保存噪声
        if save_noise_only:
            # 噪声是 (C, D, H, W) 格式，转为 (C, H, W, D)
            noise_np = noise.numpy()
            noise_np_chwD = np.transpose(noise_np, (0, 2, 3, 1))  # (C, D, H, W) -> (C, H, W, D)
            noise_ch = noise_np_chwD[ch_idx]  # (H, W, D)
            noise_path = os.path.join(subject_output_dir, f"{case_id}_{ch_name}_noise.nii.gz")
            save_as_nifti(noise_ch, noise_path)

    print(f"  Output: {subject_output_dir}")
    print(f"  Generated {num_channels} NIfTI files")


def process_from_csv(
    csv_path: str,
    noise_accessor: UEShardsAccessor,
    output_dir: str,
    dataset: str,
    max_subjects: Optional[int] = None,
    save_original: bool = False,
    save_noise_only: bool = False,
) -> None:
    """
    从CSV批量处理被试。
    """
    import pandas as pd

    df = pd.read_csv(csv_path)

    if "volume_path" not in df.columns:
        raise ValueError("CSV must have 'volume_path' column")

    total = len(df)
    if max_subjects is not None:
        df = df.head(max_subjects)

    print(f"Processing {len(df)}/{total} subjects from {csv_path}")

    for idx, row in df.iterrows():
        h5_path = row["volume_path"]

        # 确定noise_key
        # 可以是case_id或者index
        noise_key = None
        if "case_id" in row:
            case_id = str(row["case_id"])
            # 尝试case_id作为key
            try:
                noise_key = get_noise_key(case_id, noise_accessor)
            except KeyError:
                # 尝试index作为key
                noise_key = idx
        else:
            noise_key = idx

        try:
            process_single_subject(
                h5_path=h5_path,
                noise_accessor=noise_accessor,
                output_dir=output_dir,
                dataset=dataset,
                noise_key=noise_key,
                save_original=save_original,
                save_noise_only=save_noise_only,
            )
        except Exception as e:
            print(f"[ERROR] Failed to process {h5_path}: {e}")
            continue


def main():
    parser = argparse.ArgumentParser(
        description="将噪声添加到H5图像并导出为NIfTI文件",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        "--noise_dir",
        type=str,
        required=True,
        help="噪声artifacts目录（包含manifest.json）"
    )
    parser.add_argument(
        "--h5_path",
        type=str,
        default=None,
        help="单个H5文件路径（与--csv_path二选一）"
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        default=None,
        help="包含H5路径的CSV文件（与--h5_path二选一）"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="输出目录"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["brats19", "kits19"],
        default="brats19",
        help="数据集类型（决定输出通道数和命名）"
    )
    parser.add_argument(
        "--noise_key",
        type=str,
        default=None,
        help="指定噪声的key（用于单个被试时）"
    )
    parser.add_argument(
        "--max_subjects",
        type=int,
        default=None,
        help="最大处理被试数（用于CSV模式）"
    )
    parser.add_argument(
        "--save_original",
        action="store_true",
        help="同时保存原始图像"
    )
    parser.add_argument(
        "--save_noise_only",
        action="store_true",
        help="同时保存噪声（用于调试）"
    )
    parser.add_argument(
        "--list_keys",
        action="store_true",
        help="列出所有可用的噪声keys并退出"
    )

    args = parser.parse_args()

    # 检查参数
    if args.h5_path is None and args.csv_path is None:
        parser.error("必须指定 --h5_path 或 --csv_path")
    if args.h5_path is not None and args.csv_path is not None:
        parser.error("--h5_path 和 --csv_path 不能同时指定")

    # 加载噪声accessor
    manifest_path = os.path.join(args.noise_dir, "manifest.json")
    if not os.path.exists(manifest_path):
        print(f"[ERROR] Manifest not found: {manifest_path}")
        sys.exit(1)

    print(f"Loading noise from: {manifest_path}")
    accessor = UEShardsAccessor.from_manifest(manifest_path)

    print(f"  Strategy: {accessor.strategy}")
    print(f"  Scale: {accessor.scale}")
    print(f"  Image size: {accessor.image_size}")
    print(f"  Perturb type: {accessor.perturb_type}")
    print(f"  Total keys: {len(accessor.keys())}")

    # 列出keys模式
    if args.list_keys:
        print("\nAvailable keys:")
        for key in accessor.keys():
            print(f"  {key}")
        sys.exit(0)

    # 处理
    if args.h5_path is not None:
        # 单个被试模式
        noise_key = args.noise_key
        if noise_key is not None:
            # 尝试转换为整数
            try:
                noise_key = int(noise_key)
            except ValueError:
                pass  # 保持字符串

        process_single_subject(
            h5_path=args.h5_path,
            noise_accessor=accessor,
            output_dir=args.output_dir,
            dataset=args.dataset,
            noise_key=noise_key,
            save_original=args.save_original,
            save_noise_only=args.save_noise_only,
        )
    else:
        # CSV批量模式
        process_from_csv(
            csv_path=args.csv_path,
            noise_accessor=accessor,
            output_dir=args.output_dir,
            dataset=args.dataset,
            max_subjects=args.max_subjects,
            save_original=args.save_original,
            save_noise_only=args.save_noise_only,
        )

    print("\n" + "="*60)
    print("Done!")


if __name__ == "__main__":
    main()