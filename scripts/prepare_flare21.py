# FLARE21 数据预处理脚本
# 数据结构:
#   图像: TrainingImg/train_XXX_0000.nii.gz
#   标签: TrainingMask/train_XXX.nii.gz
# 按照 KiTS19 的处理方式进行预处理
import os
import glob
import argparse
import yaml
import csv
import numpy as np

import nibabel as nib
from nibabel.orientations import aff2axcodes
from scipy.ndimage import zoom
import h5py


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def load_nifti_as_canonical(path: str):
    """
    使用 nibabel 读取 NIfTI，并转换到 RAS 方向（closest canonical）
    返回 data(float32)、affine、spacing(3,) 和 axcodes。
    """
    nii = nib.load(path)
    canonical = nib.as_closest_canonical(nii)
    data = canonical.get_fdata(dtype=np.float32)
    affine = canonical.affine
    zooms = canonical.header.get_zooms()[:3]
    axcodes = aff2axcodes(affine)
    return data, affine, zooms, axcodes


def zscore_and_to01_per_modality(
    vol: np.ndarray,
    z_clip: float,
    to_01: bool = True,
) -> np.ndarray:
    """
    对单个 3D 体积做 per-case z-score + clip + 全局线性映射到 [0,1]（可选）。
    vol: 3D array, (H, W, D)
    与 BraTS19 处理逻辑严格一致：只对前景区域归一化，背景保持为0
    """
    # 对于CT图像，使用 > -500 HU 作为前景mask
    foreground_mask = vol > -500

    if np.sum(foreground_mask) == 0:
        mean = 0.0
        std = 1.0
    else:
        vals = vol[foreground_mask]
        mean = float(vals.mean())
        std = float(vals.std())
        if std < 1e-6:
            std = 1.0

    # 只对前景区域归一化，背景保持为0
    vol_z = np.zeros_like(vol, dtype=np.float32)
    vol_z[foreground_mask] = (vol[foreground_mask] - mean) / std

    # clip
    vol_z = np.clip(vol_z, -z_clip, z_clip)

    if not to_01:
        return vol_z

    # [-z_clip, z_clip] -> [0,1]
    vol_01 = (vol_z + z_clip) / (2.0 * z_clip)
    vol_01 = np.clip(vol_01, 0.0, 1.0)
    return vol_01


def ct_window_normalize(
    vol: np.ndarray,
    window_center: float = 50.0,
    window_width: float = 400.0,
    to_01: bool = True,
) -> np.ndarray:
    """
    CT专用：使用窗宽窗位（Window/Level）做归一化。

    vol: 3D array, (H, W, D), HU values
    window_center: 窗位 (Window Level), 默认50 HU（软组织窗）
    window_width: 窗宽 (Window Width), 默认400 HU
    to_01: 是否映射到[0,1]，否则映射到[-1,1]

    常用窗宽窗位参考:
      - 软组织窗: WL=40-50, WW=350-400 (适合观察腹部器官)
      - 腹部窗: WL=40, WW=400
    """
    # 计算窗宽窗位对应的HU范围
    hu_min = window_center - window_width / 2.0
    hu_max = window_center + window_width / 2.0

    # clip到窗宽范围
    vol_clipped = np.clip(vol, hu_min, hu_max)

    if to_01:
        # [hu_min, hu_max] -> [0, 1]
        vol_norm = (vol_clipped - hu_min) / (hu_max - hu_min)
    else:
        # [hu_min, hu_max] -> [-1, 1]
        vol_norm = 2.0 * (vol_clipped - hu_min) / (hu_max - hu_min) - 1.0

    return vol_norm.astype(np.float32)


def ct_nnunet_normalize(
    vol: np.ndarray,
    global_mean: float,
    global_std: float,
    lower_bound: float,
    upper_bound: float,
    to_01: bool = True,
    z_clip: float = 5.0,
) -> np.ndarray:
    """
    nnU-Net风格的CT归一化。

    步骤:
    1. clip到[lower_bound, upper_bound]（通常是0.5%和99.5%分位数）
    2. 减去全局均值
    3. 除以全局标准差
    4. (可选) 映射到[0,1]，与BraTS19保持一致

    Args:
        vol: 3D array, (H, W, D), HU values
        global_mean: 全局均值（从整个数据集前景计算）
        global_std: 全局标准差（从整个数据集前景计算）
        lower_bound: 下界（通常是0.5%分位数）
        upper_bound: 上界（通常是99.5%分位数）
        to_01: 是否映射到[0,1]
        z_clip: z-score clip范围（默认5.0）

    Returns:
        归一化后的体积
    """
    vol = vol.astype(np.float32, copy=True)
    np.clip(vol, lower_bound, upper_bound, out=vol)
    vol -= global_mean
    vol /= max(global_std, 1e-8)

    if to_01:
        vol = np.clip(vol, -z_clip, z_clip)
        vol = (vol + z_clip) / (2.0 * z_clip)
        vol = np.clip(vol, 0.0, 1.0)

    return vol


def compute_dataset_statistics(
    cases: list,
    img_root: str,
    mask_root: str,
    use_mask: bool = True,
) -> dict:
    """
    计算整个数据集的统计量，用于nnU-Net风格的归一化。

    Args:
        cases: case_id 列表
        img_root: 图像目录
        mask_root: 标签目录
        use_mask: 是否使用分割标签mask定义前景

    Returns:
        dict: {
            'mean': 全局均值,
            'std': 全局标准差,
            'percentile_00_5': 0.5%分位数,
            'percentile_99_5': 99.5%分位数,
        }
    """
    print("[INFO] Computing dataset statistics for nnU-Net normalization...")
    print(f"  Foreground definition: {'segmentation mask (seg > 0)' if use_mask else 'HU threshold (> -500)'}")

    all_foreground_values = []

    for i, case_id in enumerate(cases):
        try:
            # 加载影像
            img_path = os.path.join(img_root, f"{case_id}_0000.nii.gz")
            vol, _, _, _ = load_nifti_as_canonical(img_path)

            # 确定前景mask
            if use_mask:
                # 使用分割标签定义前景
                seg_path = os.path.join(mask_root, f"{case_id}.nii.gz")
                seg, _, _, _ = load_nifti_as_canonical(seg_path)
                seg = seg.astype(np.int16)
                foreground_mask = seg > 0
            else:
                foreground_mask = vol > -500

            if np.sum(foreground_mask) > 0:
                fg_vals = vol[foreground_mask].flatten()
                # 随机采样最多100000个值
                if len(fg_vals) > 100000:
                    idx = np.random.choice(len(fg_vals), 100000, replace=False)
                    fg_vals = fg_vals[idx]
                all_foreground_values.append(fg_vals)

            if (i + 1) % 20 == 0:
                print(f"  Processed {i + 1}/{len(cases)} cases...")

        except Exception as e:
            print(f"  [WARN] Failed to process {case_id}: {e}")
            continue

    if not all_foreground_values:
        raise ValueError("No valid foreground values found in dataset!")

    all_values = np.concatenate(all_foreground_values)
    print(f"  Total foreground voxels sampled: {len(all_values):,}")

    stats = {
        'mean': float(np.mean(all_values)),
        'std': float(np.std(all_values)),
        'percentile_00_5': float(np.percentile(all_values, 0.5)),
        'percentile_99_5': float(np.percentile(all_values, 99.5)),
    }

    print(f"  Dataset statistics:")
    print(f"    mean: {stats['mean']:.2f}")
    print(f"    std: {stats['std']:.2f}")
    print(f"    percentile_00_5: {stats['percentile_00_5']:.2f}")
    print(f"    percentile_99_5: {stats['percentile_99_5']:.2f}")

    return stats


def resize_volume(
    vol: np.ndarray,
    target_shape: tuple,
    is_label: bool = False,
) -> np.ndarray:
    """
    使用 scipy.ndimage.zoom 把 vol resize 到 target_shape。
    vol: (H, W, D) 或 (C, H, W, D)
    target_shape: (H_t, W_t, D_t)
    """
    if vol.ndim == 4:
        # (C, H, W, D)
        c, h, w, d = vol.shape
        th, tw, td = target_shape
        zoom_factors = (1.0, th / h, tw / w, td / d)
        order = 0 if is_label else 3
        resized = zoom(vol, zoom_factors, order=order)
    elif vol.ndim == 3:
        # (H, W, D)
        h, w, d = vol.shape
        th, tw, td = target_shape
        zoom_factors = (th / h, tw / w, td / d)
        order = 0 if is_label else 3
        resized = zoom(vol, zoom_factors, order=order)
    else:
        raise ValueError(f"Unsupported volume ndim={vol.ndim}")
    return resized.astype(vol.dtype)


def resample_to_spacing(
    vol: np.ndarray,
    orig_spacing: tuple,
    target_spacing: tuple,
    is_label: bool = False,
) -> np.ndarray:
    """
    基于spacing的重采样。

    vol: (H, W, D) 或 (C, H, W, D)
    orig_spacing: 原始spacing (sz, sy, sx)
    target_spacing: 目标spacing
    is_label: 如果是label则使用最近邻插值

    Returns:
        重采样后的体积
    """
    orig_spacing = np.array(orig_spacing, dtype=np.float64)
    target_spacing = np.array(target_spacing, dtype=np.float64)

    if vol.ndim == 4:
        spatial_shape = np.array(vol.shape[1:])
        zoom_factors_spatial = orig_spacing / target_spacing
        zoom_factors = np.array([1.0] + zoom_factors_spatial.tolist())
    elif vol.ndim == 3:
        spatial_shape = np.array(vol.shape)
        zoom_factors = orig_spacing / target_spacing
    else:
        raise ValueError(f"Unsupported volume ndim={vol.ndim}")

    order = 0 if is_label else 3
    resized = zoom(vol, zoom_factors, order=order)
    return resized.astype(vol.dtype)


def remap_labels(seg: np.ndarray, mapping: dict) -> np.ndarray:
    """
    seg: 3D label map
    mapping: dict, e.g. {0:0, 1:1, 2:2, 3:3, 4:4}
    """
    seg_remap = np.zeros_like(seg, dtype=np.uint8)
    for src, dst in mapping.items():
        seg_remap[seg == int(src)] = int(dst)
    return seg_remap


def scan_flare21_cases(img_root: str, mask_root: str) -> list:
    """
    扫描 FLARE21 数据目录，返回所有 case_id 列表。
    FLARE21 结构：
      - 图像: img_root/train_XXX_0000.nii.gz
      - 标签: mask_root/train_XXX.nii.gz
    """
    cases = []
    # 查找所有图像文件
    pattern = os.path.join(img_root, "train_*_0000.nii.gz")
    img_files = sorted(glob.glob(pattern))

    for img_path in img_files:
        # 从 train_XXX_0000.nii.gz 提取 train_XXX
        filename = os.path.basename(img_path)
        # 移除 _0000.nii.gz 后缀
        case_id = filename.replace("_0000.nii.gz", "")

        # 检查对应的 mask 是否存在
        mask_path = os.path.join(mask_root, f"{case_id}.nii.gz")
        if os.path.exists(mask_path):
            cases.append({
                "case_id": case_id,
                "img_path": img_path,
                "mask_path": mask_path,
            })
        else:
            print(f"[WARN] Skipping {case_id}: mask file not found at {mask_path}")

    return cases


def process_case(
    case_info: dict,
    label_remap: dict,
    target_shape: tuple,
    z_clip: float,
    to_01: bool,
    out_dir: str,
    target_spacing: tuple = None,
    resample_mode: str = "shape",
    norm_mode: str = "zscore",
    window_center: float = 50.0,
    window_width: float = 400.0,
    dataset_stats: dict = None,
):
    """
    处理单个 FLARE21 case：
    - 读入 CT 图像 + seg
    - 转 canonical orientation
    - 归一化（支持多种方式）
    - resize/resample
    - label remap
    - 保存 h5 文件

    Args:
        case_info: dict with 'case_id', 'img_path', 'mask_path'
        label_remap: 标签重映射
        target_shape: 目标尺寸
        z_clip: z-score 截断阈值
        to_01: 是否映射到[0,1]
        out_dir: 输出目录
        target_spacing: 目标spacing（当resample_mode="spacing"时使用）
        resample_mode: "shape" 或 "spacing"
        norm_mode: 归一化模式 "zscore", "ct_window", "nnunet"
        window_center: 窗位
        window_width: 窗宽
        dataset_stats: 当 norm_mode="nnunet" 时使用的全局统计量

    返回：case_id, h5_path
    """
    case_id = case_info["case_id"]
    img_path = case_info["img_path"]
    mask_path = case_info["mask_path"]

    # ---- 1. 加载图像 ----
    vol, affine, zooms, axcodes = load_nifti_as_canonical(img_path)
    orig_shape = vol.shape
    orig_spacing = zooms

    # 根据norm_mode选择归一化方式
    if norm_mode == "nnunet":
        if dataset_stats is None:
            raise ValueError("norm_mode='nnunet' requires dataset_stats!")
        vol_norm = ct_nnunet_normalize(
            vol,
            global_mean=dataset_stats['mean'],
            global_std=dataset_stats['std'],
            lower_bound=dataset_stats['percentile_00_5'],
            upper_bound=dataset_stats['percentile_99_5'],
            to_01=to_01,
            z_clip=z_clip,
        )
    elif norm_mode == "ct_window":
        vol_norm = ct_window_normalize(
            vol,
            window_center=window_center,
            window_width=window_width,
            to_01=to_01,
        )
    else:
        # 默认使用zscore
        vol_norm = zscore_and_to01_per_modality(vol, z_clip=z_clip, to_01=to_01)

    # 添加通道维度: (H, W, D) -> (1, H, W, D)
    image = vol_norm[np.newaxis, ...].astype(np.float32)

    # ---- 2. 加载 seg ----
    seg_vol, seg_affine, _, _ = load_nifti_as_canonical(mask_path)
    seg_vol = seg_vol.astype(np.int16)

    if seg_vol.shape != orig_shape:
        raise ValueError(
            f"Seg shape {seg_vol.shape} != image shape {orig_shape} in case {case_id}"
        )

    # ---- 3. resize/resample ----
    if resample_mode == "spacing" and target_spacing is not None:
        image_resized = resample_to_spacing(
            image, orig_spacing, target_spacing, is_label=False
        )
        seg_resized = resample_to_spacing(
            seg_vol, orig_spacing, target_spacing, is_label=True
        )
        final_shape = image_resized.shape[1:]
    else:
        image_resized = resize_volume(image, target_shape=target_shape, is_label=False)
        seg_resized = resize_volume(seg_vol, target_shape=target_shape, is_label=True)
        final_shape = target_shape

    # ---- 4. label remap ----
    seg_remap = remap_labels(seg_resized, label_remap)

    # ---- 5. 保存 h5 ----
    ensure_dir(out_dir)
    out_path = os.path.join(out_dir, f"{case_id}.h5")

    with h5py.File(out_path, "w") as f:
        # 数据
        f.create_dataset(
            "image",
            data=image_resized.astype(np.float32),
            compression="gzip",
        )
        f.create_dataset(
            "label",
            data=seg_remap.astype(np.uint8),
            compression="gzip",
        )
        # 元信息作为 attribute
        f.attrs["case_id"] = case_id
        f.attrs["orig_shape"] = np.array(orig_shape, dtype=np.int32)
        f.attrs["orig_spacing"] = np.array(orig_spacing, dtype=np.float32)
        f.attrs["final_shape"] = np.array(final_shape, dtype=np.int32)
        f.attrs["resample_mode"] = resample_mode
        if target_spacing is not None:
            f.attrs["target_spacing"] = np.array(target_spacing, dtype=np.float32)
        f.attrs["norm_mode"] = norm_mode
        f.attrs["z_clip"] = float(z_clip)
        f.attrs["to_01"] = int(to_01)
        if norm_mode == "ct_window":
            f.attrs["window_center"] = float(window_center)
            f.attrs["window_width"] = float(window_width)
        elif norm_mode == "nnunet" and dataset_stats is not None:
            f.attrs["global_mean"] = float(dataset_stats['mean'])
            f.attrs["global_std"] = float(dataset_stats['std'])
            f.attrs["percentile_00_5"] = float(dataset_stats['percentile_00_5'])
            f.attrs["percentile_99_5"] = float(dataset_stats['percentile_99_5'])

    return case_id, out_path


def split_cases(case_ids, split_ratio, seed=42):
    """
    按比例随机划分 train/val/test。
    split_ratio: [r_train, r_val, r_test]，不要求精确相加为 1，会自动归一化。
    """
    ratios = np.array(split_ratio, dtype=float)
    ratios = ratios / ratios.sum()
    r_train, r_val, r_test = ratios.tolist()

    n = len(case_ids)
    np.random.seed(seed)
    idx = np.random.permutation(n)

    n_train = int(round(r_train * n))
    n_val = int(round(r_val * n))
    if n_train + n_val > n:
        n_val = n - n_train
    n_test = n - n_train - n_val

    splits = {}
    for i, j in enumerate(idx):
        cid = case_ids[j]
        if i < n_train:
            splits[cid] = "train"
        elif i < n_train + n_val:
            splits[cid] = "val"
        else:
            splits[cid] = "test"
    return splits


def build_split_csvs(root_dir: str, records: list, splits: dict):
    """
    根据 splits 将记录分别写入 train.csv / val.csv / test.csv。
    每个 CSV 列为：
      case_id, grade, volume_path, label_path
    （FLARE21没有grade，使用空字符串以保持格式一致）
    """
    fieldnames = ["case_id", "grade", "volume_path", "label_path"]

    grouped = {"train": [], "val": [], "test": []}
    for r in records:
        cid = r["case_id"]
        split = splits.get(cid, "train")
        if split not in grouped:
            continue
        grouped[split].append(
            {
                "case_id": cid,
                "grade": r.get("grade", ""),
                "volume_path": r["h5_path"],
                "label_path": r["h5_path"],
            }
        )

    for split_name, rows in grouped.items():
        csv_path = os.path.join(root_dir, f"{split_name}.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
        print(f"{split_name}.csv saved to: {csv_path} (n={len(rows)})")


def main(config_path: str):
    cfg = load_config(config_path)
    data_cfg = cfg["data"]

    img_root = data_cfg["img_root"]
    mask_root = data_cfg["mask_root"]
    preproc_root = data_cfg["preproc_root"]

    label_remap = {int(k): int(v) for k, v in data_cfg["label_remap"].items()}
    target_shape = tuple(int(x) for x in data_cfg.get("target_shape", [160, 160, 160]))
    z_clip = float(data_cfg.get("z_clip", 5.0))
    to_01 = bool(data_cfg.get("to_01", True))

    resample_mode = str(data_cfg.get("resample_mode", "shape"))
    target_spacing = data_cfg.get("target_spacing", None)
    if target_spacing is not None:
        target_spacing = tuple(float(x) for x in target_spacing)

    norm_mode = str(data_cfg.get("norm_mode", "zscore"))
    window_center = float(data_cfg.get("window_center", 50.0))
    window_width = float(data_cfg.get("window_width", 400.0))

    exclude_cases = set(data_cfg.get("exclude_cases", []))

    split_ratio = data_cfg.get("split_ratio", [0.7, 0.15, 0.15])
    split_seed = int(data_cfg.get("split_seed", 42))

    run_preprocess = bool(data_cfg.get("run_preprocess", False))

    ensure_dir(preproc_root)
    out_h5_dir = os.path.join(preproc_root, "h5")
    ensure_dir(out_h5_dir)

    # --- 扫描目录获取所有 FLARE21 case 列表 ---
    cases = scan_flare21_cases(img_root, mask_root)
    print(f"Found {len(cases)} cases")
    print(f"  Image root: {img_root}")
    print(f"  Mask root: {mask_root}")
    print(f"Resample mode: {resample_mode}")
    if resample_mode == "spacing":
        print(f"Target spacing: {target_spacing}")
    else:
        print(f"Target shape: {target_shape}")
    print(f"Normalization mode: {norm_mode}")
    if norm_mode == "ct_window":
        print(f"  Window center: {window_center} HU, Window width: {window_width} HU")
        print(f"  HU range: [{window_center - window_width/2}, {window_center + window_width/2}]")
    elif norm_mode == "nnunet":
        print("  nnU-Net style: percentile clip + global mean/std normalization")
    else:
        print(f"  z_clip: {z_clip}")
    if exclude_cases:
        print(f"Excluding cases: {sorted(exclude_cases)}")

    # 如果使用nnunet模式，先计算数据集统计量
    dataset_stats = None
    if norm_mode == "nnunet" and run_preprocess:
        valid_case_ids = []
        for c in cases:
            case_id = c["case_id"]
            # 跳过排除的案例
            if case_id in exclude_cases:
                continue
            valid_case_ids.append(case_id)
        dataset_stats = compute_dataset_statistics(
            valid_case_ids,
            img_root=img_root,
            mask_root=mask_root,
        )
        # 保存统计量到文件
        stats_path = os.path.join(preproc_root, "dataset_stats.yaml")
        with open(stats_path, "w") as f:
            yaml.dump(dataset_stats, f, default_flow_style=False)
        print(f"Dataset statistics saved to: {stats_path}")

    if not run_preprocess:
        print(
            "[INFO] run_preprocess=False, 将跳过 NIfTI->H5 的预处理步骤，只根据已有 H5 生成 CSV。"
        )

    records = []
    for idx, c in enumerate(cases):
        case_id = c["case_id"]

        # 跳过排除的案例
        if case_id in exclude_cases:
            print(f"[{idx+1}/{len(cases)}] Skipping {case_id} (excluded)")
            continue

        print(f"[{idx+1}/{len(cases)}] Processing {case_id} ...")

        if run_preprocess:
            # 完整预处理流程
            case_id2, h5_path = process_case(
                case_info=c,
                label_remap=label_remap,
                target_shape=target_shape,
                z_clip=z_clip,
                to_01=to_01,
                out_dir=out_h5_dir,
                target_spacing=target_spacing,
                resample_mode=resample_mode,
                norm_mode=norm_mode,
                window_center=window_center,
                window_width=window_width,
                dataset_stats=dataset_stats,
            )
            assert case_id2 == case_id
        else:
            # 仅使用已有 H5，不重新做预处理
            h5_path = os.path.join(out_h5_dir, f"{case_id}.h5")
            if not os.path.exists(h5_path):
                raise FileNotFoundError(
                    f"[ERROR] run_preprocess=False 但找不到对应的 H5 文件: {h5_path}\n"
                    f"请先在配置中设置 run_preprocess=True 运行一遍预处理，"
                    f"或者手动确保该 H5 存在。"
                )

        records.append({"case_id": case_id, "h5_path": h5_path, "grade": ""})

    # 按比例划分 train/val/test
    case_ids = [r["case_id"] for r in records]
    splits = split_cases(case_ids, split_ratio=split_ratio, seed=split_seed)

    # 写多个 CSV：train.csv, val.csv, test.csv
    build_split_csvs(preproc_root, records, splits)

    print("Done.")
    print(f"H5 files dir: {out_h5_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess FLARE21 to 3D HDF5 volumes"
    )
    parser.add_argument(
        "--config", type=str, required=True, help="Path to YAML config"
    )
    args = parser.parse_args()
    main(args.config)