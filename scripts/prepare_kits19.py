# KiTS19 数据预处理脚本
# 数据结构: data/case_00000/imaging.nii.gz + segmentation.nii.gz
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


def find_modality_file(case_dir: str, keyword: str) -> str:
    """
    在 case_dir 底下找到包含 keyword 的 NIfTI 文件。
    例如 keyword='imaging' 时匹配 '*imaging*.nii*'
    """
    patterns = [
        os.path.join(case_dir, f"*{keyword}*.nii"),
        os.path.join(case_dir, f"*{keyword}*.nii.gz"),
    ]
    for p in patterns:
        files = glob.glob(p)
        if len(files) > 0:
            return files[0]
    raise FileNotFoundError(f"Cannot find modality '{keyword}' in {case_dir}")


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
    # （BraTS19 MRI 使用 > 0，CT背景通常是 -1000 HU）
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

    # 与BraTS19一致：只对前景区域归一化，背景保持为0
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
    这是CT图像的标准处理方式，比z-score更适合CT。

    vol: 3D array, (H, W, D), HU values
    window_center: 窗位 (Window Level), 默认50 HU（软组织窗）
    window_width: 窗宽 (Window Width), 默认400 HU
    to_01: 是否映射到[0,1]，否则映射到[-1,1]

    常用窗宽窗位参考:
      - 软组织窗: WL=40-50, WW=350-400 (适合观察肾脏/肿瘤)
      - 腹部窗: WL=40, WW=400
      - 肝脏窗: WL=60, WW=150

    参考: nnU-Net, motokimura/kits19_3d_segmentation
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
    参考: https://github.com/MIC-DKFZ/nnUNet

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
        to_01: 是否映射到[0,1]，与BraTS19保持一致
        z_clip: z-score clip范围，与BraTS19保持一致（默认5.0）

    Returns:
        归一化后的体积
    """
    vol = vol.astype(np.float32, copy=True)
    np.clip(vol, lower_bound, upper_bound, out=vol)
    vol -= global_mean
    vol /= max(global_std, 1e-8)

    if to_01:
        # 与BraTS19保持一致：clip到[-z_clip, z_clip]后映射到[0,1]
        vol = np.clip(vol, -z_clip, z_clip)
        vol = (vol + z_clip) / (2.0 * z_clip)
        vol = np.clip(vol, 0.0, 1.0)

    return vol


def compute_dataset_statistics(
    case_dirs: list,
    modality: str = "imaging",
    use_mask: bool = True,
) -> dict:
    """
    计算整个数据集的统计量，用于nnU-Net风格的归一化。

    参考nnU-Net文档:
    "collect intensity values from the foreground classes (all but the
    background and ignore) from all training cases"

    Args:
        case_dirs: case目录列表
        modality: 模态关键词
        use_mask: 是否使用分割标签mask定义前景（nnU-Net标准做法）
                  True: 使用 seg > 0 作为前景（推荐）
                  False: 使用 vol > -500 作为前景（简化方式）

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

    for i, case_dir in enumerate(case_dirs):
        if isinstance(case_dir, dict):
            case_dir = case_dir["case_dir"]

        try:
            # 加载影像
            img_path = find_modality_file(case_dir, modality)
            vol, _, _, _ = load_nifti_as_canonical(img_path)

            # 确定前景mask
            if use_mask:
                # nnU-Net标准做法：使用分割标签定义前景
                seg_path = find_modality_file(case_dir, "segmentation")
                seg, _, _, _ = load_nifti_as_canonical(seg_path)
                seg = seg.astype(np.int16)
                # 前景 = 所有非背景类（seg > 0）
                foreground_mask = seg > 0
            else:
                # 简化方式：使用HU阈值
                foreground_mask = vol > -500

            if np.sum(foreground_mask) > 0:
                # 为了节省内存，只采样一部分值
                fg_vals = vol[foreground_mask].flatten()
                # 随机采样最多100000个值
                if len(fg_vals) > 100000:
                    idx = np.random.choice(len(fg_vals), 100000, replace=False)
                    fg_vals = fg_vals[idx]
                all_foreground_values.append(fg_vals)

            if (i + 1) % 20 == 0:
                print(f"  Processed {i + 1}/{len(case_dirs)} cases...")

        except Exception as e:
            print(f"  [WARN] Failed to process {case_dir}: {e}")
            continue

    if not all_foreground_values:
        raise ValueError("No valid foreground values found in dataset!")

    # 合并所有值
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


def get_kidney_bbox(
    seg: np.ndarray,
    kidney_label: int = 1,
    tumor_label: int = 2,
    padding: int = 10,
) -> tuple:
    """
    获取肾脏区域（包含肿瘤）的 bounding box。

    Args:
        seg: 3D label map (H, W, D)
        kidney_label: 肾脏标签值
        tumor_label: 肿瘤标签值
        padding: bounding box 外扩的像素数

    Returns:
        (h_start, h_end, w_start, w_end, d_start, d_end) 或 None（如果没有肾脏）
    """
    # 肾脏区域 = kidney + tumor（因为肿瘤在肾脏内部）
    kidney_mask = (seg == kidney_label) | (seg == tumor_label)

    if not np.any(kidney_mask):
        return None

    # 找到每个维度的非零范围
    coords = np.where(kidney_mask)
    h_min, h_max = coords[0].min(), coords[0].max()
    w_min, w_max = coords[1].min(), coords[1].max()
    d_min, d_max = coords[2].min(), coords[2].max()

    # 添加 padding
    h_start = max(0, h_min - padding)
    h_end = min(seg.shape[0], h_max + 1 + padding)
    w_start = max(0, w_min - padding)
    w_end = min(seg.shape[1], w_max + 1 + padding)
    d_start = max(0, d_min - padding)
    d_end = min(seg.shape[2], d_max + 1 + padding)

    return (h_start, h_end, w_start, w_end, d_start, d_end)


def crop_to_bbox(
    vol: np.ndarray,
    bbox: tuple,
) -> np.ndarray:
    """
    根据 bounding box 裁剪体积。

    Args:
        vol: 3D array (H, W, D) 或 4D array (C, H, W, D)
        bbox: (h_start, h_end, w_start, w_end, d_start, d_end)

    Returns:
        裁剪后的体积
    """
    h_start, h_end, w_start, w_end, d_start, d_end = bbox

    if vol.ndim == 4:
        # (C, H, W, D)
        return vol[:, h_start:h_end, w_start:w_end, d_start:d_end]
    elif vol.ndim == 3:
        # (H, W, D)
        return vol[h_start:h_end, w_start:w_end, d_start:d_end]
    else:
        raise ValueError(f"Unsupported volume ndim={vol.ndim}")


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
    基于spacing的重采样（参考 motokimura/kits19_3d_segmentation）。

    vol: (H, W, D) 或 (C, H, W, D)
    orig_spacing: 原始spacing (sz, sy, sx) 或 (sy, sx, sz) 取决于数据
    target_spacing: 目标spacing，如 [3.22, 1.62, 1.62]
    is_label: 如果是label则使用最近邻插值

    Returns:
        重采样后的体积
    """
    orig_spacing = np.array(orig_spacing, dtype=np.float64)
    target_spacing = np.array(target_spacing, dtype=np.float64)

    if vol.ndim == 4:
        # (C, H, W, D)
        spatial_shape = np.array(vol.shape[1:])
        zoom_factors_spatial = orig_spacing / target_spacing
        zoom_factors = np.array([1.0] + zoom_factors_spatial.tolist())
    elif vol.ndim == 3:
        # (H, W, D)
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
    mapping: dict, e.g. {0:0, 1:1, 2:2}
    """
    seg_remap = np.zeros_like(seg, dtype=np.uint8)
    for src, dst in mapping.items():
        seg_remap[seg == int(src)] = int(dst)
    return seg_remap


def scan_case_dirs(raw_root: str) -> list:
    """
    扫描 KiTS19 数据目录，返回所有 case 目录列表。
    KiTS19 结构：raw_root/case_00000/, raw_root/case_00001/, ...
    """
    cases = []
    # 查找所有 case_xxxxx 目录
    pattern = os.path.join(raw_root, "case_*")
    case_dirs = sorted(glob.glob(pattern))

    for case_dir in case_dirs:
        if os.path.isdir(case_dir):
            case_id = os.path.basename(case_dir)
            # 确保目录中有 imaging 和 segmentation 文件
            try:
                find_modality_file(case_dir, "imaging")
                find_modality_file(case_dir, "segmentation")
                cases.append({
                    "case_id": case_id,
                    "case_dir": case_dir,
                })
            except FileNotFoundError:
                print(f"[WARN] Skipping {case_id}: missing imaging or segmentation file")
                continue

    return cases


def process_case(
    case_dir: str,
    modalities: list,
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
    crop_kidney: bool = False,
    crop_padding: int = 10,
):
    """
    处理单个 KiTS19 case：
    - 读入 CT 模态（imaging） + seg（segmentation）
    - 转 canonical orientation
    - (可选) 按肾脏区域 crop
    - 归一化（支持多种方式）
    - resize/resample
    - label remap
    - 保存 h5 文件

    Args:
        resample_mode: "shape" (固定尺寸) 或 "spacing" (基于spacing，参考nnU-Net)
        target_spacing: 当 resample_mode="spacing" 时使用，如 [3.22, 1.62, 1.62]
        norm_mode: 归一化模式
            - "zscore": 原方式，per-case z-score（与BraTS19一致）
            - "ct_window": CT窗宽窗位
            - "nnunet": nnU-Net风格（分位数clip + 全局均值/标准差）
        window_center: 当 norm_mode="ct_window" 时使用，窗位 (默认50 HU)
        window_width: 当 norm_mode="ct_window" 时使用，窗宽 (默认400 HU)
        dataset_stats: 当 norm_mode="nnunet" 时使用，包含全局统计量的dict
        crop_kidney: 是否先按肾脏 mask crop 出 ROI 区域
        crop_padding: crop 时外扩的像素数

    返回：case_id, h5_path
    """
    case_id = os.path.basename(case_dir.rstrip("/"))

    # ---- 0. 如果需要 crop_kidney，先加载 seg 获取 bbox ----
    kidney_bbox = None
    if crop_kidney:
        seg_path = find_modality_file(case_dir, "segmentation")
        seg_raw, _, _, _ = load_nifti_as_canonical(seg_path)
        seg_raw = seg_raw.astype(np.int16)
        kidney_bbox = get_kidney_bbox(seg_raw, kidney_label=1, tumor_label=2, padding=crop_padding)
        if kidney_bbox is None:
            print(f"  [WARN] {case_id}: No kidney found, skipping crop")

    # ---- 1. 加载模态（KiTS19通常只有imaging一个模态） ----
    img_list = []
    canonical_affine = None
    orig_shape = None
    orig_spacing = None

    for m in modalities:
        path = find_modality_file(case_dir, m)
        vol, affine, zooms, axcodes = load_nifti_as_canonical(path)

        if canonical_affine is None:
            canonical_affine = affine
            orig_shape = vol.shape
            orig_spacing = zooms
        else:
            if vol.shape != orig_shape:
                raise ValueError(
                    f"Modality {m} shape {vol.shape} != {orig_shape} in case {case_id}"
                )

        # 如果需要 crop，先在原始数据上 crop
        if crop_kidney and kidney_bbox is not None:
            vol = crop_to_bbox(vol, kidney_bbox)

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
                z_clip=z_clip,  # 与BraTS19保持一致
            )
        elif norm_mode == "ct_window":
            vol_norm = ct_window_normalize(
                vol,
                window_center=window_center,
                window_width=window_width,
                to_01=to_01,
            )
        else:
            # 默认使用zscore（与BraTS19一致）
            vol_norm = zscore_and_to01_per_modality(vol, z_clip=z_clip, to_01=to_01)
        img_list.append(vol_norm)

    image = np.stack(img_list, axis=0).astype(np.float32)  # (C, H, W, D)

    # ---- 2. 加载 seg ----
    seg_path = find_modality_file(case_dir, "segmentation")
    seg_vol, seg_affine, _, _ = load_nifti_as_canonical(seg_path)
    seg_vol = seg_vol.astype(np.int16)

    if seg_vol.shape != orig_shape:
        raise ValueError(
            f"Seg shape {seg_vol.shape} != image shape {orig_shape} in case {case_id}"
        )

    # 如果需要 crop，对 seg 也进行 crop
    if crop_kidney and kidney_bbox is not None:
        seg_vol = crop_to_bbox(seg_vol, kidney_bbox)

    # 记录 crop 后的尺寸（用于调试）
    cropped_shape = seg_vol.shape if (crop_kidney and kidney_bbox is not None) else None

    # ---- 3. resize/resample ----
    if resample_mode == "spacing" and target_spacing is not None:
        # 基于spacing的重采样（参考 motokimura/kits19_3d_segmentation）
        image_resized = resample_to_spacing(
            image, orig_spacing, target_spacing, is_label=False
        )
        seg_resized = resample_to_spacing(
            seg_vol, orig_spacing, target_spacing, is_label=True
        )
        final_shape = image_resized.shape[1:]  # (H, W, D)
    else:
        # 固定尺寸resize
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
        # crop 信息
        f.attrs["crop_kidney"] = int(crop_kidney)
        if crop_kidney and kidney_bbox is not None:
            f.attrs["kidney_bbox"] = np.array(kidney_bbox, dtype=np.int32)
            f.attrs["cropped_shape"] = np.array(cropped_shape, dtype=np.int32)

    return case_id, out_path


def split_cases(case_ids, split_ratio, seed=42):
    """
    按比例随机划分 train/val/test。
    split_ratio: [r_train, r_val, r_test]，不要求精确相加为 1，会自动归一化。
    """
    ratios = np.array(split_ratio, dtype=float)
    ratios = ratios / ratios.sum()  # 归一化
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
    （KiTS19没有grade，使用空字符串以保持格式一致）
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

    raw_root = data_cfg["raw_root"]
    preproc_root = data_cfg["preproc_root"]

    modalities = data_cfg["modalities"]
    label_remap = {int(k): int(v) for k, v in data_cfg["label_remap"].items()}
    target_shape = tuple(int(x) for x in data_cfg.get("target_shape", [160, 160, 160]))
    z_clip = float(data_cfg.get("z_clip", 5.0))
    to_01 = bool(data_cfg.get("to_01", True))

    # 重采样模式：shape (固定尺寸) 或 spacing (基于spacing，参考nnU-Net)
    resample_mode = str(data_cfg.get("resample_mode", "shape"))
    # 目标spacing，参考 motokimura/kits19_3d_segmentation 使用 [3.22, 1.62, 1.62]
    target_spacing = data_cfg.get("target_spacing", None)
    if target_spacing is not None:
        target_spacing = tuple(float(x) for x in target_spacing)

    # 归一化模式：zscore (与BraTS19一致) 或 ct_window (CT窗宽窗位，推荐用于CT)
    norm_mode = str(data_cfg.get("norm_mode", "zscore"))
    # 当 norm_mode=ct_window 时使用的窗宽窗位参数
    window_center = float(data_cfg.get("window_center", 50.0))  # 窗位
    window_width = float(data_cfg.get("window_width", 400.0))   # 窗宽

    # 排除的案例（参考仓库排除了一些有问题的案例）
    exclude_cases = set(data_cfg.get("exclude_cases", []))

    # 肾脏区域 crop 配置
    crop_kidney = bool(data_cfg.get("crop_kidney", False))
    crop_padding = int(data_cfg.get("crop_padding", 10))

    split_ratio = data_cfg.get("split_ratio", [0.7, 0.15, 0.15])
    split_seed = int(data_cfg.get("split_seed", 42))

    # 是否真正做预处理（默认 False -> 跳过预处理）
    run_preprocess = bool(data_cfg.get("run_preprocess", False))

    ensure_dir(preproc_root)
    out_h5_dir = os.path.join(preproc_root, "h5")
    ensure_dir(out_h5_dir)

    # --- 扫描目录获取所有 KiTS19 case 列表 ---
    cases = scan_case_dirs(raw_root)
    print(f"Found {len(cases)} cases in: {raw_root}")
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
    if crop_kidney:
        print(f"Kidney cropping: enabled (padding={crop_padding})")
    if exclude_cases:
        print(f"Excluding cases: {sorted(exclude_cases)}")

    # 如果使用nnunet模式，先计算数据集统计量
    dataset_stats = None
    if norm_mode == "nnunet" and run_preprocess:
        # 过滤掉排除的案例
        valid_cases = []
        for c in cases:
            case_num = int(c["case_id"].replace("case_", ""))
            if case_num not in exclude_cases:
                valid_cases.append(c)
        dataset_stats = compute_dataset_statistics(
            valid_cases,
            modality=modalities[0] if modalities else "imaging",
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
        case_dir = c["case_dir"]

        # 跳过排除的案例
        case_num = int(case_id.replace("case_", ""))
        if case_num in exclude_cases:
            print(f"[{idx+1}/{len(cases)}] Skipping {case_id} (excluded)")
            continue

        print(f"[{idx+1}/{len(cases)}] Processing {case_id} ...")

        if run_preprocess:
            # 完整预处理流程
            case_id2, h5_path = process_case(
                case_dir=case_dir,
                modalities=modalities,
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
                crop_kidney=crop_kidney,
                crop_padding=crop_padding,
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
        description="Preprocess KiTS19 to 3D HDF5 volumes"
    )
    parser.add_argument(
        "--config", type=str, required=True, help="Path to YAML config"
    )
    args = parser.parse_args()
    main(args.config)