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
):
    """
    处理单个 KiTS19 case：
    - 读入 CT 模态（imaging） + seg（segmentation）
    - 转 canonical orientation
    - per-case z-score + clip + to [0,1]
    - resize 到 target_shape
    - label remap
    - 保存 h5 文件
    返回：case_id, h5_path
    """
    case_id = os.path.basename(case_dir.rstrip("/"))

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

    # ---- 3. resize ----
    image_resized = resize_volume(image, target_shape=target_shape, is_label=False)
    seg_resized = resize_volume(seg_vol, target_shape=target_shape, is_label=True)

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
        f.attrs["target_shape"] = np.array(target_shape, dtype=np.int32)
        f.attrs["z_clip"] = float(z_clip)
        f.attrs["to_01"] = int(to_01)

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
    target_shape = tuple(int(x) for x in data_cfg["target_shape"])
    z_clip = float(data_cfg.get("z_clip", 5.0))
    to_01 = bool(data_cfg.get("to_01", True))

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
    if not run_preprocess:
        print(
            "[INFO] run_preprocess=False, 将跳过 NIfTI->H5 的预处理步骤，只根据已有 H5 生成 CSV。"
        )

    records = []
    for idx, c in enumerate(cases):
        case_id = c["case_id"]
        case_dir = c["case_dir"]
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