#!/usr/bin/env python
"""
KiTS19 数据诊断工具
用于排查tumor_dc为0的问题

使用方法:
  # 检查预处理后的H5文件
  python scripts/diagnose_kits19.py h5 --h5_dir /path/to/h5

  # 检查原始NIfTI数据
  python scripts/diagnose_kits19.py raw --raw_dir /path/to/kits19/data

  # 对比resize前后的变化（推荐）
  python scripts/diagnose_kits19.py compare --raw_dir /path/to/raw --h5_dir /path/to/h5
"""
import os
import sys
import glob
import argparse
import numpy as np
import h5py

try:
    import nibabel as nib
    from nibabel.orientations import aff2axcodes
    HAS_NIBABEL = True
except ImportError:
    HAS_NIBABEL = False

try:
    from scipy.ndimage import zoom
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


def count_labels(arr: np.ndarray):
    """统计label分布"""
    unique, counts = np.unique(arr, return_counts=True)
    return dict(zip(unique.tolist(), counts.tolist()))


# ============== H5检查 ==============
def check_h5_file(h5_path: str):
    """分析单个H5文件的label分布"""
    with h5py.File(h5_path, "r") as f:
        label = f["label"][()]
        image = f["image"][()]

    label_counts = count_labels(label)
    total = label.size

    return {
        "case_id": os.path.basename(h5_path).replace(".h5", ""),
        "label_shape": label.shape,
        "image_shape": image.shape,
        "label_counts": label_counts,
        "has_tumor": 2 in label_counts,
        "tumor_voxels": label_counts.get(2, 0),
        "tumor_ratio": label_counts.get(2, 0) / total * 100,
    }


def cmd_h5(args):
    """检查H5文件"""
    h5_files = sorted(glob.glob(os.path.join(args.h5_dir, "*.h5")))
    if not h5_files:
        print(f"[ERROR] No H5 files found in: {args.h5_dir}")
        return

    print(f"Found {len(h5_files)} H5 files")
    print("=" * 80)

    has_tumor_count = 0
    no_tumor_cases = []

    for h5_path in h5_files:
        r = check_h5_file(h5_path)
        if r["has_tumor"]:
            has_tumor_count += 1
        else:
            no_tumor_cases.append(r["case_id"])

        if args.verbose or not r["has_tumor"]:
            status = "✓" if r["has_tumor"] else "✗"
            print(f"{status} {r['case_id']}: tumor={r['tumor_voxels']:,} ({r['tumor_ratio']:.3f}%)")

    print("\n" + "=" * 80)
    print(f"有tumor: {has_tumor_count}/{len(h5_files)} ({has_tumor_count/len(h5_files)*100:.1f}%)")
    print(f"无tumor: {len(no_tumor_cases)}")
    if no_tumor_cases and len(no_tumor_cases) <= 20:
        print(f"无tumor案例: {', '.join(no_tumor_cases)}")


# ============== 原始数据检查 ==============
def check_raw_label(case_dir: str):
    """分析原始NIfTI分割"""
    if not HAS_NIBABEL:
        raise ImportError("需要安装nibabel: pip install nibabel")

    seg_files = glob.glob(os.path.join(case_dir, "*segmentation*.nii*"))
    if not seg_files:
        return None

    nii = nib.load(seg_files[0])
    canonical = nib.as_closest_canonical(nii)
    data = canonical.get_fdata().astype(np.int16)
    label_counts = count_labels(data)

    return {
        "case_id": os.path.basename(case_dir),
        "shape": data.shape,
        "label_counts": label_counts,
        "has_tumor": 2 in label_counts,
        "tumor_voxels": label_counts.get(2, 0),
    }


def cmd_raw(args):
    """检查原始数据"""
    if not HAS_NIBABEL:
        print("[ERROR] 需要安装nibabel: pip install nibabel")
        return

    case_dirs = sorted(glob.glob(os.path.join(args.raw_dir, "case_*")))
    if not case_dirs:
        print(f"[ERROR] No case directories found in: {args.raw_dir}")
        return

    print(f"Found {len(case_dirs)} cases")
    print("=" * 80)

    has_tumor_count = 0
    no_tumor_cases = []

    for case_dir in case_dirs:
        r = check_raw_label(case_dir)
        if r is None:
            continue
        if r["has_tumor"]:
            has_tumor_count += 1
        else:
            no_tumor_cases.append(r["case_id"])

        if args.verbose:
            status = "✓" if r["has_tumor"] else "✗"
            print(f"{status} {r['case_id']}: shape={r['shape']}, tumor={r['tumor_voxels']:,}")

    print("\n" + "=" * 80)
    print(f"有tumor: {has_tumor_count}/{len(case_dirs)}")
    print(f"无tumor: {len(no_tumor_cases)}")


# ============== 对比检查 ==============
def simulate_resize(raw_label: np.ndarray, target_shape: tuple):
    """模拟resize过程"""
    if not HAS_SCIPY:
        raise ImportError("需要安装scipy: pip install scipy")
    h, w, d = raw_label.shape
    th, tw, td = target_shape
    zoom_factors = (th / h, tw / w, td / d)
    resized = zoom(raw_label, zoom_factors, order=0)  # order=0 = 最近邻
    return resized.astype(raw_label.dtype)


def cmd_compare(args):
    """对比resize前后"""
    if not HAS_NIBABEL or not HAS_SCIPY:
        print("[ERROR] 需要安装: pip install nibabel scipy")
        return

    target_shape = tuple(args.target_shape)
    case_dirs = sorted(glob.glob(os.path.join(args.raw_dir, "case_*")))

    print(f"对比 {len(case_dirs)} 个案例, target_shape={target_shape}")
    print("=" * 100)

    issues = []

    for case_dir in case_dirs:
        case_id = os.path.basename(case_dir)
        h5_path = os.path.join(args.h5_dir, f"{case_id}.h5")

        if not os.path.exists(h5_path):
            continue

        # 原始数据
        raw_info = check_raw_label(case_dir)
        if raw_info is None:
            continue

        # H5数据
        h5_info = check_h5_file(h5_path)

        # 模拟resize
        seg_files = glob.glob(os.path.join(case_dir, "*segmentation*.nii*"))
        nii = nib.load(seg_files[0])
        raw_label = nib.as_closest_canonical(nii).get_fdata().astype(np.int16)
        simulated = simulate_resize(raw_label, target_shape)
        sim_counts = count_labels(simulated)
        sim_has_tumor = 2 in sim_counts
        sim_tumor = sim_counts.get(2, 0)

        # 检查问题
        issue = None
        if raw_info["has_tumor"] and not sim_has_tumor:
            issue = "Resize导致tumor丢失"
        elif raw_info["has_tumor"] and not h5_info["has_tumor"]:
            issue = "H5中tumor丢失"
        elif abs(sim_tumor - h5_info["tumor_voxels"]) > 100:
            issue = f"体素数差异大: sim={sim_tumor}, h5={h5_info['tumor_voxels']}"

        if issue:
            issues.append({"case_id": case_id, "issue": issue})
            print(f"✗ {case_id}: {issue}")
            print(f"    原始: tumor={raw_info['tumor_voxels']:,}")
            print(f"    模拟: tumor={sim_tumor:,}")
            print(f"    H5:   tumor={h5_info['tumor_voxels']:,}")
        elif args.verbose:
            print(f"✓ {case_id}: OK")

    print("\n" + "=" * 100)
    if issues:
        print(f"发现 {len(issues)} 个问题案例")
        resize_loss = sum(1 for i in issues if "Resize" in i["issue"])
        if resize_loss > 0:
            print(f"[!] {resize_loss}个案例因resize丢失tumor，考虑使用更大的target_shape")
    else:
        print("所有案例正常!")


# ============== 主入口 ==============
def main():
    parser = argparse.ArgumentParser(description="KiTS19数据诊断工具")
    subparsers = parser.add_subparsers(dest="cmd", help="命令")

    # h5子命令
    p_h5 = subparsers.add_parser("h5", help="检查预处理后的H5文件")
    p_h5.add_argument("--h5_dir", required=True, help="H5文件目录")
    p_h5.add_argument("-v", "--verbose", action="store_true", help="显示所有案例")

    # raw子命令
    p_raw = subparsers.add_parser("raw", help="检查原始NIfTI数据")
    p_raw.add_argument("--raw_dir", required=True, help="原始数据目录")
    p_raw.add_argument("-v", "--verbose", action="store_true", help="显示所有案例")

    # compare子命令
    p_cmp = subparsers.add_parser("compare", help="对比resize前后")
    p_cmp.add_argument("--raw_dir", required=True, help="原始数据目录")
    p_cmp.add_argument("--h5_dir", required=True, help="H5文件目录")
    p_cmp.add_argument("--target_shape", type=int, nargs=3, default=[160, 160, 160])
    p_cmp.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args()

    if args.cmd == "h5":
        cmd_h5(args)
    elif args.cmd == "raw":
        cmd_raw(args)
    elif args.cmd == "compare":
        cmd_compare(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()