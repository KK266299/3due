# file: src/tools/visualize_ue.py
from __future__ import annotations

import os
import json
import argparse
import random
from typing import Any, Dict, Optional, Tuple, List

import numpy as np
import torch
from omegaconf import OmegaConf, DictConfig
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import cm

# ---- SSIM (requires scikit-image) ----
try:
    from skimage.metrics import structural_similarity as ssim_fn
except Exception as e:
    raise ImportError("This script requires scikit-image for SSIM. Please `pip install scikit-image`.") from e

# ---- project imports ----
from src.registry import get_dataset_builder
from src.utils.config import get_config, require_config
from src.utils.logger import setup_logger, get_logger
from src.core.ue_keys import extract_key
from src.core.ue_orchestrator import build_unlearnable_provider_instance
from src.core.ue_artifacts import UEShardsAccessor


# ============================ utils ============================ #
def _set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _resolve_task_for_builder(cfg: DictConfig) -> str:
    name = get_config(cfg, "ue.base_task_builder", None)
    if name:
        return str(name)
    return str(require_config(cfg, "task.name"))


def _resolve_key_spec(cfg: DictConfig, manifest_path: Optional[str]) -> Dict[str, Any]:
    kspec = get_config(cfg, "ue.key", None)
    if kspec is None:
        kspec = get_config(cfg, "training.data.poison.key",
                           OmegaConf.create({"type": "samplewise", "from": "index"}))
    return dict(OmegaConf.to_container(kspec, resolve=True))


def _build_dataset(cfg: DictConfig, split: str):
    task_for_builder = _resolve_task_for_builder(cfg)
    builder_cls = get_dataset_builder(task_for_builder)
    builder = builder_cls(cfg)
    ds = builder.get_dataset(split=split)
    return ds


def _choose_source(cfg: DictConfig, manifest_cli: Optional[str]) -> Tuple[str, Optional[str]]:
    if manifest_cli:
        return "shards", manifest_cli
    stype = str(get_config(cfg, "training.data.poison.source.type", "provider")).lower()
    if stype == "shards":
        mpath = require_config(cfg, "training.data.poison.source.manifest_path")
        return "shards", str(mpath)
    return "provider", None


def _ensure_chw_uint8(img_chw01: torch.Tensor) -> np.ndarray:
    """
    [C,H,W] float32 in [0,1] -> [H,W,3] uint8 for saving
    """
    if img_chw01.ndim != 3:
        raise ValueError(f"Expect [C,H,W], got {tuple(img_chw01.shape)}")
    c, h, w = img_chw01.shape
    x = img_chw01.clamp(0.0, 1.0).detach().cpu()
    if c == 1:
        x = x.repeat(3, 1, 1)
    elif c != 3:
        # keep the first 3 channels if >3
        x = x[:3, :, :]
        if x.shape[0] == 1:
            x = x.repeat(3, 1, 1)
    x = (x * 255.0).round().byte().permute(1, 2, 0).numpy()
    return x


def _chw01_to_hwc01(img_chw01: torch.Tensor) -> np.ndarray:
    """
    [C,H,W] float32 in [0,1] -> [H,W,C] float32 in [0,1] for SSIM
    """
    if img_chw01.ndim != 3:
        raise ValueError(f"Expect [C,H,W], got {tuple(img_chw01.shape)}")
    x = img_chw01.clamp(0.0, 1.0).detach().cpu().permute(1, 2, 0).numpy()
    return x



def _colorize_noise_bwr(noise_chw: torch.Tensor, eps_vis: float) -> np.ndarray:
    """
    [C,H,W] float -> RGB uint8
    规则：每个像素取“绝对值最大的通道”的带符号值做着色；
    颜色条范围使用 [-eps_vis,+eps_vis]；若 eps_vis<=0 则按该图自身自适应（不利于横向对比）。
    """
    if noise_chw.ndim != 3:
        raise ValueError(f"Expect [C,H,W], got {tuple(noise_chw.shape)}")

    n = noise_chw.detach().cpu().float()
    C, H, W = n.shape

    if C == 1:
        signed = n[0]                                  # [H,W]
        vmax = signed.abs().max()
    else:
        # 逐像素选绝对值最大的通道，并保留其符号（避免通道均值抵消）
        mag, idx = n.abs().max(dim=0)                  # [H,W]
        signed = n.gather(0, idx.unsqueeze(0)).squeeze(0)  # [H,W]
        vmax = mag.max()

    # 颜色范围：固定到 [-eps_vis, +eps_vis]，便于跨方法对比
    if eps_vis is None or eps_vis <= 0:
        m = float(vmax)                                # 自适应（仅单图查看时可用）
    else:
        m = float(eps_vis)
    m = max(m, 1e-12)

    vis01 = (signed / (2.0 * m) + 0.5).clamp(0.0, 1.0).numpy()
    rgba = cm.get_cmap("bwr")(vis01)                   # [H,W,4] in [0,1]
    rgb = (rgba[..., :3] * 255.0).round().astype(np.uint8)
    return rgb


def _sanitize_key_for_name(k: Any, maxlen: int = 24) -> str:
    s = str(k).replace("/", "_").replace("\\", "_").replace(":", "_")
    return (s[:maxlen] + "...") if len(s) > maxlen else s


def _resolve_vis_epsilon(
    cfg: DictConfig,
    source_type: str,
    manifest_path: Optional[str],
    vis_eps_cli: Optional[float],
) -> float:
    if vis_eps_cli is not None:
        return float(vis_eps_cli)
    if source_type == "shards" and manifest_path and os.path.isfile(manifest_path):
        try:
            with open(manifest_path, "r") as f:
                mf = json.load(f)
            scale = float(mf.get("scale"))
            if scale > 0:
                return 127.0 * scale  # int8 symmetric dequant
        except Exception:
            pass
    return float(get_config(cfg, "ue.algorithm.params.epsilon", 0.0313725))


def _psnr_01(x: torch.Tensor, y: torch.Tensor) -> float:
    """
    PSNR in [0,1] domain, per-image.
    x,y: [C,H,W] float in [0,1]
    """
    diff = (x - y).clamp(-1, 1)
    mse = float(torch.mean(diff * diff).item())
    if mse <= 1e-12:
        return float("inf")
    return 10.0 * np.log10(1.0 / mse)


def _ssim_01(x: torch.Tensor, y: torch.Tensor) -> float:
    """
    SSIM in [0,1] domain using scikit-image.
    x,y: [C,H,W] float in [0,1]
    """
    a = _chw01_to_hwc01(x)
    b = _chw01_to_hwc01(y)
    if a.shape[2] == 1:
        # single channel
        return float(ssim_fn(a[..., 0], b[..., 0], data_range=1.0))
    elif a.shape[2] == 3:
        return float(ssim_fn(a, b, data_range=1.0, channel_axis=-1))
    else:
        # for >3 channels, take first 3
        return float(ssim_fn(a[..., :3], b[..., :3], data_range=1.0, channel_axis=-1))


# ============================ core ============================ #
def _process_split(
    split: str,
    ds,
    *,
    accessor: Optional[UEShardsAccessor],
    provider,
    key_spec: Dict[str, Any],
    eps_vis: float,
    output_dir: str,
    num: int,
    seed: int,
    perturb_type: str,
) -> Tuple[List[float], List[float], int]:
    """
    Iterate one split:
      - save *_orig / *_noise_bwr / *_poisoned
      - compute per-image PSNR/SSIM in [0,1]
    Returns (psnrs, ssims, saved_count)
    """
    logger = get_logger("main")
    n_total = len(ds)
    if n_total == 0:
        logger.warning(f"[VIS] split={split} dataset is empty.")
        return [], [], 0

    # choose indices
    if num <= 0 or num >= n_total:
        indices = list(range(n_total))         # ALL
    else:
        rng = random.Random(seed)              # per split deterministic
        indices = list(range(n_total))
        rng.shuffle(indices)
        indices = indices[:int(num)]

    out_root = os.path.join(output_dir, split)
    os.makedirs(out_root, exist_ok=True)

    psnrs: List[float] = []
    ssims: List[float] = []
    saved = 0

    for rank, idx in enumerate(tqdm(indices, desc=f"visualize({split})")):
        sample = ds[idx]
        img0: torch.Tensor = sample["image"].float()   # [C,H,W] in [0,1]
        key = extract_key(sample, idx, key_spec)

        # get noise [Cn,H,W]
        if accessor is not None:
            noise = accessor.get(key, perturb_type=perturb_type)
        else:
            noise = provider.get_noise(key, perturb_type)

        # minimal channel rules
        c_i, h_i, w_i = img0.shape
        c_n, h_n, w_n = noise.shape
        if (h_n != h_i) or (w_n != w_i):
            raise ValueError(f"[{split}] HW mismatch: image={img0.shape} noise={noise.shape}")

        if c_n == c_i:
            pass  # as is
        elif c_n == 1 and c_i == 3:
            noise = noise.repeat(3, 1, 1)
        elif c_n == 3 and c_i == 1:
            raise ValueError(f"[{split}] Channel mismatch (noise=3, image=1) is not supported.")
        else:
            raise ValueError(f"[{split}] Channel mismatch: image C={c_i}, noise C={c_n}")

        img1 = (img0 + noise).clamp(0.0, 1.0)

        # save images
        key_tag = _sanitize_key_for_name(key)
        base = f"{rank:04d}_{idx:06d}_{key_tag}"
        plt.imsave(os.path.join(out_root, f"{base}_orig.png"), _ensure_chw_uint8(img0))
        plt.imsave(os.path.join(out_root, f"{base}_noise_bwr.png"), _colorize_noise_bwr(noise, eps_vis))
        plt.imsave(os.path.join(out_root, f"{base}_poisoned.png"), _ensure_chw_uint8(img1))

        # metrics (per-image, in [0,1])
        psnrs.append(_psnr_01(img0, img1))
        ssims.append(_ssim_01(img0, img1))
        saved += 1

    return psnrs, ssims, saved


def visualize_ue_multi(
    cfg: DictConfig,
    *,
    splits: List[str],
    num: int = -1,
    seed: int = 123,
    output_dir: str = "./outputs/visualize_ue",
    manifest_override: Optional[str] = None,
    vis_eps_cli: Optional[float] = None,
):
    logger = get_logger("main")
    _set_seed(int(seed))

    # build datasets for given splits
    datasets: Dict[str, Any] = {}
    for sp in splits:
        ds = _build_dataset(cfg, sp)
        datasets[sp] = ds
        logger.info(f"[VIS] split={sp} | size={len(ds)}")

    # noise source and key spec
    source_type, manifest_path = _choose_source(cfg, manifest_override)
    key_spec = _resolve_key_spec(cfg, manifest_path)
    perturb_type = str(key_spec.get("type", "samplewise")).lower()

    # unified vis epsilon
    eps_vis = _resolve_vis_epsilon(cfg, source_type, manifest_path, vis_eps_cli)

    # accessor / provider (single shared instance)
    accessor: Optional[UEShardsAccessor] = None
    provider = None
    if source_type == "shards":
        if not manifest_path or not os.path.isfile(manifest_path):
            raise FileNotFoundError(f"[VIS] manifest not found: {manifest_path}")
        accessor = UEShardsAccessor.from_manifest(manifest_path)
        p_mf = getattr(accessor, "perturb_type", None)
        if p_mf:
            perturb_type = str(p_mf)
        logger.info(f"[VIS] Using shards: {manifest_path}")
    else:
        # if both train/val exist pass them; otherwise pass the first twice
        ds_list = [datasets[sp] for sp in splits if sp in datasets]
        ds_a = ds_list[0]
        ds_b = ds_list[1] if len(ds_list) > 1 else ds_a
        provider = build_unlearnable_provider_instance(cfg, ds_a, ds_b)
        if provider is None:
            raise RuntimeError("[VIS] provider mode requested but provider_instance is None (check config).")
        logger.info(f"[VIS] Using provider: {type(provider).__name__}")

    # output root
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"[VIS] output_dir={output_dir} | eps_vis={eps_vis:.6f} | source={source_type} | type={perturb_type}")

    # run each split and collect metrics
    metrics: Dict[str, Dict[str, float]] = {}
    all_psnr: List[float] = []
    all_ssim: List[float] = []
    total_count = 0

    for sp in splits:
        ps, ss, cnt = _process_split(
            sp, datasets[sp],
            accessor=accessor, provider=provider,
            key_spec=key_spec, eps_vis=eps_vis,
            output_dir=output_dir, num=num, seed=seed, perturb_type=perturb_type
        )
        if cnt > 0:
            metrics[sp] = {
                "count": cnt,
                "psnr_mean": float(np.mean(ps)),
                "psnr_std": float(np.std(ps, ddof=0)),
                "ssim_mean": float(np.mean(ss)),
                "ssim_std": float(np.std(ss, ddof=0)),
            }
            all_psnr.extend(ps)
            all_ssim.extend(ss)
            total_count += cnt
        else:
            metrics[sp] = {"count": 0, "psnr_mean": None, "psnr_std": None, "ssim_mean": None, "ssim_std": None}

    combined = {
        "count": total_count,
        "psnr_mean": float(np.mean(all_psnr)) if total_count > 0 else None,
        "psnr_std": float(np.std(all_psnr, ddof=0)) if total_count > 0 else None,
        "ssim_mean": float(np.mean(all_ssim)) if total_count > 0 else None,
        "ssim_std": float(np.std(all_ssim, ddof=0)) if total_count > 0 else None,
    }

    summary = {
        "splits": metrics,
        "combined": combined,
        "config": {
            "eps_vis": eps_vis,
            "source": source_type,
            "perturb_type": perturb_type,
            "num_per_split": num,
            "splits": splits,
        },
    }

    with open(os.path.join(output_dir, "metrics_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"[VIS] summary saved → {os.path.join(output_dir, 'metrics_summary.json')}")


# ============================ CLI ============================ #
def main():
    parser = argparse.ArgumentParser("UE Visualization (train+val, PSNR/SSIM in [0,1])")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML/JSON config file")
    parser.add_argument("--splits", type=str, default="train,val", help="Comma-separated splits, e.g., 'train,val'")
    parser.add_argument("--num", type=int, default=-1, help="Samples per split; -1/0 means ALL (default: ALL)")
    parser.add_argument("--seed", type=int, default=123, help="Random seed for sampling when num>0")
    parser.add_argument("--output", type=str, default=None, help="Output directory")
    parser.add_argument("--manifest", type=str, default=None, help="Override manifest.json path to force shards mode")
    parser.add_argument("--vis-eps", type=float, default=None,
                        help="Visualization epsilon for noise colormap; fixes range to [-vis_eps, +vis_eps].")
    parser.add_argument("--log_file", type=str, default=None, help="Optional log file path")
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)
    output_dir = get_config(cfg, "experiment.save_dir")
    if args.output is not None:
        output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)
    setup_logger(
        name="main",
        log_file=args.log_file or os.path.join(output_dir, "visualize.log"),
        level="INFO"
    )
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]

    visualize_ue_multi(
        cfg,
        splits=splits,
        num=args.num,
        seed=args.seed,
        output_dir=output_dir,
        manifest_override=args.manifest,
        vis_eps_cli=args.vis_eps,
    )


if __name__ == "__main__":
    main()
