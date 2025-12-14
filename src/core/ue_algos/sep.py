# file: src/core/ue_algos/sep.py
from __future__ import annotations
from typing import Dict, Iterable, List, Tuple, Any, Hashable
from omegaconf import ListConfig
import numpy as np
import torch
import torch.nn.functional as F

from ...registry import register_plugin
from ...utils.config import get_config, require_config
from torch.utils.data import DataLoader, ConcatDataset, SequentialSampler
from tqdm import tqdm

# -------- canonicalizers --------
def _canon_val(v: Any) -> Hashable:
    """
    Canonicalize scalar tensor / numpy scalar to python scalar; keep strings as-is.
    - Tensor(0-dim) -> item()
    - np.integer -> int, np.floating -> float
    - Tensor(n>0) -> tuple of list (rare)
    """
    if torch.is_tensor(v):
        if v.ndim == 0:
            return v.item()
        return tuple(np.asarray(v.cpu()).reshape(-1).tolist())
    if isinstance(v, (np.integer,)):
        return int(v.item())
    if isinstance(v, (np.floating,)):
        return float(v.item())
    return v

def _pbar(iterable, *, enabled: bool=True, desc: str) -> Iterable:
    return tqdm(iterable, desc=desc, leave=False) if enabled else iterable

@register_plugin("sep")
class SepFaVrUE:
    """
    SEP-FA-VR（Self-Ensemble Protection, Feature-Alignment with Variance Reduction）

    关键设计：
      - Y 空间（类标签，用于中心与错类索引） == batch["targets"]["reid"]
      - K 空间（噪声键，用于噪声表读写/聚合/像素盒） == batch["key"]
      - 两者严格解耦，杜绝再用噪声键字典查类索引的做法（避免 KeyError: <label>）
    """

    # -------------------- helpers -------------------- #
    @staticmethod
    def _norm_inplace(x: torch.Tensor, mean: Tuple[float, ...], std: Tuple[float, ...]) -> torch.Tensor:
        for c, (m, s) in enumerate(zip(mean, std)):
            x[:, c].sub_(float(m)).div_(float(s))
        return x

    def _build_full_train_loader(self, trainer):
        ds_all = getattr(trainer.train_loader, "dataset", None)
        if ds_all is None:
            raise RuntimeError("[UE][SEP] trainer.train_loader.dataset is None")

        bs = int(get_config(trainer.config, "training.eval_batch_size",
                            get_config(trainer.config, "training.batch_size", 64)))
        nw = int(get_config(trainer.config, "training.num_workers", 0))

        # 顺序采样，确保“全量 + 可重复一致”
        sam = SequentialSampler(ds_all)
        loader = DataLoader(ds_all, batch_size=bs, sampler=sam,
                            shuffle=False, drop_last=False, num_workers=nw, pin_memory=True)
        return loader

    def _collect_train_labels(self) -> List[Hashable]:
        """
        扫描 train_loader，收集 reid 类标签全集（Y 空间），返回去重+排序后的列表（稳定）。
        """
        loader = self._full_train_loader
        seen: set[Hashable] = set()
        for batch in loader:
            y = batch["targets"]["reid"]
            if torch.is_tensor(y):
                ys = y.detach().cpu().tolist()
            else:
                ys = list(y)
            for vv in ys:
                seen.add(_canon_val(vv))
        # 排序：按 str 表示排序，确保跨类型也稳定（int/str 混合等场景）
        ids_sorted = sorted(seen, key=lambda k: str(k))
        return ids_sorted

    @torch.no_grad()
    def _infer_feat_dim(self, model, device, mean, std) -> int:
        loader = self._full_train_loader
        model.eval()
        for batch in loader:
            x = batch["image"].to(device).float().clamp(0, 1)
            xn = x.clone()
            self._norm_inplace(xn, mean, std)
            out = model(xn)
            emb = out[1] if isinstance(out, (tuple, list)) else out
            return int(emb.shape[-1])
        raise RuntimeError("[UE][SEP] Empty train loader.")

    @torch.no_grad()
    def _compute_centers_for_ckpt(
        self,
        model,
        device,
        mean,
        std,
        cls2idx: Dict[Hashable, int],
        C: int,
        D: int
    ) -> torch.Tensor:
        """
        严格全量通扫，基于 Y 空间（reid）累加求均值 -> 返回 [C, D]（存 CPU）。
        绝不使用 batch["key"] / 噪声键体系。
        """
        loader = self._full_train_loader

        centers_sum = torch.zeros(C, D, dtype=torch.float32, device="cpu")
        counts = torch.zeros(C, dtype=torch.long, device="cpu")
        model.eval()

        for batch in _pbar(loader, desc="Compute Centers"):
            x = batch["image"].to(device).float().clamp(0, 1)
            y = batch["targets"]["reid"]

            if torch.is_tensor(y):
                ys = y.detach().cpu().tolist()
            else:
                ys = list(y)
            y_keys = [_canon_val(v) for v in ys]

            xn = x.clone()
            self._norm_inplace(xn, mean, std)
            out = model(xn)
            emb = out[1] if isinstance(out, (tuple, list)) else out  # [N, D]
            emb_cpu = emb.detach().to("cpu")

            try:
                idx = torch.tensor([cls2idx[k] for k in y_keys], dtype=torch.long)
            except KeyError as e:
                raise ValueError(f"[UE][SEP] Unknown class label during center computation: {e}")

            centers_sum.index_add_(0, idx, emb_cpu)
            counts.index_add_(0, idx, torch.ones_like(idx, dtype=torch.long))

        counts = counts.clamp_min(1).to(torch.float32).unsqueeze(1)
        centers = centers_sum / counts
        return centers  # [C, D] (CPU)

    # -------------------- runtime state -------------------- #
    def __init__(self):
        # Y 空间（类标签）映射
        self.cls_sorted: List[Hashable] | None = None
        self.cls2idx: Dict[Hashable, int] | None = None
        self.C: int | None = None
        self.D: int | None = None
        self.K_offset: int | None = None

        # 自集成权重
        self.ckpt_paths: List[str] | None = None
        self.ckpt_states: List[Dict[str, torch.Tensor]] | None = None  # state_dict on CPU
        self.centers_per_ckpt: List[torch.Tensor] | None = None        # list of [C, D] on CPU

        # 模型与归一化
        self.s_model = None
        self.mean: Tuple[float, ...] | None = None
        self.std: Tuple[float, ...] | None = None

    def surrogate_step_batch(self, trainer, batch) -> Dict[str, float]:
        # SEP-FA-VR 无需额外训练代理，保留接口以符合 UETrainer 双环
        self._maybe_init(trainer)
        return {"surrogate_loss": 0.0, "loss": 0.0}

    def _maybe_init(self, trainer) -> None:
        if self.cls2idx is not None:
            return
        self._full_train_loader = self._build_full_train_loader(trainer)

        cfg = trainer.config
        device = trainer.device
        nb = trainer.noise_backend
        if nb is None:
            raise RuntimeError("[UE][SEP] noise_backend is required (for noise I/O).")

        # 归一化统计
        self.mean = tuple(get_config(cfg, "training.data.transforms.mean", (0.485, 0.456, 0.406)))
        self.std  = tuple(get_config(cfg, "training.data.transforms.std",  (0.229, 0.224, 0.225)))

        # surrogate 模型（任取第一个）
        if not trainer.surrogates:
            raise RuntimeError("[UE][SEP] No surrogate bound.")
        _, mdl = next(iter(trainer.surrogates.items()))
        self.s_model = mdl.to(device)

        # ckpt 列表（接受 list 或 ListConfig）
        ckpt_paths = (
            get_config(cfg, f"ue.surrogates.{list(trainer.surrogates.keys())[0]}.ckpt_paths", None)
            or get_config(cfg, "ue.sep.ckpt_paths", None)
        )
        if ckpt_paths is None or not isinstance(ckpt_paths, (list, ListConfig)) or len(ckpt_paths) == 0:
            raise RuntimeError("[UE][SEP] Please provide ckpt paths at ue.surrogates.<name>.ckpt_paths or ue.sep.ckpt_paths.")
        self.ckpt_paths = list(ckpt_paths)

        # === 关键：构建 Y 空间（reid）的全量映射 ===
        self.cls_sorted = self._collect_train_labels()               # 全集（去重+排序）
        self.cls2idx = {k: i for i, k in enumerate(self.cls_sorted)}
        self.C = len(self.cls_sorted)

        # 特征维度 D
        self.D = self._infer_feat_dim(self.s_model, device, self.mean, self.std)

        # 错类偏移（半环，避免自指；C==1 时退化为 0）
        self.K_offset = 0 if self.C == 1 else max(1, self.C // 2)

        # 预加载 ckpt 的 state_dict（CPU）
        self.ckpt_states = []
        for p in self.ckpt_paths:
            sd = torch.load(p, map_location="cpu", weights_only=False)
            if isinstance(sd, dict):
                if "state_dict" in sd:
                    state = sd["state_dict"]
                elif "net" in sd:
                    state = sd["net"]
                elif "model" in sd:
                    state = sd["model"]
                else:
                    state = sd
            else:
                raise RuntimeError(f"[UE][SEP] Unexpected ckpt format: {p}")
            self.ckpt_states.append(state)

        # 为每个 ckpt 计算类中心表（CPU）
        self.centers_per_ckpt = []
        for state in self.ckpt_states:
            self.s_model.load_state_dict(state, strict=False)
            centers_k = self._compute_centers_for_ckpt(
                self.s_model.to(device).eval(), device, self.mean, self.std,
                self.cls2idx, int(self.C), int(self.D)
            )
            self.centers_per_ckpt.append(centers_k)  # [C, D] on CPU

    # -------------------- N-step：Update noise (PGD descent, class-wise on Y; per-key I/O on K) -------------------- #
    def noise_step_batch(self, trainer, batch) -> Dict[str, float]:
        self._maybe_init(trainer)

        cfg = trainer.config
        device = trainer.device
        nb = trainer.noise_backend
        if nb is None:
            raise RuntimeError("[UE][SEP] noise_backend is required.")

        # ---- batch data ----
        x = batch["image"].to(device).float()                 # [N, C_in, H, W]
        # Y：类标签（用于中心/错类）
        y_raw = batch["targets"]["reid"]
        y_list: List[Hashable] = (y_raw.detach().cpu().tolist() if torch.is_tensor(y_raw) else list(y_raw))
        y_keys = [_canon_val(v) for v in y_list]
        # K：噪声键（用于噪声表操作/聚合）
        k_list: List[Any] = list(batch["key"])

        # ---- cfg ----
        algo = require_config(cfg, "ue.algorithm")
        params = require_config(algo, "params")
        eps = float(get_config(params, "epsilon", 2/255.0))
        step_size = float(get_config(params, "step_size", 1/255.0))
        T = int(get_config(params, "noise_step", 30))           # 外层迭代
        M = int(get_config(params, "svrg_M", 15))               # 内层 SVRG
        tied_channels = bool(get_config(params, "tied_channels", False))

        mean, std = self.mean, self.std

        # freeze surrogate
        for p in self.s_model.parameters():
            p.requires_grad = False

        # ---- noise table (per-key) ----
        delta_tbl = nb.batch_noise(k_list).to(device)          # [N, C_table, H, W]
        N, C_in, H, W = x.shape
        C_table = delta_tbl.size(1)

        # channel policy
        repeat_to_Cin = (C_table == 1 and C_in > 1)
        if not repeat_to_Cin and C_table != C_in:
            raise RuntimeError(f"[UE][SEP] channel mismatch: table C={C_table} vs input C={C_in}")
        fold_even_if_multi = bool(tied_channels)

        # ---- per-key grouping (K 空间) ----
        first_idx: Dict[Hashable, int] = {}
        unique_keys: List[Hashable] = []
        for i, k in enumerate(k_list):
            if k not in first_idx:
                first_idx[k] = i
                unique_keys.append(k)

        group: Dict[Hashable, List[int]] = {}
        for i, k in enumerate(k_list):
            group.setdefault(k, []).append(i)

        # init per-key delta buffer
        key_delta: Dict[Hashable, torch.Tensor] = {}
        for k in unique_keys:
            di = delta_tbl[first_idx[k]]  # [C_table, H, W]
            key_delta[k] = di.clone()

        # ---- map Y -> class index & wrong index ----
        assert self.cls2idx is not None and self.C is not None and self.K_offset is not None
        try:
            y_idx_cpu = torch.tensor([self.cls2idx[yk] for yk in y_keys], dtype=torch.long)
        except KeyError as e:
            raise ValueError(f"[UE][SEP] unknown class label in batch: {e}")

        y_idx = y_idx_cpu.to(device)
        if int(self.C) <= 1:
            wrong_idx = y_idx  # 退化（几乎不会发生）
        else:
            wrong_idx = (y_idx + int(self.K_offset)) % int(self.C)  # 半环偏移

        last_loss = torch.tensor(0.0, device=device)

        # ---- PGD descent (outer T) ----
        for _ in _pbar(range(max(1, T)), desc="Noise Step"):
            # 1) per-sample delta view from per-key buffer
            delta_forward_list = [key_delta[k] for k in k_list]              # len=N
            delta_forward = torch.stack(delta_forward_list, dim=0)           # [N, C_table, H, W]
            if repeat_to_Cin:
                delta_forward = delta_forward.repeat(1, C_in, 1, 1)          # [N, C_in, H, W]

            # 2) forward-once container (we'll rebuild graph for each ckpt anyway)
            perturb_img = (x + delta_forward).clamp(0.0, 1.0).detach().requires_grad_(True)

            # 存每个 ckpt 的“按 key 聚合”的梯度（用于 g_avg）
            grads_per_ckpt: List[Dict[Hashable, torch.Tensor]] = []

            for j, state in enumerate(self.ckpt_states):
                self.s_model.load_state_dict(state, strict=False)
                m = self.s_model.to(device).eval()

                # 为避免图复用引发 retain_graph 复杂性，每个 ckpt 都单独构建规范化分支
                xn = perturb_img.clone()
                self._norm_inplace(xn, mean, std)

                out = m(xn)
                emb = out[1] if isinstance(out, (tuple, list)) else out     # [N, D]
                centers = self.centers_per_ckpt[j].to(device)[wrong_idx]     # [N, D]
                loss = F.mse_loss(emb, centers, reduction="mean")            # 目标：对齐到“错误中心”
                last_loss = loss
                (g,) = torch.autograd.grad(loss, perturb_img, retain_graph=False)  # [N, C_in, H, W]

                # 通道折叠
                g_fold = g.mean(dim=1, keepdim=True) if (repeat_to_Cin or fold_even_if_multi) else g

                # 按噪声键 k 聚合梯度（一个键可能在 batch 中出现多次）
                key_grad: Dict[Hashable, torch.Tensor] = {}
                for k, idxs in group.items():
                    gk = g_fold[idxs]                            # [m, C_fold, H, W]
                    gk_mean = gk.mean(dim=0, keepdim=False)      # [C_fold, H, W]
                    key_grad[k] = gk_mean
                grads_per_ckpt.append(key_grad)

            # 3) per-key g_avg（跨 ckpt 平均）
            g_avg_key: Dict[Hashable, torch.Tensor] = {}
            for k in unique_keys:
                g_avg_key[k] = torch.stack([d[k] for d in grads_per_ckpt], dim=0).mean(dim=0)

            # 4) per-key 像素盒上下界（在该键对应的样本集合上聚合）
            key_lower: Dict[Hashable, torch.Tensor] = {}
            key_upper: Dict[Hashable, torch.Tensor] = {}
            for k, idxs in group.items():
                xi = x[idxs]                                               # [m, C_in, H, W]
                if repeat_to_Cin or fold_even_if_multi:
                    lower_i = (-xi).amax(dim=1, keepdim=True)              # [m,1,H,W]
                    upper_i = (1 - xi).amin(dim=1, keepdim=True)
                    lower_k = lower_i.max(dim=0).values                    # [1,H,W]
                    upper_k = upper_i.min(dim=0).values
                    lb = torch.maximum(torch.full_like(key_delta[k], -eps), lower_k)
                    ub = torch.minimum(torch.full_like(key_delta[k],  eps), upper_k)
                else:
                    lower_i = (-xi)                                        # [m,C_in,H,W]
                    upper_i = (1 - xi)
                    lower_k = lower_i.max(dim=0).values                    # [C_in,H,W]
                    upper_k = upper_i.min(dim=0).values
                    lb = torch.maximum(torch.full_like(key_delta[k], -eps), lower_k)
                    ub = torch.minimum(torch.full_like(key_delta[k],  eps), upper_k)
                key_lower[k] = lb
                key_upper[k] = ub

            # 5) 内层 M 次 SVRG（在 per-key 空间更新）
            grad_sum_key: Dict[Hashable, torch.Tensor] = {k: torch.zeros_like(v) for k, v in key_delta.items()}
            J = len(self.ckpt_states)

            for _m in range(max(1, M)):
                j = int(torch.randint(J, (1,), device=device).item())

                # 重新构造基于“最新 key_delta”的 per-sample 视图
                delta_forward_list = [key_delta[k] for k in k_list]
                delta_forward = torch.stack(delta_forward_list, dim=0)
                if repeat_to_Cin:
                    delta_forward = delta_forward.repeat(1, C_in, 1, 1)
                x_v = (x + delta_forward).clamp(0.0, 1.0).detach().requires_grad_(True)

                self.s_model.load_state_dict(self.ckpt_states[j], strict=False)
                m_j = self.s_model.to(device).eval()
                xn_v = x_v.clone()
                self._norm_inplace(xn_v, mean, std)

                out_v = m_j(xn_v)
                emb_v = out_v[1] if isinstance(out_v, (tuple, list)) else out_v
                centers_v = self.centers_per_ckpt[j].to(device)[wrong_idx]
                loss_v = F.mse_loss(emb_v, centers_v, reduction="mean")
                (g_vj,) = torch.autograd.grad(loss_v, x_v, retain_graph=False)

                g_vj_fold = g_vj.mean(dim=1, keepdim=True) if (repeat_to_Cin or fold_even_if_multi) else g_vj

                # 聚合成 per-key，并做 SVRG 累积：g_vj - (g_hist - g_avg)
                for k, idxs in group.items():
                    gk_v = g_vj_fold[idxs].mean(dim=0)          # 当前点 v 的 per-key 梯度
                    gk_hist = grads_per_ckpt[j][k]               # 历史点 u 的 per-key 梯度（第 2 步算的）
                    gk_avg = g_avg_key[k]
                    grad_sum_key[k] = grad_sum_key[k] + (gk_v - (gk_hist - gk_avg))

                # 立刻以 sign 下降并投影（L_inf + 像素盒）
                for k in unique_keys:
                    dk = key_delta[k]
                    step = step_size * grad_sum_key[k].sign()
                    dk = dk - step                                   # 下降
                    dk = dk.clamp(-eps, +eps)                        # L_inf
                    dk = dk.clamp(key_lower[k], key_upper[k]).detach()  # 像素盒
                    key_delta[k] = dk

        # ---- commit back（per-sample view；避免 last-write-wins） ----
        delta_commit = torch.stack([key_delta[k] for k in k_list], dim=0)   # [N, C_tbl_or_fold, H, W]
        nb.commit_batch(k_list, delta_commit.detach().cpu())

        return {
            "noise_loss": float(last_loss.detach().cpu()),
            "delta_linf": float(delta_commit.detach().abs().max().cpu()),
            "num_unique_keys": len(unique_keys),
        }
