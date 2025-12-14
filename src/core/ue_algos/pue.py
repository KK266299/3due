# file: src/ue_algos/min_min.py
from __future__ import annotations
from typing import Dict, Iterable, List
import torch
import torch.nn.functional as F

from ...registry import register_plugin
from ...utils.config import get_config, require_config


from ...utils.losses import TripletLoss


@register_plugin("pue")
class PUE:
    @staticmethod
    def _norm_inplace(x: torch.Tensor, mean, std):
        for c, (m, s) in enumerate(zip(mean, std)):
            x[:, c].sub_(float(m)).div_(float(s))
        return x

    # === NEW: 临时加/减 权重噪声（不污染参数与优化器状态） ===
    @staticmethod
    def _add_weight_noise_(model: torch.nn.Module, sigma: float):
        if sigma <= 0:
            return None  # fast path
        noises = []
        with torch.no_grad():
            for p in model.parameters():
                eps = torch.randn_like(p) * sigma
                p.add_(eps)
                noises.append(eps)
        return noises

    @staticmethod
    def _remove_weight_noise_(model: torch.nn.Module, noises):
        if noises is None:
            return
        with torch.no_grad():
            for p, eps in zip(model.parameters(), noises):
                if eps is not None:
                    p.sub_(eps)

    # ---------------- Surrogate-step：Update surrogate ---------------- #
    def surrogate_step_batch(self, trainer, batch) -> Dict[str, float]:
        cfg = trainer.config
        device = trainer.device
        nb = trainer.noise_backend
        if nb is None:
            raise RuntimeError("[UE] noise_backend is required.")

        # --- RWP 配置读取 ---
        rwp_cfg = get_config(cfg, "ue.rwp", {})
        rwp_enabled = bool(get_config(rwp_cfg, "enabled", True))
        rwp_sigma   = float(get_config(rwp_cfg, "sigma", 0.05))
        U_train     = int(get_config(rwp_cfg, "U_train", 4))
        if (not rwp_enabled) or (rwp_sigma <= 0) or (U_train <= 1):
            rwp_enabled = False  # 自动退化

        x = batch["image"].to(device).float()  # [N,C,H,W], not Normalized yet
        y = batch["targets"]["reid"]
        y = y.to(device).long() if torch.is_tensor(y) else torch.as_tensor(y, device=device, dtype=torch.long)
        keys: Iterable[int] = batch["key"]

        mean = tuple(get_config(cfg, "training.data.transforms.mean", (0.485, 0.456, 0.406)))
        std = tuple(get_config(cfg, "training.data.transforms.std", (0.229, 0.224, 0.225)))
        margin = float(get_config(cfg, "ue.reid_margin", 0.2))

        # 全图噪声（backend只取放，不做任何PGD逻辑）
        delta = nb.batch_noise(keys).to(device)                   # [N,C_table,H,W]
        delta = delta.repeat(1, x.size(1), 1, 1) if delta.size(1) == 1 and x.size(1) > 1 else delta

        # surrogate 模型与优化器
        if not trainer.surrogates:
            raise RuntimeError("[UE] No surrogate bound.")
        name, s_model = next(iter(trainer.surrogates.items()))
        opt = trainer.opt_surrogates.get(name, None)
        if opt is None:
            raise RuntimeError(f"[UE] No optimizer for surrogate '{name}'.")

        s_model.train()
        for p in s_model.parameters():
            p.requires_grad = True

        # ========= 与论文一致：训练端启用 RWP (U_train 次抽样) =========
        opt.zero_grad(set_to_none=True)
        loss_scalar = 0.0

        if rwp_enabled:
            for _ in range(U_train):
                noises = self._add_weight_noise_(s_model, rwp_sigma)  # 临时加噪
                noisy = (x + delta).clamp(0.0, 1.0)
                xn = noisy.clone()
                self._norm_inplace(xn, mean, std)

                out = s_model(xn)
                emb = out[1] if isinstance(out, (tuple, list)) else out
                triplet_loss = TripletLoss(margin=margin)
                loss = triplet_loss(emb, y) / float(U_train)  # 均值聚合
                loss.backward()  # 累计梯度
                loss_scalar += float(loss.detach().cpu())     # 仅记录
                self._remove_weight_noise_(s_model, noises)   # 立刻减回噪声
        else:
            noisy = (x + delta).clamp(0.0, 1.0)
            xn = noisy.clone()
            self._norm_inplace(xn, mean, std)
            out = s_model(xn)
            emb = out[1] if isinstance(out, (tuple, list)) else out
            triplet_loss = TripletLoss(margin=margin)
            loss = triplet_loss(emb, y)
            loss.backward()
            loss_scalar = float(loss.detach().cpu())

        opt.step()
        return {"surrogate_loss": loss_scalar, "loss": loss_scalar}

    # ---------------- N-step：Update noise (PGD with optional RWP) ---------------- #
    def noise_step_batch(self, trainer, batch) -> Dict[str, float]:
        cfg = trainer.config
        device = trainer.device
        nb = trainer.noise_backend
        if nb is None:
            raise RuntimeError("[UE] noise_backend is required.")

        # --- RWP 配置读取 ---
        rwp_cfg = get_config(cfg, "ue.rwp", {})
        rwp_enabled = bool(get_config(rwp_cfg, "enabled", True))
        rwp_sigma   = float(get_config(rwp_cfg, "sigma", 0.05))
        U_noise     = int(get_config(rwp_cfg, "U_noise", 8))
        if (not rwp_enabled) or (rwp_sigma <= 0) or (U_noise <= 1):
            rwp_enabled = False  # 自动退化

        x = batch["image"].to(device).float()  # [N,C_in,H,W]
        y = batch["targets"]["reid"]
        y = y.to(device).long() if torch.is_tensor(y) else torch.as_tensor(y, device=device, dtype=torch.long)
        keys = batch["key"]
        keys_list: List[int] = list(keys)

        # 配置
        algo = require_config(cfg, "ue.algorithm")
        params = require_config(algo, "params")
        eps = float(get_config(params, "epsilon", 8/255.0))
        step_size = float(get_config(params, "step_size", 2/255.0))
        num_steps = int(get_config(params, "noise_step", 10))
        margin = float(get_config(cfg, "ue.reid_margin", 0.2))

        mean = tuple(get_config(cfg, "training.data.transforms.mean", (0.485, 0.456, 0.406)))
        std  = tuple(get_config(cfg, "training.data.transforms.std",  (0.229, 0.224, 0.225)))

        # 冻结 surrogate
        if not trainer.surrogates:
            raise RuntimeError("[UE] No surrogate bound.")
        _, s_model = next(iter(trainer.surrogates.items()))
        s_model.eval()
        for p in s_model.parameters():
            p.requires_grad = False

        # 取噪声表（class-wise 时同一 key 会在 batch 内重复）
        delta_tbl = nb.batch_noise(keys_list).to(device)  # [N,C_table,H,W]
        N, C_in, H, W = x.shape
        C_table = delta_tbl.size(1)

        # 1→C 通道处理策略：仅在表=1ch 且 输入多通道时 repeat 到 C_in
        repeat_to_Cin = (C_table == 1 and C_in > 1)
        if not repeat_to_Cin and C_table != C_in:
            raise RuntimeError(f"[UE] channel mismatch: table C={C_table} vs input C={C_in}")

        # 构建 unique keys 与 分组（一次就够）
        first_idx, unique_keys = {}, []
        for i, k in enumerate(keys_list):
            if k not in first_idx:
                first_idx[k] = i
                unique_keys.append(k)
        group: Dict[int, List[int]] = {}
        for i, k in enumerate(keys_list):
            group.setdefault(k, []).append(i)

        # 以 key 粒度维护 delta（形状与表一致：1ch 或 C_in）
        key_delta: Dict[int, torch.Tensor] = {}
        for k in unique_keys:
            di = delta_tbl[first_idx[k]]  # [C_table, H, W]
            key_delta[k] = di.clone()

        triplet = TripletLoss(margin=margin)
        last_loss = torch.tensor(0.0, device=device)

        with torch.enable_grad():
            for _ in range(max(1, num_steps)):
                # 展开到 per-sample 做前向
                delta_forward = torch.stack([key_delta[k] for k in keys_list], dim=0)  # [N,C_table,H,W]
                if repeat_to_Cin:
                    delta_forward = delta_forward.repeat(1, C_in, 1, 1)                 # [N,C_in,H,W]

                if rwp_enabled:
                    grad_accum = torch.zeros_like(delta_forward)
                    for _u in range(U_noise):
                        noises = self._add_weight_noise_(s_model, rwp_sigma)
                        perturb_img = (x + delta_forward).clamp(0.0, 1.0).detach().requires_grad_(True)
                        xn = perturb_img.clone()
                        self._norm_inplace(xn, mean, std)
                        out = s_model(xn)
                        emb = out[1] if isinstance(out, (tuple, list)) else out
                        loss_u = triplet(emb, y)             # error-minimizing：最小化
                        (g_u,) = torch.autograd.grad(loss_u, perturb_img, retain_graph=False, create_graph=False)
                        grad_accum.add_(g_u)
                        last_loss = loss_u
                        self._remove_weight_noise_(s_model, noises)
                    g = grad_accum / float(U_noise)
                else:
                    perturb_img = (x + delta_forward).clamp(0.0, 1.0).detach().requires_grad_(True)
                    xn = perturb_img.clone()
                    self._norm_inplace(xn, mean, std)
                    out = s_model(xn)
                    emb = out[1] if isinstance(out, (tuple, list)) else out
                    loss = triplet(emb, y)
                    (g,) = torch.autograd.grad(loss, perturb_img, retain_graph=False, create_graph=False)
                    last_loss = loss

                # 表=1ch 才折回 1ch；否则保持逐通道
                g_fold = g.mean(dim=1, keepdim=True) if repeat_to_Cin else g  # [N,1,H,W] or [N,C_in,H,W]

                # —— 按 key 聚合梯度 + 计算该 key 的像素盒界（对本批所有该 key 样本安全）——
                key_grad: Dict[int, torch.Tensor] = {}
                key_lower: Dict[int, torch.Tensor] = {}
                key_upper: Dict[int, torch.Tensor] = {}

                for k, idxs in group.items():
                    gk = g_fold[idxs]                           # [m, C_tbl_or_fold, H, W]
                    gk_mean = gk.mean(dim=0, keepdim=False)     # [C_tbl_or_fold, H, W]
                    key_grad[k] = gk_mean

                    xi = x[idxs]                                # [m, C_in, H, W]
                    if repeat_to_Cin:
                        lower_i = (-xi).amax(dim=1, keepdim=True)   # [m,1,H,W]
                        upper_i = (1 - xi).amin(dim=1, keepdim=True)
                        lower_k = lower_i.max(dim=0).values         # [1,H,W]
                        upper_k = upper_i.min(dim=0).values
                        lb = torch.maximum(torch.full_like(key_delta[k], -eps), lower_k)
                        ub = torch.minimum(torch.full_like(key_delta[k],  eps), upper_k)
                    else:
                        lower_i = (-xi)                              # [m,C_in,H,W]
                        upper_i = (1 - xi)
                        lower_k = lower_i.max(dim=0).values          # [C_in,H,W]
                        upper_k = upper_i.min(dim=0).values
                        lb = torch.maximum(torch.full_like(key_delta[k], -eps), lower_k)
                        ub = torch.minimum(torch.full_like(key_delta[k],  eps), upper_k)

                    key_lower[k] = lb
                    key_upper[k] = ub

                # —— 梯度“下降”更新（error-minimizing），再投影 ——
                for k in unique_keys:
                    dk = key_delta[k]
                    gk = key_grad[k].sign()
                    dk = dk - step_size * gk                         # ← 下降（保留你原来的方向）
                    dk = dk.clamp(-eps, +eps)                        # L_inf 投影
                    dk = dk.clamp(key_lower[k], key_upper[k]).detach()  # 像素盒一致性
                    key_delta[k] = dk

        # 一次性写回（展开为 per-sample 视图，兼容现有 backend）
        delta_commit = torch.stack([key_delta[k] for k in keys_list], dim=0)  # [N, C_table, H, W]
        nb.commit_batch(keys_list, delta_commit.detach().cpu())

        return {
            "noise_loss": float(last_loss.detach().cpu()),
            "delta_linf": float(delta_commit.detach().abs().max().cpu())
        }
