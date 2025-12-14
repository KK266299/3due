# file: src/ue_algos/lpp.py
import math
from contextlib import contextmanager
from typing import List, Tuple, Dict, Optional

import torch
from torch import nn
from src.utils.logger import get_logger

ParamRefs = List[Tuple[str, torch.nn.Parameter]]


class LocalPosteriorSampler:
    """
    LPP（简化 ReID 版）：
      - 仅按 last_k 从 backbone.features 选“最后 K 个块”的所有参数
      - 总是包含 neck / proj（把最后一层包进去）
      - 选参只在 __init__ 执行一次；会打印清单
      - 粒子：SGLD / SAM；权重临时替换确保退出后还原
      - 稳定项：信任域投影 / 可选梯度裁剪 / 可选 AMP（finite_guard 已移除）
    """

    def __init__(
        self,
        model: nn.Module,
        last_k: int = 2,
        device: Optional[torch.device] = None,
        *,
        amp: bool = False,
        trust_region: Optional[Dict[str, float]] = None,  # {'enabled': True, 'r_rel': 0.05, 'r_abs': 1e-3}
        grad_clip_max_norm: float = 0.0,                  # <=0 关闭
    ):
        self.model = model
        self.device = device or next(model.parameters()).device
        self.last_k = max(1, int(last_k))
        self.amp = bool(amp)
        self.grad_clip_max_norm = float(grad_clip_max_norm)

        tr = trust_region or {}
        self.tr_enabled = bool(tr.get("enabled", True))
        self.tr_r_rel = float(tr.get("r_rel", 0.05))
        self.tr_r_abs = float(tr.get("r_abs", 1e-3))

        self.logger = get_logger()
        self.param_refs: ParamRefs = self._select_tail_params_once()
        self._log_selected_params(self.param_refs)

        # 缓存 ||θ0|| 用于信任域半径
        self._theta0_norm_cache: Optional[float] = None

    # -------------------- 选“最后 K 个块”的参数（只执行一次） -------------------- #
    def _select_tail_params_once(self) -> ParamRefs:
        refs: ParamRefs = []

        def _find_module_prefix(root: nn.Module, target: nn.Module) -> Optional[str]:
            for nm, mod in root.named_modules():
                if mod is target:
                    return nm
            return None

        bb = getattr(self.model, "backbone", None) or self.model
        bb_prefix = _find_module_prefix(self.model, bb) or ""

        features = getattr(bb, "features", None)
        if isinstance(features, nn.Sequential) and len(features) > 0:
            feat_prefix = f"{bb_prefix}.features" if bb_prefix else "features"
            L = len(features)
            start = max(0, L - self.last_k)
            tail_block_prefixes = [f"{feat_prefix}.{i}" for i in range(start, L)]
            for pn, p in self.model.named_parameters():
                for pref in tail_block_prefixes:
                    if pn.startswith(pref + ".") or pn == pref:
                        refs.append((pn, p)); break
        else:
            # fallback：限制在 backbone 命名空间的“叶子”
            leaf_prefixes = []
            for subname, mod in bb.named_modules():
                if len(list(mod.children())) == 0:
                    full = bb_prefix if subname == "" else (f"{bb_prefix}.{subname}" if bb_prefix else subname)
                    leaf_prefixes.append(full)
            leaf_prefixes = leaf_prefixes[-self.last_k:] if leaf_prefixes else []
            for pn, p in self.model.named_parameters():
                for pref in leaf_prefixes:
                    if pn.startswith(pref + ".") or pn == pref:
                        refs.append((pn, p)); break

        # 总是包含 neck / proj
        for attr in ("neck", "proj"):
            mod = getattr(self.model, attr, None)
            if isinstance(mod, nn.Module):
                prefix = _find_module_prefix(self.model, mod) or attr
                for pn, p in self.model.named_parameters():
                    if pn.startswith(prefix + ".") or pn == prefix:
                        refs.append((pn, p))

        # 去重
        seen, uniq = set(), []
        for n, p in refs:
            if n not in seen:
                uniq.append((n, p)); seen.add(n)

        if not uniq:
            raise RuntimeError("[LPP] 未从 backbone/features/neck/proj 选到任何参数，请检查 surrogate 结构与 last_k 设置。")
        return uniq

    def _log_selected_params(self, refs: ParamRefs):
        total = 0
        self.logger.info(f"[LPP] Selected {len(refs)} tensors (last_k={self.last_k}):")
        for n, p in refs:
            num = p.numel(); total += num
            self.logger.info(f"    - {n}: {tuple(p.shape)} ({num} params)")
        self.logger.info(f"[LPP] Total tail params: {total}")

    # -------------------- 权重临时替换与状态 -------------------- #
    @contextmanager
    def temp_assign_weights(self, theta_dict: Dict[str, torch.Tensor]):
        backup = {}
        with torch.no_grad():
            for n, p in self.param_refs:
                backup[n] = p.data.clone()
                p.data.copy_(theta_dict[n])
        try:
            yield
        finally:
            with torch.no_grad():
                for n, p in self.param_refs:
                    p.data.copy_(backup[n])

    def snapshot_theta0(self) -> Dict[str, torch.Tensor]:
        theta0 = {n: p.data.detach().clone() for n, p in self.param_refs}
        self._theta0_norm_cache = self._vec_norm(theta0)
        return theta0

    def _zero_subset_grad(self):
        for _, p in self.param_refs:
            if p.grad is not None:
                p.grad.zero_()

    @contextmanager
    def _temp_requires_grad(self, flag: bool):
        """临时打开/关闭所选参数的 requires_grad（外部把 s_model 冻结了，这里局部打开）"""
        backup = {}
        for _, p in self.param_refs:
            backup[id(p)] = p.requires_grad
            p.requires_grad = flag
        try:
            yield
        finally:
            for _, p in self.param_refs:
                p.requires_grad = backup[id(p)]

    # -------------------- 工具：范数与信任域投影/裁剪 -------------------- #
    @staticmethod
    def _vec_norm(theta: Dict[str, torch.Tensor]) -> float:
        s = 0.0
        for v in theta.values():
            s += float(v.detach().pow(2).sum().item())
        return s ** 0.5

    def _proj_ball_(self, theta: Dict[str, torch.Tensor], theta0: Dict[str, torch.Tensor]):
        if not self.tr_enabled:
            return
        base = self._theta0_norm_cache or self._vec_norm(theta0)
        self._theta0_norm_cache = base
        r = max(self.tr_r_abs, self.tr_r_rel * base)

        # 正确的球投影：θ ← θ0 + (r/d) * (θ - θ0)
        diff_norm2 = 0.0
        for n in theta:
            d = (theta[n] - theta0[n]).detach()
            diff_norm2 += float(d.pow(2).sum().item())
        d = diff_norm2 ** 0.5
        if d <= r or d <= 1e-12:
            return

        scale = r / d
        with torch.no_grad():
            for n in theta:
                theta[n] = theta0[n] + (theta[n] - theta0[n]) * scale

    def _maybe_clip_grads_(self, max_norm: float):
        if max_norm is None or max_norm <= 0:
            return
        sq = 0.0
        for _, p in self.param_refs:
            if p.grad is not None:
                sq += float(p.grad.detach().pow(2).sum().item())
        total = sq ** 0.5
        if total <= 1e-12 or total <= max_norm:
            return
        scale = max_norm / total
        with torch.no_grad():
            for _, p in self.param_refs:
                if p.grad is not None:
                    p.grad.mul_(scale)

    # -------------------- 采样器：SGLD / SAM -------------------- #
    def _loss_with_optional_amp(self, make_loss_fn):
        if self.amp:
            with torch.cuda.amp.autocast(enabled=True):
                return make_loss_fn()
        else:
            return make_loss_fn()

    def sgld_particle(
        self,
        make_loss_fn,
        theta0: Dict[str, torch.Tensor],
        steps: int = 3,
        eta: float = 1e-4,
        gamma: float = 1e-4,
        sigma0: float = 1e-4,
    ) -> Dict[str, torch.Tensor]:
        """
        SGLD with quadratic pullback + trust-region projection + optional grad clip.
            θ ← θ - η(∇L + γ(θ-θ0)) + sqrt(2ηγ)·ξ,  再投影到 ||θ-θ0|| ≤ r
        """
        theta = {n: t + sigma0 * torch.randn_like(t) for n, t in theta0.items()}

        with torch.enable_grad(), self._temp_requires_grad(True):
            for _ in range(max(1, steps)):
                with self.temp_assign_weights(theta):
                    loss = self._loss_with_optional_amp(make_loss_fn)

                self._zero_subset_grad()
                loss.backward()
                self._maybe_clip_grads_(self.grad_clip_max_norm)

                with torch.no_grad():
                    for n, p in self.param_refs:
                        g = p.grad if p.grad is not None else torch.zeros_like(p)
                        noise = torch.randn_like(theta[n]) * math.sqrt(max(1e-12, 2.0 * eta * gamma))
                        theta[n] = theta[n] - eta * (g + gamma * (theta[n] - theta0[n])) + noise

                self._zero_subset_grad()
                self._proj_ball_(theta, theta0)

        return theta

    def sam_particle(
        self,
        make_loss_fn,
        theta0: Dict[str, torch.Tensor],
        radius: float = 5e-3,
    ) -> Dict[str, torch.Tensor]:
        """
        SAM 一步对抗方向 + 信任域投影 / 可选裁剪：
            g = ∂L/∂θ |_{θ0}
            θ_adv = θ0 + ρ * g / ||g||  →  投影到 ||θ-θ0|| ≤ r
        """
        with torch.enable_grad(), self._temp_requires_grad(True):
            with self.temp_assign_weights(theta0):
                loss = self._loss_with_optional_amp(make_loss_fn)

            self._zero_subset_grad()
            loss.backward()
            self._maybe_clip_grads_(self.grad_clip_max_norm)

            theta_adv = {}
            with torch.no_grad():
                sq = 0.0
                for _, p in self.param_refs:
                    if p.grad is not None:
                        sq += float(p.grad.detach().pow(2).sum().item())
                denom = (sq ** 0.5) + 1e-12

                for n, p in self.param_refs:
                    g = p.grad if p.grad is not None else torch.zeros_like(p)
                    theta_adv[n] = theta0[n] + radius * g / denom

            self._zero_subset_grad()
            self._proj_ball_(theta_adv, theta0)

        return theta_adv
