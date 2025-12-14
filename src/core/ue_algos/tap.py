# file: src/ue_algos/tap.py
from __future__ import annotations
from typing import Dict, Iterable, List, Tuple
import os
import torch

from ...registry import register_plugin
from ...utils.config import get_config, require_config
from ...utils.losses import TripletLoss


@register_plugin("tap")
class TAP:
    """
    TAP-Triplet (open-set ReID):
      - surrogate_step_batch: lazy-load weights (if any), freeze surrogate; no training
      - noise_step_batch: PGD ascent on TripletLoss to craft class-wise noise
        * class-wise: per-key gradient aggregation & per-key pixel-box bounds
        * L_inf projection + pixel-box consistency
        * fold-to-1ch only when table=1ch and input has C_in>1 (unless tied_channels=True)
    """

    # ---------------- utils ---------------- #
    @staticmethod
    def _norm_inplace(x: torch.Tensor, mean, std):
        for c, (m, s) in enumerate(zip(mean, std)):
            x[:, c].sub_(float(m)).div_(float(s))
        return x

    @staticmethod
    def _strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # handle "module." prefix from DDP/DataParallel
        if not state_dict:
            return state_dict
        keys = list(state_dict.keys())
        if all(k.startswith("module.") for k in keys):
            return {k[len("module."):]: v for k, v in state_dict.items()}
        return state_dict

    def _lazy_load_surrogates(self, trainer) -> int:
        """
        Load weights_path for each surrogate once, then freeze.
        Returns how many were (newly) loaded.
        """
        if not hasattr(trainer, "_tap_loaded"):
            trainer._tap_loaded = {}
        cfg = trainer.config
        loaded = 0

        for name, s_model in trainer.surrogates.items():
            if trainer._tap_loaded.get(name, False):
                # already loaded & frozen
                s_model.eval()
                for p in s_model.parameters(): p.requires_grad = False
                continue

            # read weights_path from config: ue.surrogates.<name>.weights_path
            wpath = get_config(cfg, f"ue.surrogates.{name}.weights_path", None)
            if wpath and isinstance(wpath, str):
                if not os.path.isfile(wpath):
                    raise FileNotFoundError(f"[UE][TAP] weights_path not found for surrogate '{name}': {wpath}")
                obj = torch.load(wpath, map_location=trainer.device, weights_only=False)
                # support both raw state_dict or a dict with 'state_dict'
                state = obj.get("state_dict", obj) if isinstance(obj, dict) else obj
                state = self._strip_module_prefix(state)
                missing, unexpected = s_model.load_state_dict(state, strict=False)
                if missing or unexpected:
                    # 不要中断，打印提醒即可（适配不同头/投影层）
                    print(f"[UE][TAP] Loaded '{name}' with non-strict keys. "
                          f"missing={len(missing)}, unexpected={len(unexpected)}")

                loaded += 1

            # freeze regardless of whether weights were provided
            s_model.eval()
            for p in s_model.parameters(): p.requires_grad = False
            trainer._tap_loaded[name] = True

        if not trainer.surrogates:
            raise RuntimeError("[UE][TAP] No surrogate bound. Please bind a crafting (frozen) model.")
        return loaded

    # ---------------- Surrogate-step：lazy-load only (no training) ---------------- #
    def surrogate_step_batch(self, trainer, batch) -> Dict[str, float]:
        """
        TAP 生成阶段不训练 surrogate。
        这里只做：按 weights_path 懒加载一次 + 冻结；可选做一次前向就绪检查。
        """
        n_loaded = self._lazy_load_surrogates(trainer)

        # optional readiness check on the first surrogate
        try:
            device = trainer.device
            x = batch["image"].to(device).float()
            mean = tuple(get_config(trainer.config, "training.data.transforms.mean", (0.485, 0.456, 0.406)))
            std  = tuple(get_config(trainer.config, "training.data.transforms.std",  (0.229, 0.224, 0.225)))
            name, s_model = next(iter(trainer.surrogates.items()))
            xn = x.clone()
            self._norm_inplace(xn, mean, std)
            out = s_model(xn)
            _ = out[1] if isinstance(out, (tuple, list)) else out  # embedding exists
        except Exception as e:
            # 不阻塞主流程；若你希望严格校验可 raise
            print(f"[UE][TAP] Surrogate readiness check failed (non-fatal): {e}")

        return {"surrogate_loaded": float(n_loaded), "surrogate_loss": 0.0, "loss": 0.0}

    # ---------------- N-step：Update noise (PGD ascent, class-wise) ---------------- #
    def noise_step_batch(self, trainer, batch) -> Dict[str, float]:
        """
        对 class-wise 噪声做 PGD 上升（最大化 TripletLoss）。
        - 同一 key 在一个 batch 多次出现：先按 key 聚合梯度与像素盒界，再唯一写回；
        - 表=1ch 且输入多通道：把多通道梯度均值折回 1ch；否则保持逐通道；
        - 若 tied_channels=True，则即使表=多通道也会按通道均值折回（你当前配置是 false）。
        """
        cfg = trainer.config
        device = trainer.device
        nb = trainer.noise_backend
        if nb is None:
            raise RuntimeError("[UE][TAP] noise_backend is required.")

        # ensure surrogates loaded & frozen
        self._lazy_load_surrogates(trainer)

        # ---- batch data ----
        x = batch["image"].to(device).float()                # [N, C_in, H, W]
        y = batch["targets"]["reid"]
        y = y.to(device).long() if torch.is_tensor(y) else torch.as_tensor(y, device=device, dtype=torch.long)
        keys: Iterable[int] = batch["key"]
        keys_list: List[int] = list(keys)

        # ---- cfg ----
        algo = require_config(cfg, "ue.algorithm")
        params = require_config(algo, "params")
        eps = float(get_config(params, "epsilon", 8/255.0))
        step_size = float(get_config(params, "step_size", 2/255.0))
        num_steps = int(get_config(params, "noise_step", 10))
        tied_channels = bool(get_config(params, "tied_channels", False))

        margin = float(get_config(cfg, "ue.reid_margin", 0.2))
        mean = tuple(get_config(cfg, "training.data.transforms.mean", (0.485, 0.456, 0.406)))
        std  = tuple(get_config(cfg, "training.data.transforms.std",  (0.229, 0.224, 0.225)))

        # ---- surrogate (crafting) ----
        _, s_model = next(iter(trainer.surrogates.items()))
        s_model.eval()
        for p in s_model.parameters(): p.requires_grad = False

        # ---- noise table (class-wise) ----
        delta_tbl = nb.batch_noise(keys_list).to(device)     # [N, C_table, H, W] (same key repeats)
        N, C_in, H, W = x.shape
        C_table = delta_tbl.size(1)

        # channel policy
        repeat_to_Cin = (C_table == 1 and C_in > 1)
        if not repeat_to_Cin and C_table != C_in:
            raise RuntimeError(f"[UE][TAP] channel mismatch: table C={C_table} vs input C={C_in}")
        fold_even_if_multi = bool(tied_channels)

        # build unique key index
        first_idx: Dict[int, int] = {}
        unique_keys: List[int] = []
        for i, k in enumerate(keys_list):
            if k not in first_idx:
                first_idx[k] = i
                unique_keys.append(k)

        # init per-key delta (shape: 1ch or C_in)
        key_delta: Dict[int, torch.Tensor] = {}
        for k in unique_keys:
            di = delta_tbl[first_idx[k]]  # [C_table, H, W]
            key_delta[k] = di.clone()

        triplet = TripletLoss(margin=margin)
        last_loss = torch.tensor(0.0, device=device)

        # grouping indices per key
        group: Dict[int, List[int]] = {}
        for i, k in enumerate(keys_list):
            group.setdefault(k, []).append(i)

        # ---- PGD ascent loop ----
        for _ in range(max(1, num_steps)):
            # expand per-key delta to per-sample
            delta_forward_list = [key_delta[k] for k in keys_list]              # len=N, each [C_table, H, W]
            delta_forward = torch.stack(delta_forward_list, dim=0)              # [N, C_table, H, W]
            if repeat_to_Cin:
                delta_forward = delta_forward.repeat(1, C_in, 1, 1)             # [N, C_in, H, W]

            # forward on crafting model
            perturb_img = (x + delta_forward).clamp(0.0, 1.0).detach().requires_grad_(True)
            xn = perturb_img.clone()
            self._norm_inplace(xn, mean, std)

            out = s_model(xn)
            emb = out[1] if isinstance(out, (tuple, list)) else out
            loss = triplet(emb, y)                    # maximize
            last_loss = loss

            (g,) = torch.autograd.grad(loss, perturb_img, retain_graph=False, create_graph=False)  # [N, C_in, H, W]

            # fold channel grad?
            if repeat_to_Cin or fold_even_if_multi:
                g_fold = g.mean(dim=1, keepdim=True)                                        # [N,1,H,W]
            else:
                g_fold = g                                                                  # [N,C_in,H,W]

            # ---- per-key gradient & pixel-box bounds aggregation ----
            key_grad: Dict[int, torch.Tensor] = {}
            key_lower: Dict[int, torch.Tensor] = {}
            key_upper: Dict[int, torch.Tensor] = {}

            for k, idxs in group.items():
                gk = g_fold[idxs]                                           # [m, C_tbl_or_fold, H, W]
                gk_mean = gk.mean(dim=0, keepdim=False)                     # [C_tbl_or_fold, H, W]
                key_grad[k] = gk_mean

                xi = x[idxs]                                                # [m, C_in, H, W]
                if repeat_to_Cin or fold_even_if_multi:
                    lower_i = (-xi).amax(dim=1, keepdim=True)               # [m,1,H,W]
                    upper_i = (1 - xi).amin(dim=1, keepdim=True)
                    lower_k = lower_i.max(dim=0).values                     # [1,H,W]
                    upper_k = upper_i.min(dim=0).values
                    lb = torch.maximum(torch.full_like(key_delta[k], -eps), lower_k)
                    ub = torch.minimum(torch.full_like(key_delta[k],  eps), upper_k)
                else:
                    lower_i = (-xi)                                         # [m,C_in,H,W]
                    upper_i = (1 - xi)
                    lower_k = lower_i.max(dim=0).values                     # [C_in,H,W]
                    upper_k = upper_i.min(dim=0).values
                    lb = torch.maximum(torch.full_like(key_delta[k], -eps), lower_k)
                    ub = torch.minimum(torch.full_like(key_delta[k],  eps), upper_k)

                key_lower[k] = lb
                key_upper[k] = ub

            # ---- ascent + projection ----
            for k in unique_keys:
                dk = key_delta[k]
                gk = key_grad[k].sign()
                dk = dk + step_size * gk                       # ascend
                dk = dk.clamp(-eps, +eps)                      # L_inf
                dk = dk.clamp(key_lower[k], key_upper[k]).detach()   # pixel-box
                key_delta[k] = dk

        # ---- commit back (per-sample view) ----
        delta_commit = torch.stack([key_delta[k] for k in keys_list], dim=0)   # [N, C_tbl_or_fold, H, W]
        nb.commit_batch(keys_list, delta_commit.detach().cpu())

        return {
            "noise_loss": float(last_loss.detach().cpu()),
            "delta_linf": float(delta_commit.detach().abs().max().cpu()),
            "num_unique_keys": len(unique_keys)
        }
