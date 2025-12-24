# file: src/core/ue_algos/sep.py
"""
SEP (Self-Ensemble Protection) with Variance Reduction for 3D Medical Image Segmentation.

Based on: "Self-ensemble protection: Training checkpoints are good data protectors"

Algorithm:
  1. Pre-train DNN for N epochs, save n equidistant checkpoints (done externally)
  2. For T protection iterations:
     a. Compute gradients g_k for all n checkpoints
     b. Compute ensemble average g_ens = (1/n) * Σ g_k
     c. SVRG inner loop (M iterations):
        - Pick random checkpoint k
        - g_upd = g_upd + G(h_k, x̂_m) - (g_k - g_ens)
        - x̂_{m+1} = Π_ε(x̂_m - α·sign(g_upd))
     d. Update: x_{t+1} = Π_ε(x_t - α·sign(g_upd))

Key: Training-free - only needs pre-trained checkpoints, no surrogate training.
"""
from __future__ import annotations
from typing import Dict, List, Any

import torch
from omegaconf import DictConfig, ListConfig
from monai.losses import DiceCELoss

from ...registry import register_plugin
from ...utils.config import get_config, require_config


@register_plugin("sep")
class SepUE:
    """
    SEP UE with SVRG for 3D segmentation (e.g., BraTS19).

    Training-free: Uses pre-trained checkpoints as ensemble.
    No surrogate training needed - surrogate_step is a no-op.
    """

    def __init__(self):
        self._seg_loss: DiceCELoss | None = None
        self._ckpt_states: List[Dict[str, torch.Tensor]] | None = None
        self._ckpt_paths: List[str] | None = None
        self._initialized: bool = False
        self._s_model: torch.nn.Module | None = None

    @staticmethod
    def _norm_inplace(x: torch.Tensor, mean, std):
        for c, (m, s) in enumerate(zip(mean, std)):
            x[:, c].sub_(float(m)).div_(float(s))
        return x

    def _get_seg_loss(self, trainer) -> DiceCELoss:
        if self._seg_loss is not None:
            return self._seg_loss

        cfg = trainer.config
        crit_cfg = get_config(cfg, "training.criterion", DictConfig({}))
        include_background = bool(get_config(crit_cfg, "include_background", False))
        squared_pred = bool(get_config(crit_cfg, "squared_pred", False))
        jaccard = bool(get_config(crit_cfg, "jaccard", False))
        lambda_dice = float(get_config(crit_cfg, "lambda_dice", 1.0))
        lambda_ce = float(get_config(crit_cfg, "lambda_ce", 1.0))

        self._seg_loss = DiceCELoss(
            include_background=include_background,
            to_onehot_y=True,
            softmax=True,
            squared_pred=squared_pred,
            jaccard=jaccard,
            lambda_dice=lambda_dice,
            lambda_ce=lambda_ce,
            reduction="mean",
        )
        return self._seg_loss

    def _maybe_init(self, trainer) -> None:
        if self._initialized:
            return

        cfg = trainer.config

        if not trainer.surrogates:
            raise RuntimeError("[UE][SEP] No surrogate bound.")
        name, s_model = next(iter(trainer.surrogates.items()))
        self._s_model = s_model

        ckpt_paths = (
            get_config(cfg, f"ue.surrogates.{name}.ckpt_paths", None)
            or get_config(cfg, "ue.sep.ckpt_paths", None)
        )

        if ckpt_paths is None or not isinstance(ckpt_paths, (list, ListConfig)) or len(ckpt_paths) == 0:
            print("[UE][SEP] Warning: No ckpt_paths provided, using current model state only.")
            self._ckpt_states = [None]
            self._ckpt_paths = []
        else:
            self._ckpt_paths = list(ckpt_paths)
            self._ckpt_states = []

            for p in self._ckpt_paths:
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
                self._ckpt_states.append(state)

            print(f"[UE][SEP] Loaded {len(self._ckpt_states)} checkpoints for ensemble.")

        self._initialized = True

    # ---------------- Surrogate-step: NO-OP (Training-free) ---------------- #
    def surrogate_step_batch(self, trainer, batch) -> Dict[str, float]:
        """
        SEP is training-free: no surrogate training needed.
        Only requires pre-trained checkpoints.
        """
        self._maybe_init(trainer)
        return {"surrogate_loss": 0.0, "loss": 0.0}

    # ---------------- Compute gradient for a checkpoint ---------------- #
    def _compute_gradient(
        self,
        ckpt_state,
        perturb_img: torch.Tensor,
        y: torch.Tensor,
        mean,
        std,
        seg_loss_fn,
        device,
    ) -> tuple:
        """Compute gradient of loss w.r.t. perturbed image for a single checkpoint."""
        if ckpt_state is not None:
            self._s_model.load_state_dict(ckpt_state, strict=False)

        self._s_model.eval()
        for p in self._s_model.parameters():
            p.requires_grad = False

        xn = perturb_img.clone()
        self._norm_inplace(xn, mean, std)

        out = self._s_model(xn)
        logits = out[0] if isinstance(out, (tuple, list)) else out

        loss = seg_loss_fn(logits, y.unsqueeze(1))
        (g,) = torch.autograd.grad(loss, perturb_img, retain_graph=False, create_graph=False)

        return g, float(loss.detach())

    # ---------------- N-step: Update noise using SEP with SVRG ---------------- #
    def noise_step_batch(self, trainer, batch) -> Dict[str, float]:
        """
        Update noise using SEP with SVRG optimization.

        Algorithm (per batch):
          For T outer iterations:
            1. Compute gradients g_k for all n checkpoints at current point
            2. g_ens = (1/n) * Σ g_k (ensemble average)
            3. For M inner SVRG iterations:
               - Pick random checkpoint k
               - g_upd += G(h_k, x̂_m) - (g_k - g_ens)
               - x̂_{m+1} = Π_ε(x̂_m - α·sign(g_upd))
            4. δ = Π_ε(δ - α·sign(g_upd))
        """
        self._maybe_init(trainer)

        cfg = trainer.config
        device = trainer.device
        nb = trainer.noise_backend
        if nb is None:
            raise RuntimeError("[UE][SEP] noise_backend is required.")

        # -------- data & config --------
        x = batch["image"].to(device).float()
        y = batch["label"]
        y = y.to(device).long() if torch.is_tensor(y) else torch.as_tensor(y, device=device, dtype=torch.long)
        keys = batch["key"]
        keys_list: List[int] = list(keys)

        N, C_in = x.shape[:2]

        algo = require_config(cfg, "ue.algorithm")
        params = require_config(algo, "params")
        eps = float(get_config(params, "epsilon", 8 / 255.0))
        step_size = float(get_config(params, "step_size", 2 / 255.0))
        T = int(get_config(params, "noise_step", 10))       # outer iterations
        M = int(get_config(params, "svrg_M", 5))            # inner SVRG iterations

        mean = tuple(get_config(cfg, "training.data.transforms.mean", (0.0,) * C_in))
        std = tuple(get_config(cfg, "training.data.transforms.std", (1.0,) * C_in))

        seg_loss_fn = self._get_seg_loss(trainer)

        # -------- init noise --------
        delta = nb.batch_noise(keys_list).to(device).float()
        if delta.shape[:2] != x.shape[:2]:
            raise RuntimeError(f"[UE][SEP] noise shape mismatch: {tuple(delta.shape)} vs {tuple(x.shape)}")

        delta = delta.clamp(-eps, eps)
        num_ckpts = len(self._ckpt_states)
        last_loss = 0.0

        # -------- Outer loop: T protection iterations --------
        with torch.enable_grad():
            for t in range(max(1, T)):
                # Current perturbed image
                x_t = (x + delta).clamp(0.0, 1.0)

                # Step 1: Compute gradients g_k for ALL checkpoints
                g_k_list = []
                for k, ckpt_state in enumerate(self._ckpt_states):
                    perturb_img = x_t.detach().requires_grad_(True)
                    g_k, loss_k = self._compute_gradient(
                        ckpt_state, perturb_img, y, mean, std, seg_loss_fn, device
                    )
                    g_k_list.append(g_k.detach())
                    last_loss = loss_k

                # Step 2: Compute ensemble average g_ens = (1/n) * Σ g_k
                g_ens = torch.stack(g_k_list, dim=0).mean(dim=0)

                # Step 3: Initialize for inner SVRG loop
                g_upd = torch.zeros_like(delta)
                x_hat = x_t.clone()  # virtual sample for inner loop

                # Step 4: Inner SVRG loop (M iterations)
                for m in range(max(1, M)):
                    # Pick random checkpoint index
                    k = int(torch.randint(num_ckpts, (1,)).item())

                    # Compute gradient at current x_hat
                    perturb_img = x_hat.detach().requires_grad_(True)
                    g_new, _ = self._compute_gradient(
                        self._ckpt_states[k], perturb_img, y, mean, std, seg_loss_fn, device
                    )

                    # SVRG gradient update: g_upd += g_new - (g_k - g_ens)
                    # This reduces variance compared to pure stochastic gradient
                    g_upd = g_upd + g_new - (g_k_list[k] - g_ens)

                    # Update virtual sample x̂_{m+1} = Π_ε(x̂_m - α·sign(g_upd))
                    # Note: We update in the direction that MINIMIZES loss (min-min)
                    x_hat = x_hat - step_size * g_upd.sign()
                    # Project to valid range: |x_hat - x| <= eps and x_hat in [0,1]
                    x_hat = torch.max(torch.min(x_hat, x + eps), x - eps)
                    x_hat = x_hat.clamp(0.0, 1.0)

                # Step 5: Final update δ = Π_ε(δ - α·sign(g_upd))
                delta = delta - step_size * g_upd.sign()
                delta = delta.clamp(-eps, eps)

        # -------- Write back to noise backend --------
        nb.commit_batch(keys_list, delta.detach().cpu())

        delta_linf = float(delta.detach().abs().max().cpu())
        return {
            "noise_loss": last_loss,
            "delta_linf": delta_linf,
            "num_checkpoints": num_ckpts,
        }