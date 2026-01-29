# file: src/core/ue_algos/pue.py
"""
PUE (Provably Unlearnable Examples) for 3D Medical Image Segmentation.

Based on: "Provably Unlearnable Data Examples" (NDSS 2025)
Authors: Derui Wang, Minhui Xue, Bo Li, Seyit Camtepe, Liming Zhu
Paper: https://www.ndss-symposium.org/ndss-paper/provably-unlearnable-data-examples/
PDF: https://www.ndss-symposium.org/wp-content/uploads/2025-886-paper.pdf
Code: https://github.com/NeuralSec/certified-data-learnability

Algorithm:
  核心思想：通过在训练和噪声优化过程中对模型权重添加随机高斯噪声（Random Weight Perturbation, RWP），
  使得生成的噪声对模型权重扰动具有鲁棒性，从而产生"可证明不可学习"的样本。

  论文提出 (q, η)-Learnability 认证机制，通过参数平滑（parametric smoothing）来认证
  不可学习数据集的有效性，提供测试准确率的理论上界保证。

  1. Surrogate Step (S-step):
     - 对代理模型权重添加临时高斯噪声 N(0, σ²)
     - 进行多次采样（U_train次），累积梯度后更新
     - 训练后立即移除噪声，保持原始权重

  2. Noise Step (N-step):
     - 冻结代理模型
     - 对权重添加随机噪声，进行U_noise次采样
     - 对每次采样的梯度取平均，再进行PGD更新
     - 使用min-min策略最小化分割损失

Key: RWP (Random Weight Perturbation) 使噪声对模型初始化更加鲁棒，
     提供理论上的 (q, η)-Learnability 认证保证。
"""
from __future__ import annotations
from typing import Dict, Iterable, List

import torch
from omegaconf import DictConfig
from monai.losses import DiceCELoss

from ...registry import register_plugin
from ...utils.config import get_config, require_config


@register_plugin("pue")
class PUE:
    """
    PUE (Provably Unlearnable Examples) for 3D segmentation (e.g., BraTS19).

    假设：
      - Batch:
          batch["image"]: FloatTensor [B, C, ...]      (3D: [B,C,D,H,W])
          batch["label"]: LongTensor  [B,   ...]       (3D: [B,D,H,W])
          batch["key"]:   sample-wise key（每个样本一个 key）
      - Surrogate:
          s_model(x) -> logits: [B, C_seg, ...]
      - Noise backend:
          noise_backend.batch_noise(keys) -> [N, C_in, ...]
            * 只支持 sample-wise，不考虑 class-wise
            * 通道数必须与输入一致：C_noise == C_in
    """

    def __init__(self):
        # segmentation 用的 DiceCE loss（lazy 构建）
        self._seg_loss: DiceCELoss | None = None

    @staticmethod
    def _norm_inplace(x: torch.Tensor, mean, std):
        """
        In-place per-channel normalize for ND volume.
        x: [B, C, ...]
        """
        for c, (m, s) in enumerate(zip(mean, std)):
            x[:, c].sub_(float(m)).div_(float(s))
        return x

    # === RWP: 临时加/减 权重噪声（不污染参数与优化器状态） ===
    @staticmethod
    def _add_weight_noise_(model: torch.nn.Module, sigma: float):
        """
        对模型所有可训练参数添加临时高斯噪声 N(0, σ²)。
        返回噪声列表，用于之后恢复原始权重。
        """
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
        """
        移除之前添加的权重噪声，恢复原始参数。
        """
        if noises is None:
            return
        with torch.no_grad():
            for p, eps in zip(model.parameters(), noises):
                if eps is not None:
                    p.sub_(eps)

    def _get_seg_loss(self, trainer) -> DiceCELoss:
        """
        构建与 SegTrainer 一致配置的 DiceCELoss，用于 surrogate / noise step。
        """
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
            to_onehot_y=True,   # 标签是 index [B,...]
            softmax=True,       # 多类别 segmentation
            squared_pred=squared_pred,
            jaccard=jaccard,
            lambda_dice=lambda_dice,
            lambda_ce=lambda_ce,
            reduction="mean",
        )
        return self._seg_loss

    # ---------------- Surrogate-step：Update surrogate with RWP ---------------- #
    def surrogate_step_batch(self, trainer, batch) -> Dict[str, float]:
        """
        使用 RWP 更新 surrogate 参数。

        论文一致：训练端启用 RWP (U_train 次抽样)，对每次采样的loss求平均后反传。
        Loss: segmentation DiceCE
        """
        cfg = trainer.config
        device = trainer.device
        nb = trainer.noise_backend
        if nb is None:
            raise RuntimeError("[UE][PUE] noise_backend is required.")

        # --- RWP 配置读取 ---
        rwp_cfg = get_config(cfg, "ue.rwp", {})
        rwp_enabled = bool(get_config(rwp_cfg, "enabled", True))
        rwp_sigma = float(get_config(rwp_cfg, "sigma", 0.05))
        U_train = int(get_config(rwp_cfg, "U_train", 4))
        if (not rwp_enabled) or (rwp_sigma <= 0) or (U_train <= 1):
            rwp_enabled = False  # 自动退化为普通训练

        # data
        x = batch["image"].to(device).float()  # [B,C,...]
        y = batch["label"]
        y = y.to(device).long() if torch.is_tensor(y) else torch.as_tensor(
            y, device=device, dtype=torch.long
        )
        keys: Iterable[int] = batch["key"]

        B, C_in = x.shape[:2]

        # normalization config（默认 no-op: mean=0, std=1）
        mean = tuple(get_config(cfg, "training.data.transforms.mean", (0.0,) * C_in))
        std = tuple(get_config(cfg, "training.data.transforms.std", (1.0,) * C_in))

        # noise: sample-wise，通道数必须与输入一致
        delta = nb.batch_noise(list(keys)).to(device).float()  # [B,C_in,...]
        if delta.shape[:2] != x.shape[:2]:
            raise RuntimeError(
                f"[UE][PUE] noise shape mismatch: noise {tuple(delta.shape)} vs input {tuple(x.shape)}"
            )

        # select surrogate and optimizer
        if not trainer.surrogates:
            raise RuntimeError("[UE][PUE] No surrogate bound.")
        name, s_model = next(iter(trainer.surrogates.items()))
        opt = trainer.opt_surrogates.get(name, None)
        if opt is None:
            raise RuntimeError(f"[UE][PUE] No optimizer for surrogate '{name}'.")

        seg_loss_fn = self._get_seg_loss(trainer)

        s_model.train()
        for p in s_model.parameters():
            p.requires_grad = True

        # ========= RWP: 训练端启用 (U_train 次抽样) =========
        opt.zero_grad(set_to_none=True)
        loss_scalar = 0.0

        if rwp_enabled:
            for _ in range(U_train):
                noises = self._add_weight_noise_(s_model, rwp_sigma)  # 临时加噪

                # forward with noisy input and noisy weights
                noisy = (x + delta).clamp(0.0, 1.0)
                xn = noisy.clone()
                self._norm_inplace(xn, mean, std)

                out = s_model(xn)
                logits = out[0] if isinstance(out, (tuple, list)) else out

                loss = seg_loss_fn(logits, y.unsqueeze(1)) / float(U_train)  # 均值聚合
                loss.backward()  # 累计梯度
                loss_scalar += float(loss.detach().cpu())

                self._remove_weight_noise_(s_model, noises)  # 立刻减回噪声
        else:
            # 普通训练（无RWP）
            noisy = (x + delta).clamp(0.0, 1.0)
            xn = noisy.clone()
            self._norm_inplace(xn, mean, std)

            out = s_model(xn)
            logits = out[0] if isinstance(out, (tuple, list)) else out

            loss = seg_loss_fn(logits, y.unsqueeze(1))
            loss.backward()
            loss_scalar = float(loss.detach().cpu())

        opt.step()

        return {
            "surrogate_loss": loss_scalar,
            "loss": loss_scalar,
        }

    # ---------------- N-step：Update noise (PGD with RWP) ---------------- #
    def noise_step_batch(self, trainer, batch) -> Dict[str, float]:
        """
        使用 PGD + RWP 更新噪声。

        论文一致：噪声优化端启用 RWP (U_noise 次抽样)，对每次采样的梯度求平均后进行PGD更新。
        使用 min-min 策略：最小化分割损失。
        """
        cfg = trainer.config
        device = trainer.device
        nb = trainer.noise_backend
        if nb is None:
            raise RuntimeError("[UE][PUE] noise_backend is required.")

        # --- RWP 配置读取 ---
        rwp_cfg = get_config(cfg, "ue.rwp", {})
        rwp_enabled = bool(get_config(rwp_cfg, "enabled", True))
        rwp_sigma = float(get_config(rwp_cfg, "sigma", 0.05))
        U_noise = int(get_config(rwp_cfg, "U_noise", 8))
        if (not rwp_enabled) or (rwp_sigma <= 0) or (U_noise <= 1):
            rwp_enabled = False  # 自动退化

        # -------- data & config --------
        x = batch["image"].to(device).float()  # [N, C_in, ...]
        y = batch["label"]
        y = y.to(device).long() if torch.is_tensor(y) else torch.as_tensor(
            y, device=device, dtype=torch.long
        )
        keys = batch["key"]
        keys_list: List[int] = list(keys)

        N, C_in = x.shape[:2]

        algo = require_config(cfg, "ue.algorithm")
        params = require_config(algo, "params")
        eps = float(get_config(params, "epsilon", 8 / 255.0))
        step_size = float(get_config(params, "step_size", 2 / 255.0))
        num_steps = int(get_config(params, "noise_step", 10))

        # normalization config
        mean = tuple(get_config(cfg, "training.data.transforms.mean", (0.0,) * C_in))
        std = tuple(get_config(cfg, "training.data.transforms.std", (1.0,) * C_in))

        seg_loss_fn = self._get_seg_loss(trainer)

        # -------- freeze surrogate --------
        if not trainer.surrogates:
            raise RuntimeError("[UE][PUE] No surrogate bound.")
        _, s_model = next(iter(trainer.surrogates.items()))
        s_model.eval()
        for p in s_model.parameters():
            p.requires_grad = False

        # -------- init / clamp noise --------
        delta_tbl = nb.batch_noise(keys_list).to(device).float()  # [N, C_in, ...]
        if delta_tbl.shape[:2] != x.shape[:2]:
            raise RuntimeError(
                f"[UE][PUE] noise shape mismatch: noise {tuple(delta_tbl.shape)} vs input {tuple(x.shape)}"
            )

        # 保险：把历史噪声先 clamp 一次，防止之前 epoch 的越界残留
        delta_tbl = delta_tbl.clamp(-eps, eps)

        last_loss = torch.tensor(0.0, device=device)

        # -------- PGD 内层循环 with RWP --------
        with torch.enable_grad():
            for _ in range(max(1, num_steps)):
                if rwp_enabled:
                    # RWP: 对 U_noise 次权重采样取平均梯度
                    grad_accum = torch.zeros_like(delta_tbl)

                    for _u in range(U_noise):
                        noises = self._add_weight_noise_(s_model, rwp_sigma)

                        # x + delta，保证输入 surrogate 的图像始终在 [0,1]
                        perturb_img = (x + delta_tbl).clamp(0.0, 1.0).detach().requires_grad_(True)
                        xn = perturb_img.clone()
                        self._norm_inplace(xn, mean, std)

                        out = s_model(xn)
                        logits = out[0] if isinstance(out, (tuple, list)) else out

                        loss_u = seg_loss_fn(logits, y.unsqueeze(1))
                        (g_u,) = torch.autograd.grad(
                            loss_u, perturb_img, retain_graph=False, create_graph=False
                        )
                        grad_accum.add_(g_u)
                        last_loss = loss_u.detach()

                        self._remove_weight_noise_(s_model, noises)

                    g = grad_accum / float(U_noise)
                else:
                    # 普通 PGD（无RWP）
                    perturb_img = (x + delta_tbl).clamp(0.0, 1.0).detach().requires_grad_(True)
                    xn = perturb_img.clone()
                    self._norm_inplace(xn, mean, std)

                    out = s_model(xn)
                    logits = out[0] if isinstance(out, (tuple, list)) else out

                    loss = seg_loss_fn(logits, y.unsqueeze(1))
                    last_loss = loss.detach()

                    (g,) = torch.autograd.grad(
                        loss, perturb_img, retain_graph=False, create_graph=False
                    )

                # PGD 梯度下降（min-min）
                delta_tbl = delta_tbl - step_size * g.sign()

                # **唯一且强制的 L_inf 投影**
                delta_tbl = delta_tbl.clamp(-eps, eps)

        # -------- 写回 noise backend --------
        nb.commit_batch(keys_list, delta_tbl.detach().cpu())

        delta_linf = float(delta_tbl.detach().abs().max().cpu())
        return {
            "noise_loss": float(last_loss.cpu()),
            "delta_linf": delta_linf,
        }