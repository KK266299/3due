# file: src/ue_algos/tap.py
"""
TAP: Adversarial Examples Make Strong Poisons
适配 3D 医学图像分割任务 (BraTS19)

核心思想:
  - 与 min-min 方法相反，TAP 使用 PGD **梯度上升** 来最大化 loss
  - 这样生成的噪声是"有毒的"，会使模型在训练时难以学习有效特征
  - 对于分割任务，使用 DiceCE loss

方法对比:
  - min-min: delta = delta - step_size * grad.sign()  (最小化 loss)
  - TAP:     delta = delta + step_size * grad.sign()  (最大化 loss)

支持两种模式:
  1. 使用预训练权重 (weights_path): 加载权重后冻结，不训练 surrogate
  2. 在线训练 (无 weights_path): 与 min-min 类似，边训练 surrogate 边生成噪声

Reference:
  - Witches' Brew: Industrial Scale Data Poisoning via Gradient Matching (ICLR 2021)
  - Adversarial Examples Make Strong Poisons
  - Safeguarding Medical Image Segmentation Datasets against Unauthorized Training
    via Contour- and Texture-Aware Perturbations
"""
from __future__ import annotations
from typing import Dict, Iterable, List
import os

import torch
from omegaconf import DictConfig
from monai.losses import DiceCELoss

from ...registry import register_plugin
from ...utils.config import get_config, require_config


@register_plugin("tap")
class TAP:
    """
    TAP UE for 3D segmentation (e.g., BraTS19).

    与 min-min 的核心区别：
      - min-min: PGD descent (梯度下降，最小化 loss)
      - TAP: PGD ascent (梯度上升，最大化 loss)

    支持两种模式：
      1. 预训练模式 (weights_path 存在): 加载预训练权重，冻结模型，不训练
      2. 在线训练模式 (无 weights_path): 边训练 surrogate 边生成噪声

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

    @staticmethod
    def _strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """处理 DDP/DataParallel 产生的 'module.' 前缀"""
        if not state_dict:
            return state_dict
        keys = list(state_dict.keys())
        if all(k.startswith("module.") for k in keys):
            return {k[len("module."):]: v for k, v in state_dict.items()}
        return state_dict

    def _lazy_load_surrogates(self, trainer) -> int:
        """
        懒加载预训练权重并冻结模型。
        返回新加载的模型数量。

        如果配置了 weights_path，则加载权重并冻结模型。
        如果没有配置 weights_path，则返回 0（使用在线训练模式）。
        """
        if not hasattr(trainer, "_tap_loaded"):
            trainer._tap_loaded = {}
        cfg = trainer.config
        loaded = 0

        for name, s_model in trainer.surrogates.items():
            if trainer._tap_loaded.get(name, False):
                # 已经加载并冻结
                s_model.eval()
                for p in s_model.parameters():
                    p.requires_grad = False
                continue

            # 读取 weights_path: ue.surrogates.<name>.weights_path
            wpath = get_config(cfg, f"ue.surrogates.{name}.weights_path", None)
            if wpath and isinstance(wpath, str):
                if not os.path.isfile(wpath):
                    raise FileNotFoundError(
                        f"[UE][TAP] weights_path not found for surrogate '{name}': {wpath}"
                    )
                obj = torch.load(wpath, map_location=trainer.device, weights_only=False)
                # 支持 raw state_dict 或包含 'state_dict' 的字典
                state = obj.get("state_dict", obj) if isinstance(obj, dict) else obj
                state = self._strip_module_prefix(state)
                missing, unexpected = s_model.load_state_dict(state, strict=False)
                if missing or unexpected:
                    print(f"[UE][TAP] Loaded '{name}' with non-strict keys. "
                          f"missing={len(missing)}, unexpected={len(unexpected)}")
                print(f"[UE][TAP] Successfully loaded pretrained weights for '{name}' from {wpath}")
                loaded += 1

                # 有预训练权重时冻结模型
                s_model.eval()
                for p in s_model.parameters():
                    p.requires_grad = False
                trainer._tap_loaded[name] = True

        if not trainer.surrogates:
            raise RuntimeError("[UE][TAP] No surrogate bound. Please bind a model.")
        return loaded

    def _has_pretrained_weights(self, trainer) -> bool:
        """检查是否配置了预训练权重"""
        cfg = trainer.config
        for name in trainer.surrogates.keys():
            wpath = get_config(cfg, f"ue.surrogates.{name}.weights_path", None)
            if wpath and isinstance(wpath, str):
                return True
        return False

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

    # ---------------- Surrogate-step ---------------- #
    def surrogate_step_batch(self, trainer, batch) -> Dict[str, float]:
        """
        Surrogate step 行为取决于是否有预训练权重：

        1. 有 weights_path: 懒加载权重 + 冻结，不训练（类似原 ReID TAP）
        2. 无 weights_path: 训练 surrogate（类似 min-min）
        """
        # 检查是否使用预训练模式
        if self._has_pretrained_weights(trainer):
            # ========== 预训练模式：加载权重，冻结模型，不训练 ==========
            n_loaded = self._lazy_load_surrogates(trainer)
            return {
                "surrogate_loaded": float(n_loaded),
                "surrogate_loss": 0.0,
                "loss": 0.0,
            }

        # ========== 在线训练模式：与 min-min 类似 ==========
        cfg = trainer.config
        device = trainer.device
        nb = trainer.noise_backend
        if nb is None:
            raise RuntimeError("[UE][TAP] noise_backend is required.")

        # data
        x = batch["image"].to(device).float()          # [B,C,...]
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
                f"[UE][TAP] noise shape mismatch: noise {tuple(delta.shape)} vs input {tuple(x.shape)}"
            )

        # select surrogate and optimizer
        if not trainer.surrogates:
            raise RuntimeError("[UE][TAP] No surrogate bound.")
        name, s_model = next(iter(trainer.surrogates.items()))
        opt = trainer.opt_surrogates.get(name, None)
        if opt is None:
            raise RuntimeError(f"[UE][TAP] No optimizer for surrogate '{name}'.")

        seg_loss_fn = self._get_seg_loss(trainer)

        s_model.train()
        for p in s_model.parameters():
            p.requires_grad = True

        # forward with noisy input
        noisy = (x + delta).clamp(0.0, 1.0)
        xn = noisy.clone()
        self._norm_inplace(xn, mean, std)

        out = s_model(xn)
        logits = out[0] if isinstance(out, (tuple, list)) else out  # [B,C_seg,...]

        loss = seg_loss_fn(logits, y.unsqueeze(1))

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        loss_val = float(loss.detach().cpu())
        return {
            "surrogate_loss": loss_val,
            "loss": loss_val,
        }

    # ---------------- N-step：Update noise (PGD ASCENT, sample-wise) ---------------- #
    def noise_step_batch(self, trainer, batch) -> Dict[str, float]:
        """
        对 sample-wise 噪声做 PGD **上升**（最大化 DiceCE Loss）。

        核心区别（vs min-min）：
          - min-min: delta = delta - step_size * g.sign()  # descent
          - TAP:     delta = delta + step_size * g.sign()  # ASCENT

        TAP 方法通过最大化 loss 来生成"有毒"噪声，使得：
          1. 带噪声的样本难以被模型正确学习
          2. 模型在这些样本上的泛化性能下降
        """
        cfg = trainer.config
        device = trainer.device
        nb = trainer.noise_backend
        if nb is None:
            raise RuntimeError("[UE][TAP] noise_backend is required.")

        # 如果使用预训练模式，确保权重已加载
        if self._has_pretrained_weights(trainer):
            self._lazy_load_surrogates(trainer)

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
            raise RuntimeError("[UE][TAP] No surrogate bound.")
        _, s_model = next(iter(trainer.surrogates.items()))
        s_model.eval()
        for p in s_model.parameters():
            p.requires_grad = False

        # -------- init / clamp noise --------
        delta_tbl = nb.batch_noise(keys_list).to(device).float()  # [N, C_in, ...]
        if delta_tbl.shape[:2] != x.shape[:2]:
            raise RuntimeError(
                f"[UE][TAP] noise shape mismatch: noise {tuple(delta_tbl.shape)} vs input {tuple(x.shape)}"
            )

        # 保险：把历史噪声先 clamp 一次，防止之前 epoch 的越界残留
        delta_tbl = delta_tbl.clamp(-eps, eps)

        last_loss = torch.tensor(0.0, device=device)

        # -------- PGD 内层循环 (ASCENT) --------
        with torch.enable_grad():
            for _ in range(max(1, num_steps)):
                # x + delta，保证输入 surrogate 的图像始终在 [0,1]
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

                # ============================================================
                # TAP 核心：PGD 梯度**上升**（与 min-min 相反）
                # min-min: delta = delta - step_size * g.sign()  # 最小化 loss
                # TAP:     delta = delta + step_size * g.sign()  # 最大化 loss
                # ============================================================
                delta_tbl = delta_tbl + step_size * g.sign()

                # **唯一且强制的 L_inf 投影**
                delta_tbl = delta_tbl.clamp(-eps, eps)

        # -------- 写回 noise backend --------
        nb.commit_batch(keys_list, delta_tbl.detach().cpu())

        delta_linf = float(delta_tbl.detach().abs().max().cpu())
        return {
            "noise_loss": float(last_loss.cpu()),
            "delta_linf": delta_linf,
        }