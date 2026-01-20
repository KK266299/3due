# file: src/core/ue_algos/noise_coherent.py
"""
Noise Coherent: Learnable P (Complex Spectral Parameters) for 3D Segmentation UE

核心思想:
  - 维护可学习的 P 参数（频域，实部和虚部）
  - 应用固定的频域 mask，然后 IFFT 到空域
  - tanh + epsilon 限制扰动范围
  - 可选的 ROI 软边缘 mask（可开关）
  - 最小化 surrogate 在扰动图像上的分割损失（min-min）

流程:
  1. 初始化可学习的 P_real 和 P_imag
  2. 构建复数 P = P_real + i * P_imag
  3. 应用频域 mask: P_masked = P * M
  4. IFFT 到空域: delta_raw = IFFT(P_masked).real
  5. tanh + epsilon: delta = tanh(delta_raw) * epsilon
  6. 可选 ROI gate: delta = delta * ROI_mask
  7. 生成扰动图像: x_perturbed = clip(x + delta, 0, 1)
  8. 前向冻结的 surrogate: logits = surrogate(x_perturbed)
  9. 最小化分割损失: loss = DiceCELoss(logits, label)
  10. 反向传播更新 P

软边缘梯度回传:
  - 当 roi_gate.enabled=true 时，ROI_mask 通过高斯平滑生成，值在 [0, 1]
  - 软边缘区域 (0 < ROI_mask < 1) 的扰动会被缩放
  - 梯度回传时，软边缘区域的梯度会乘以 ROI_mask，梯度平滑衰减
  - 当 roi_gate.enabled=false 时，使用硬边缘（ROI_mask 为 0 或 1）
"""
from __future__ import annotations
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
from monai.losses import DiceCELoss

from ...registry import register_plugin
from ...utils.config import get_config, require_config
from ...utils.logger import get_logger


def _gaussian_kernel_3d(sigma: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """
    创建 3D 高斯卷积核用于平滑。

    Args:
        sigma: 高斯标准差
        device: torch device
        dtype: torch dtype

    Returns:
        kernel: [1, 1, K, K, K] 高斯卷积核
    """
    if sigma <= 0:
        raise ValueError("sigma must be > 0 for gaussian kernel.")

    radius = int(round(2.0 * sigma))
    coords = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g = g / g.sum()

    kernel_1d = g.view(1, 1, -1)
    kernel_3d = (
        kernel_1d[:, :, :, None, None]
        * kernel_1d[:, :, None, :, None]
        * kernel_1d[:, :, None, None, :]
    )
    return kernel_3d


class SpectralMaskGenerator:
    """
    生成固定的频域 mask。

    约束:
      - Z 轴（层间方向）: 低频 (|f_z| <= z_max)
      - XY 平面（层内）: 带通 (xy_min <= |f_xy| <= xy_max)
    """

    @staticmethod
    def build_mask(
        D: int,
        H: int,
        W: int,
        cfg: DictConfig,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """
        构建 3D 频域 mask [1, D, H, W]。
        """
        z_max = float(get_config(cfg, "z_max", 0.125))
        xy_min = float(get_config(cfg, "xy_min", 0.05))
        xy_max = float(get_config(cfg, "xy_max", 0.3))

        # 构建频率网格
        fz = torch.fft.fftfreq(D, device=device, dtype=dtype).abs()  # [D]
        fy = torch.fft.fftfreq(H, device=device, dtype=dtype).abs()  # [H]
        fx = torch.fft.fftfreq(W, device=device, dtype=dtype).abs()  # [W]

        # 创建 3D meshgrid
        zz, yy, xx = torch.meshgrid(fz, fy, fx, indexing="ij")  # [D, H, W]

        # Z 轴约束: 低频
        mask_z = (zz <= z_max)

        # XY 平面约束: 带通（径向频率）
        fxy = torch.sqrt(yy**2 + xx**2)
        mask_xy = (fxy >= xy_min) & (fxy <= xy_max)

        # 组合约束
        mask = (mask_z & mask_xy).to(dtype=dtype)

        return mask.unsqueeze(0)  # [1, D, H, W]


class ROISpatialGate(nn.Module):
    """
    ROI 空域门控，支持硬边缘和软边缘（高斯平滑）。

    - enabled=false: 硬边缘（ROI_mask 为 0 或 1）
    - enabled=true: 软边缘（高斯平滑，ROI_mask 在 [0, 1]）

    梯度回传:
      - 硬边缘: 梯度在 ROI 边界截断
      - 软边缘: 梯度在边缘区域平滑衰减，完全可微
    """

    def __init__(self, roi_config: DictConfig):
        super().__init__()
        self.enabled = bool(get_config(roi_config, "enabled", False))
        self.dilate_kernel_size = int(get_config(roi_config, "dilate_kernel_size", 3))
        self.gaussian_sigma = float(get_config(roi_config, "gaussian_sigma", 2.0))

    def forward(self, label: torch.Tensor) -> torch.Tensor:
        """
        从分割标签生成 ROI mask。

        Args:
            label: [B, D, H, W] or [B, 1, D, H, W] 分割标签

        Returns:
            roi_mask: [B, 1, D, H, W] ROI mask
                - enabled=false: 值为 0 或 1（硬边缘）
                - enabled=true: 值在 [0, 1]（软边缘）
        """
        # 确保 label 是 [B, D, H, W]
        if label.dim() == 5:
            label = label.squeeze(1)

        # 创建二值 mask: ROI = label > 0
        mask = (label > 0).float()  # [B, D, H, W]

        # 可选: 形态学膨胀
        if self.dilate_kernel_size > 0:
            mask_unsqueezed = mask.unsqueeze(1)  # [B, 1, D, H, W]
            kernel_size = self.dilate_kernel_size
            padding = kernel_size // 2
            mask = F.max_pool3d(
                mask_unsqueezed,
                kernel_size=kernel_size,
                stride=1,
                padding=padding,
            ).squeeze(1)

        # 添加通道维度: [B, 1, D, H, W]
        mask = mask.unsqueeze(1)

        # 高斯平滑用于软边缘（仅当 enabled=true）
        if self.enabled and self.gaussian_sigma > 0:
            kernel = _gaussian_kernel_3d(
                self.gaussian_sigma, device=mask.device, dtype=mask.dtype
            )
            padding = kernel.shape[-1] // 2
            mask = F.conv3d(mask, kernel, padding=padding)
            # 归一化到 [0, 1]
            mask = mask / mask.max().clamp_min(1e-6)

        return mask


class SpectralPerturbation(nn.Module):
    """
    频域扰动模块：维护可学习的 P（实部和虚部），应用频域 mask，IFFT 到空域。

    流程:
      1. P = P_real + i * P_imag  [C, D, H, W]
      2. 应用频域 mask: P_masked = P * M
      3. IFFT 到空域: delta_raw = IFFT(P_masked).real
      4. tanh + epsilon: delta = tanh(delta_raw) * epsilon
    """

    def __init__(
        self,
        p_init_config: DictConfig,
        spectral_mask_config: DictConfig,
        image_shape: tuple,  # (C, D, H, W)
        epsilon: float,
        device: torch.device,
    ):
        super().__init__()
        self.C, self.D, self.H, self.W = image_shape
        self.epsilon = epsilon

        # P 初始化：使用较大的随机值，确保初始 delta 不为 0
        init_scale = float(get_config(p_init_config, "init_scale", 10.0))
        P_real_init = torch.randn(self.C, self.D, self.H, self.W, device=device) * init_scale
        P_imag_init = torch.randn(self.C, self.D, self.H, self.W, device=device) * init_scale

        self.P_real = nn.Parameter(P_real_init)
        self.P_imag = nn.Parameter(P_imag_init)

        # 固定频域 mask
        mask = SpectralMaskGenerator.build_mask(
            self.D, self.H, self.W, spectral_mask_config, device, torch.float32
        )
        self.register_buffer("spectral_mask", mask)

    def forward(self, batch_size: int) -> torch.Tensor:
        """
        生成空域扰动。

        Args:
            batch_size: batch size

        Returns:
            delta: [B, C, D, H, W] 扰动，范围 [-epsilon, epsilon]
        """
        # 1. 构建复数 P
        P_complex = torch.complex(self.P_real, self.P_imag)  # [C, D, H, W]

        # 2. 应用频域 mask
        P_masked = P_complex * self.spectral_mask  # [C, D, H, W] * [1, D, H, W]

        # 3. IFFT 到空域
        delta_raw = torch.fft.ifftn(P_masked, dim=(-3, -2, -1)).real  # [C, D, H, W]

        # 4. tanh + epsilon
        delta = torch.tanh(delta_raw) * self.epsilon  # [C, D, H, W]

        # 5. 扩展到 batch
        delta = delta.unsqueeze(0).expand(batch_size, -1, -1, -1, -1)  # [B, C, D, H, W]

        return delta


@register_plugin("noise_coherent")
class NoiseCoherent:
    """
    Noise Coherent: Learnable P (Complex Spectral Parameters) for 3D Segmentation UE

    核心流程:
      1. 维护可学习的 P_real 和 P_imag
      2. 应用频域 mask，IFFT 到空域
      3. tanh + epsilon 限制扰动
      4. 可选 ROI gate（软边缘或硬边缘）
      5. 生成扰动图像
      6. 最小化 surrogate 的分割损失
      7. 反向传播更新 P

    Loss 计算:
      - 使用 DiceCELoss（Dice + CrossEntropy）
      - 目标: 最小化 surrogate 在扰动图像上的分割损失
      - 梯度回传到 P_real 和 P_imag

    软边缘梯度回传:
      - roi_gate.enabled=true: 软边缘，梯度平滑衰减
      - roi_gate.enabled=false: 硬边缘，梯度在边界截断
    """

    def __init__(self):
        self._seg_loss: DiceCELoss | None = None
        self._spectral_module: SpectralPerturbation | None = None
        self._roi_gate: ROISpatialGate | None = None
        self._optimizer: torch.optim.Optimizer | None = None
        self._initialized: bool = False
        self.logger = get_logger()

    @staticmethod
    def _norm_inplace(x: torch.Tensor, mean, std):
        """原地归一化（按通道）。"""
        for c, (m, s) in enumerate(zip(mean, std)):
            x[:, c].sub_(float(m)).div_(float(s))
        return x

    def _init_components(self, trainer, image_shape: tuple):
        """
        初始化组件（延迟初始化，仅一次）。

        Args:
            trainer: UETrainer 实例
            image_shape: (C, D, H, W)
        """
        if self._initialized:
            return

        cfg = trainer.config
        device = trainer.device
        C, D, H, W = image_shape

        # 获取配置
        algo = require_config(cfg, "ue.algorithm")
        params = require_config(algo, "params")
        eps = float(get_config(params, "epsilon", 8 / 255.0))
        p_init_cfg = get_config(params, "p_init", DictConfig({}))
        spectral_cfg = get_config(params, "spectral_mask", DictConfig({}))
        roi_cfg = get_config(params, "roi_gate", DictConfig({}))

        # 初始化频域扰动模块
        self._spectral_module = SpectralPerturbation(
            p_init_cfg, spectral_cfg, image_shape, eps, device
        )

        # 初始化 ROI gate
        self._roi_gate = ROISpatialGate(roi_cfg).to(device)

        # 初始化优化器
        opt_cfg = get_config(params, "optimizer", DictConfig({}))
        lr = float(get_config(opt_cfg, "lr", 1e-3))
        weight_decay = float(get_config(opt_cfg, "weight_decay", 1e-5))
        betas = tuple(get_config(opt_cfg, "betas", (0.9, 0.999)))

        self._optimizer = torch.optim.Adam(
            self._spectral_module.parameters(), lr=lr, weight_decay=weight_decay, betas=betas
        )

        self._initialized = True
        roi_enabled = bool(get_config(roi_cfg, "enabled", False))
        self.logger.info(
            f"[NoiseCoherent] Initialized: image_shape={image_shape}, "
            f"epsilon={eps:.6f}, roi_soft_edge={roi_enabled}, lr={lr}"
        )

    def _get_seg_loss(self, trainer) -> DiceCELoss:
        """构建与 SegTrainer 一致的 DiceCELoss。"""
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

    # ---------------- Surrogate-step: 更新 surrogate ---------------- #
    def surrogate_step_batch(self, trainer, batch) -> Dict[str, float]:
        """
        仅更新 surrogate 参数，不更新 P。
        使用来自 backend 的 noise（已应用频域 + ROI 约束）。
        """
        cfg = trainer.config
        device = trainer.device
        nb = trainer.noise_backend
        if nb is None:
            raise RuntimeError("[UE] noise_backend is required.")

        x = batch["image"].to(device).float()
        y = batch["label"]
        y = y.to(device).long() if torch.is_tensor(y) else torch.as_tensor(
            y, device=device, dtype=torch.long
        )
        keys = batch["key"]

        B, C_in = x.shape[:2]
        image_shape = tuple(x.shape[1:])  # (C, D, H, W)

        self._init_components(trainer, image_shape)

        mean = tuple(get_config(cfg, "training.data.transforms.mean", (0.0,) * C_in))
        std = tuple(get_config(cfg, "training.data.transforms.std", (1.0,) * C_in))

        delta = nb.batch_noise(list(keys)).to(device).float()
        if delta.shape[:2] != x.shape[:2]:
            raise RuntimeError(
                f"[UE] noise shape mismatch: noise {tuple(delta.shape)} vs input {tuple(x.shape)}"
            )

        if not trainer.surrogates:
            raise RuntimeError("[UE] No surrogate bound.")
        name, s_model = next(iter(trainer.surrogates.items()))
        opt = trainer.opt_surrogates.get(name, None)
        if opt is None:
            raise RuntimeError(f"[UE] No optimizer for surrogate '{name}'.")

        seg_loss_fn = self._get_seg_loss(trainer)

        s_model.train()
        for p in s_model.parameters():
            p.requires_grad = True

        noisy = (x + delta).clamp(0.0, 1.0)
        xn = noisy.clone()
        self._norm_inplace(xn, mean, std)

        out = s_model(xn)
        logits = out[0] if isinstance(out, (tuple, list)) else out

        loss = seg_loss_fn(logits, y.unsqueeze(1))

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        loss_val = float(loss.detach().cpu())
        return {
            "surrogate_loss": loss_val,
            "loss": loss_val,
        }

    # ---------------- N-step: 更新 P ---------------- #
    def noise_step_batch(self, trainer, batch) -> Dict[str, float]:
        """
        更新 P 参数以最小化 surrogate 损失。

        流程:
          1. 生成扰动: delta = tanh(IFFT(P * M)) * epsilon
          2. 应用 ROI gate: delta = delta * ROI_mask
          3. 生成扰动图像: x_perturbed = clip(x + delta, 0, 1)
          4. 前向冻结的 surrogate: logits = surrogate(x_perturbed)
          5. 计算损失: loss = DiceCELoss(logits, label)
          6. 反向传播更新 P
          7. 将最终 noise 存储到 backend

        软边缘梯度回传:
          - roi_gate.enabled=true: ROI_mask 通过高斯平滑生成，值在 [0, 1]
            梯度: dL/d(delta) = dL/d(delta_masked) * ROI_mask（平滑衰减）
          - roi_gate.enabled=false: ROI_mask 为 0 或 1（硬边缘）
            梯度: ROI 外部梯度为 0，内部梯度不变
        """
        cfg = trainer.config
        device = trainer.device
        nb = trainer.noise_backend
        if nb is None:
            raise RuntimeError("[UE] noise_backend is required.")

        x = batch["image"].to(device).float()
        y = batch["label"]
        y = y.to(device).long() if torch.is_tensor(y) else torch.as_tensor(
            y, device=device, dtype=torch.long
        )
        keys = batch["key"]
        keys_list: List[int] = list(keys)

        B, C_in = x.shape[:2]
        image_shape = tuple(x.shape[1:])  # (C, D, H, W)

        self._init_components(trainer, image_shape)

        algo = require_config(cfg, "ue.algorithm")
        params = require_config(algo, "params")
        eps = float(get_config(params, "epsilon", 8 / 255.0))
        num_steps = int(get_config(params, "noise_step", 1))

        mean = tuple(get_config(cfg, "training.data.transforms.mean", (0.0,) * C_in))
        std = tuple(get_config(cfg, "training.data.transforms.std", (1.0,) * C_in))

        seg_loss_fn = self._get_seg_loss(trainer)

        # 冻结 surrogate
        if not trainer.surrogates:
            raise RuntimeError("[UE] No surrogate bound.")
        _, s_model = next(iter(trainer.surrogates.items()))
        s_model.eval()
        for p in s_model.parameters():
            p.requires_grad = False

        # 获取 ROI mask
        roi_mask = self._roi_gate(y)  # [B, 1, D, H, W]
        roi_mask = roi_mask.expand(-1, C_in, -1, -1, -1)  # [B, C, D, H, W]

        # 训练 P
        self._spectral_module.train()
        last_loss = torch.tensor(0.0, device=device)

        for _ in range(max(1, num_steps)):
            # 生成扰动
            delta_raw = self._spectral_module(B)  # [B, C, D, H, W]

            # 应用 ROI gate
            delta = delta_raw * roi_mask

            # 创建扰动图像
            perturb_img = (x + delta).clamp(0.0, 1.0)
            xn = perturb_img.clone()
            self._norm_inplace(xn, mean, std)

            # 前向 surrogate
            out = s_model(xn)
            logits = out[0] if isinstance(out, (tuple, list)) else out

            # 计算损失（最小化）
            loss = seg_loss_fn(logits, y.unsqueeze(1))
            last_loss = loss.detach()

            # 反向传播更新 P
            if self._optimizer is None:
                raise RuntimeError("[UE] Optimizer not initialized.")

            self._optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self._optimizer.step()

        # 存储最终 noise 到 backend
        self._spectral_module.eval()
        with torch.no_grad():
            final_delta_raw = self._spectral_module(B)
            final_delta = final_delta_raw * roi_mask
            final_delta = final_delta.clamp(-eps, eps)

        nb.commit_batch(keys_list, final_delta.detach().cpu())

        delta_linf = float(final_delta.detach().abs().max().cpu())
        return {
            "noise_loss": float(last_loss.cpu()),
            "delta_linf": delta_linf,
        }
