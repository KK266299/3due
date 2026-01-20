# file: src/core/ue_algos/noise_coherent.py
"""
Noise Coherent: UNet-based P (Complex Spectral Parameters) Update

核心思想:
  - 使用 UNet 输出 P 的实部和虚部增量 (delta_P_real, delta_P_imag)
  - 维护一个可学习的 P 参数，通过 UNet 的输出进行更新
  - P 更新后应用频域 mask，然后 IFFT 得到空域扰动
  - 支持 ROI 软边缘 mask，梯度可以回传更新 UNet 和 P

Loss 计算方法:
  1. 通过 UNet 生成 P 的实部和虚部增量
  2. 更新 P: P_new = P + scale * UNet_output (additive mode)
  3. 应用频域 mask: P_masked = P_new * M
  4. IFFT 到空域: delta = IFFT(P_masked)
  5. 应用 ROI 软边缘: delta = delta * ROI_mask
  6. 生成扰动图像: x_perturbed = clip(x + delta, 0, 1)
  7. 前向 surrogate: logits = surrogate(normalize(x_perturbed))
  8. 计算分割损失: loss = DiceCELoss(logits, label)
  9. 反向传播更新 UNet 和 P

软边缘梯度回传:
  - ROI_mask 通过高斯平滑生成，值在 [0, 1] 范围内
  - 软边缘区域 (0 < ROI_mask < 1) 的扰动会被缩放
  - 梯度回传时，软边缘区域的梯度会乘以 ROI_mask
  - 这意味着软边缘部分**可以参与梯度更新**，但梯度会被衰减
  - 边缘越软（gaussian_sigma 越大），梯度衰减越平滑
"""
from __future__ import annotations
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
from monai.losses import DiceCELoss
from monai.networks.nets import UNet

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
    ROI 空域门控，带高斯平滑边缘。

    梯度回传:
      - 软边缘通过高斯平滑生成，完全可微
      - 梯度在边缘区域平滑衰减
      - 边缘软化程度由 gaussian_sigma 控制
    """

    def __init__(self, roi_config: DictConfig):
        super().__init__()
        self.dilate_kernel_size = int(get_config(roi_config, "dilate_kernel_size", 3))
        self.gaussian_sigma = float(get_config(roi_config, "gaussian_sigma", 2.0))

    def forward(self, label: torch.Tensor) -> torch.Tensor:
        """
        从分割标签生成平滑的 ROI mask。

        Args:
            label: [B, D, H, W] or [B, 1, D, H, W] 分割标签

        Returns:
            roi_mask: [B, 1, D, H, W] 平滑 ROI mask，值在 [0, 1]
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

        # 高斯平滑用于软边缘（关键：保持梯度流动）
        if self.gaussian_sigma > 0:
            kernel = _gaussian_kernel_3d(
                self.gaussian_sigma, device=mask.device, dtype=mask.dtype
            )
            padding = kernel.shape[-1] // 2
            mask = F.conv3d(mask, kernel, padding=padding)
            # 归一化到 [0, 1]
            mask = mask / mask.max().clamp_min(1e-6)

        return mask


class UNetPGenerator(nn.Module):
    """
    使用 UNet 生成 P 的实部和虚部。

    输入: 图像 x [B, C_in, D, H, W]
    输出: P 的实部和虚部 [B, 2*C_out, D, H, W]
           前 C_out 个通道是实部，后 C_out 个通道是虚部
    """

    def __init__(self, unet_config: DictConfig):
        super().__init__()
        in_channels = int(get_config(unet_config, "in_channels", 4))
        out_channels = int(get_config(unet_config, "out_channels", 8))
        spatial_dims = int(get_config(unet_config, "spatial_dims", 3))
        channels = list(get_config(unet_config, "channels", [32, 64, 128, 256]))
        strides = list(get_config(unet_config, "strides", [2, 2, 2]))
        num_res_units = int(get_config(unet_config, "num_res_units", 2))
        norm = str(get_config(unet_config, "norm", "INSTANCE"))
        act = str(get_config(unet_config, "act", "RELU"))
        dropout = float(get_config(unet_config, "dropout", 0.0))

        # UNet 输出实部和虚部，所以输出通道是 out_channels * 2
        self.out_channels = out_channels
        self.unet = UNet(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels * 2,  # 实部 + 虚部
            channels=channels,
            strides=strides,
            num_res_units=num_res_units,
            norm=norm,
            act=act,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        生成 P 的实部和虚部。

        Args:
            x: [B, C_in, D, H, W] 输入图像

        Returns:
            P_real: [B, C_out, D, H, W] P 的实部
            P_imag: [B, C_out, D, H, W] P 的虚部
        """
        output = self.unet(x)  # [B, 2*C_out, D, H, W]

        # 分离实部和虚部
        P_real = output[:, : self.out_channels]  # [B, C_out, D, H, W]
        P_imag = output[:, self.out_channels :]  # [B, C_out, D, H, W]

        return P_real, P_imag


class SpectralPerturbationModule(nn.Module):
    """
    频域扰动模块：维护 P 参数，通过 UNet 更新，应用频域 mask，IFFT 到空域。

    流程:
      1. 输入图像 x [B, C_in, D, H, W]
      2. UNet 生成 P 的实部和虚部增量 [B, C_out, D, H, W]
      3. 更新 P: P_new = P + scale * delta_P (或 P_new = delta_P)
      4. 应用频域 mask: P_masked = P_new * M
      5. IFFT 到空域: delta = IFFT(P_masked)
      6. 返回空域扰动 delta [B, C_out, D, H, W]
    """

    def __init__(
        self,
        unet_config: DictConfig,
        p_init_config: DictConfig,
        p_update_config: DictConfig,
        spectral_mask_config: DictConfig,
        image_shape: Tuple[int, int, int, int],  # (C, D, H, W)
        device: torch.device,
    ):
        super().__init__()
        self.C, self.D, self.H, self.W = image_shape

        # UNet 生成器
        self.unet_generator = UNetPGenerator(unet_config).to(device)

        # P 初始化
        p_enabled = bool(get_config(p_init_config, "enabled", True))
        init_scale = float(get_config(p_init_config, "init_scale", 0.01))
        learnable = bool(get_config(p_init_config, "learnable", True))

        if p_enabled:
            P_real_init = torch.randn(self.C, self.D, self.H, self.W, device=device) * init_scale
            P_imag_init = torch.randn(self.C, self.D, self.H, self.W, device=device) * init_scale
            if learnable:
                self.P_real = nn.Parameter(P_real_init)
                self.P_imag = nn.Parameter(P_imag_init)
            else:
                self.register_buffer("P_real", P_real_init)
                self.register_buffer("P_imag", P_imag_init)
        else:
            # 从零开始
            if learnable:
                self.P_real = nn.Parameter(torch.zeros(self.C, self.D, self.H, self.W, device=device))
                self.P_imag = nn.Parameter(torch.zeros(self.C, self.D, self.H, self.W, device=device))
            else:
                self.register_buffer("P_real", torch.zeros(self.C, self.D, self.H, self.W, device=device))
                self.register_buffer("P_imag", torch.zeros(self.C, self.D, self.H, self.W, device=device))

        # P 更新策略
        self.update_mode = str(get_config(p_update_config, "mode", "additive"))
        self.update_scale = float(get_config(p_update_config, "update_scale", 0.1))

        # 固定频域 mask
        mask = SpectralMaskGenerator.build_mask(
            self.D, self.H, self.W, spectral_mask_config, device, torch.float32
        )
        self.register_buffer("spectral_mask", mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        生成空域扰动。

        Args:
            x: [B, C_in, D, H, W] 输入图像

        Returns:
            delta: [B, C, D, H, W] 空域扰动
        """
        B = x.shape[0]

        # 1. UNet 生成 P 的实部和虚部增量
        delta_P_real, delta_P_imag = self.unet_generator(x)  # [B, C, D, H, W]

        # 2. 更新 P
        if self.update_mode == "additive":
            # P_new = P + scale * delta_P
            P_real_new = self.P_real.unsqueeze(0) + self.update_scale * delta_P_real
            P_imag_new = self.P_imag.unsqueeze(0) + self.update_scale * delta_P_imag
        elif self.update_mode == "replace":
            # P_new = delta_P
            P_real_new = delta_P_real
            P_imag_new = delta_P_imag
        else:
            raise ValueError(f"Unknown update mode: {self.update_mode}")

        # 3. 构建复数 P
        P_complex = torch.complex(P_real_new, P_imag_new)  # [B, C, D, H, W]

        # 4. 应用频域 mask
        P_masked = P_complex * self.spectral_mask  # [B, C, D, H, W] * [1, D, H, W]

        # 5. IFFT 到空域
        delta = torch.fft.ifftn(P_masked, dim=(-3, -2, -1)).real  # [B, C, D, H, W]

        return delta


@register_plugin("noise_coherent")
class NoiseCoherent:
    """
    Noise Coherent: UNet-based P (Complex Spectral Parameters) Update

    核心流程:
      1. 使用 UNet 生成 P 的实部和虚部增量
      2. 更新 P（additive 或 replace 模式）
      3. 应用频域 mask
      4. IFFT 到空域得到扰动
      5. 应用 ROI 软边缘 mask
      6. 生成扰动图像
      7. 前向 surrogate 计算损失
      8. 反向传播更新 UNet 和 P

    Loss 计算:
      - 使用 DiceCELoss（Dice + CrossEntropy）
      - 目标是最大化 surrogate 的分割损失
      - 梯度回传到 UNet 和 P 参数

    软边缘梯度回传:
      - ROI mask 通过高斯平滑生成，值在 [0, 1]
      - 扰动在边缘区域被平滑缩放: delta_masked = delta * ROI_mask
      - 反向传播时，梯度也会被 ROI_mask 缩放
      - 软边缘区域（0 < mask < 1）的梯度会被衰减但不会完全消失
      - 这保证了边缘区域也能参与学习，避免硬边界的梯度截断
    """

    def __init__(self):
        self._seg_loss: DiceCELoss | None = None
        self._spectral_module: SpectralPerturbationModule | None = None
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

    def _init_components(self, trainer, image_shape: Tuple[int, int, int, int]):
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
        unet_cfg = get_config(params, "unet", DictConfig({}))
        p_init_cfg = get_config(params, "p_init", DictConfig({}))
        p_update_cfg = get_config(params, "p_update", DictConfig({}))
        spectral_cfg = get_config(params, "spectral_mask", DictConfig({}))
        roi_cfg = get_config(params, "roi_gate", DictConfig({}))

        # 初始化频域扰动模块
        self._spectral_module = SpectralPerturbationModule(
            unet_cfg, p_init_cfg, p_update_cfg, spectral_cfg, image_shape, device
        )

        # 初始化 ROI gate
        roi_enabled = bool(get_config(roi_cfg, "enabled", True))
        if roi_enabled:
            self._roi_gate = ROISpatialGate(roi_cfg).to(device)
        else:
            self._roi_gate = None

        # 初始化优化器
        opt_cfg = get_config(params, "optimizer", DictConfig({}))
        lr = float(get_config(opt_cfg, "lr", 1e-3))
        weight_decay = float(get_config(opt_cfg, "weight_decay", 1e-5))
        betas = tuple(get_config(opt_cfg, "betas", (0.9, 0.999)))

        # 收集所有需要优化的参数
        params_to_optimize = list(self._spectral_module.parameters())

        self._optimizer = torch.optim.Adam(
            params_to_optimize, lr=lr, weight_decay=weight_decay, betas=betas
        )

        self._initialized = True
        self.logger.info(
            f"[NoiseCoherent] Initialized: image_shape={image_shape}, "
            f"roi_enabled={roi_enabled}, lr={lr}"
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
        仅更新 surrogate 参数，不更新频域参数。
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

    # ---------------- N-step: 更新频域参数 ---------------- #
    def noise_step_batch(self, trainer, batch) -> Dict[str, float]:
        """
        更新 UNet 和 P 参数以最大化 surrogate 损失。

        流程:
          1. UNet 生成 P 的实部和虚部增量
          2. 更新 P: P_new = P + scale * delta_P (additive mode)
          3. 应用频域 mask: P_masked = P_new * M
          4. IFFT 到空域: delta = IFFT(P_masked)
          5. 应用 ROI gate: delta = delta * ROI_mask
          6. Clip 到 epsilon
          7. 前向冻结的 surrogate
          8. 计算损失并反向传播更新 P
          9. 将最终 noise 存储到 backend

        软边缘梯度回传机制:
          - ROI_mask 是通过高斯平滑生成的，值在 [0, 1] 范围
          - 在边缘区域，mask 值在 0 和 1 之间平滑过渡
          - 扰动被 mask 缩放: delta_masked = delta * ROI_mask
          - 反向传播时，损失对 delta 的梯度为: dL/d(delta) = dL/d(delta_masked) * ROI_mask
          - 因此，软边缘区域的梯度会被衰减但不会完全消失
          - 这允许网络学习如何在边缘区域生成更好的扰动
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

        # 获取 ROI mask（如果启用）
        if self._roi_gate is not None:
            roi_mask = self._roi_gate(y)  # [B, 1, D, H, W]
            roi_mask = roi_mask.expand(-1, C_in, -1, -1, -1)  # [B, C, D, H, W]
        else:
            roi_mask = None

        # 训练频域参数
        last_loss = torch.tensor(0.0, device=device)

        for step in range(max(1, num_steps)):
            # 通过 UNet 生成扰动
            delta = self._spectral_module(x)  # [B, C, D, H, W]

            # 应用 ROI gate（软边缘，梯度可以回传）
            if roi_mask is not None:
                delta = delta * roi_mask

            # Clip 到 epsilon
            delta = delta.clamp(-eps, eps)

            # 创建扰动图像
            perturb_img = (x + delta).clamp(0.0, 1.0)
            xn = perturb_img.clone()
            self._norm_inplace(xn, mean, std)

            # 前向 surrogate
            out = s_model(xn)
            logits = out[0] if isinstance(out, (tuple, list)) else out

            # 计算损失
            loss = seg_loss_fn(logits, y.unsqueeze(1))
            last_loss = loss.detach()

            # 反向传播更新频域参数
            if self._optimizer is None:
                raise RuntimeError("[UE] Optimizer not initialized.")

            self._optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self._optimizer.step()

        # 存储最终 noise 到 backend
        with torch.no_grad():
            final_delta = self._spectral_module(x)  # [B, C, D, H, W]

            # 应用 ROI gate
            if roi_mask is not None:
                final_delta = final_delta * roi_mask

            # Clip 到 epsilon
            final_delta = final_delta.clamp(-eps, eps)

        nb.commit_batch(keys_list, final_delta.detach().cpu())

        delta_linf = float(final_delta.detach().abs().max().cpu())
        return {
            "noise_loss": float(last_loss.cpu()),
            "delta_linf": delta_linf,
        }
