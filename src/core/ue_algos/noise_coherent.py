# file: src/core/ue_algos/noise_coherent.py
"""
Noise Coherent: UNet-based Complex P Optimization for 3D Segmentation UE

核心思想:
  - 使用 P-UNet 直接生成频域参数 P（复数的实部和虚部）
  - 应用固定的频域 mask，然后 IFFT 到空域
  - tanh + epsilon 限制扰动范围，再 clamp
  - 可选的 ROI 软边缘 mask（可开关）
  - 最小化 surrogate 在扰动图像上的分割损失（min-min）

流程:
  1. P-UNet 接收输入图像 x，输出 P_real 和 P_imag
  2. 构建复数 P = P_real + i * P_imag
  3. 应用频域 mask: P_masked = P * M
  4. IFFT 到空域: delta_raw = IFFT(P_masked).real
  5. tanh + epsilon: delta = tanh(delta_raw) * epsilon
  6. clamp: delta = clamp(delta, -eps, eps)
  7. 可选 ROI gate: delta = delta * ROI_mask
  8. 生成扰动图像: x_perturbed = clip(x + delta, 0, 1)
  9. 前向冻结的 surrogate: logits = surrogate(x_perturbed)
  10. 最小化分割损失: loss = DiceCELoss(logits, label)
  11. 反向传播更新 P-UNet 参数

软边缘梯度回传:
  - 当 soft_edge=true 时，ROI_mask 通过高斯平滑生成，值在 [0, 1]
  - 软边缘区域 (0 < ROI_mask < 1) 的扰动会被缩放
  - 梯度回传时，软边缘区域的梯度会乘以 ROI_mask，梯度平滑衰减
  - 当 soft_edge=false 时，使用硬边缘（ROI_mask 为 0 或 1）
"""
from __future__ import annotations
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
from monai.losses import DiceCELoss
from monai.networks.nets import UNet as MonaiUNet

from ...registry import register_plugin
from ...utils.config import get_config, require_config
from ...utils.logger import get_logger


def _build_p_unet(cfg: DictConfig, in_channels: int, spatial_dims: int = 3) -> nn.Module:
    """
    Build P-UNet for generating complex P (real and imaginary parts).

    Input: original image [B, C, D, H, W]
    Output: P_real and P_imag concatenated [B, 2*C, D, H, W]
    """
    channels = list(get_config(cfg, "channels", [16, 32, 64, 128]))
    strides = list(get_config(cfg, "strides", [2, 2, 2]))
    num_res_units = int(get_config(cfg, "num_res_units", 1))
    act = get_config(cfg, "act", "LEAKYRELU")
    norm = get_config(cfg, "norm", "INSTANCE")
    dropout = float(get_config(cfg, "dropout", 0.0))

    # Output 2*C channels: first C for P_real, second C for P_imag
    unet = MonaiUNet(
        spatial_dims=spatial_dims,
        in_channels=in_channels,
        out_channels=in_channels * 2,
        channels=channels,
        strides=strides,
        num_res_units=num_res_units,
        act=act,
        norm=norm,
        dropout=dropout,
    )
    return unet


class PUNetWrapper(nn.Module):
    """
    Wrapper for P-UNet that outputs complex P components.

    The UNet outputs [B, 2*C, D, H, W], which is split into:
    - P_real: [B, C, D, H, W]
    - P_imag: [B, C, D, H, W]
    """
    def __init__(self, unet: nn.Module, in_channels: int):
        super().__init__()
        self.unet = unet
        self.in_channels = in_channels

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: Input image [B, C, D, H, W] in [0, 1]
        Returns:
            P_real: [B, C, D, H, W]
            P_imag: [B, C, D, H, W]
        """
        out = self.unet(x)  # [B, 2*C, D, H, W]
        P_real = out[:, :self.in_channels]  # [B, C, D, H, W]
        P_imag = out[:, self.in_channels:]  # [B, C, D, H, W]
        return P_real, P_imag


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


class SoftROIMask(nn.Module):
    """
    ROI 空域门控，支持硬边缘和软边缘（高斯平滑）。

    - soft_edge=false: 硬边缘（ROI_mask 为 0 或 1）
    - soft_edge=true: 软边缘（高斯平滑，ROI_mask 在 [0, 1]）

    梯度回传:
      - 硬边缘: 梯度在 ROI 边界截断
      - 软边缘: 梯度在边缘区域平滑衰减，完全可微
    """

    def __init__(
        self,
        soft_edge: bool = False,
        dilate_kernel_size: int = 3,
        gaussian_sigma: float = 2.0,
    ):
        super().__init__()
        self.soft_edge = soft_edge
        self.dilate_kernel_size = dilate_kernel_size
        self.gaussian_sigma = gaussian_sigma
        self._gaussian_kernel = None

    def _build_gaussian_kernel(
        self,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Build 3D Gaussian kernel for smoothing."""
        sigma = self.gaussian_sigma
        if sigma <= 0:
            return None

        kernel_size = int(6 * sigma + 1)
        if kernel_size % 2 == 0:
            kernel_size += 1

        coords = torch.arange(kernel_size, device=device, dtype=dtype) - kernel_size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = g / g.sum()

        # Create separable 3D kernel
        kernel_3d = g.view(-1, 1, 1) * g.view(1, -1, 1) * g.view(1, 1, -1)
        kernel_3d = kernel_3d / kernel_3d.sum()

        return kernel_3d.view(1, 1, kernel_size, kernel_size, kernel_size)

    def forward(self, label: torch.Tensor, num_channels: int) -> torch.Tensor:
        """
        从分割标签生成 ROI mask。

        Args:
            label: [B, D, H, W] or [B, 1, D, H, W] 分割标签
            num_channels: 输出通道数

        Returns:
            roi_mask: [B, C, D, H, W] ROI mask
                - soft_edge=false: 值为 0 或 1（硬边缘）
                - soft_edge=true: 值在 [0, 1]（软边缘）
        """
        device = label.device
        dtype = torch.float32

        # 确保 label 是 [B, D, H, W]
        if label.dim() == 5:
            label = label.squeeze(1)

        # 创建二值 mask: ROI = label > 0
        mask = (label > 0).float()  # [B, D, H, W]
        mask = mask.unsqueeze(1)  # [B, 1, D, H, W]

        # soft_edge=true: 膨胀 + 高斯模糊
        # soft_edge=false: 只使用二值化硬边缘
        if self.soft_edge:
            # 形态学膨胀
            if self.dilate_kernel_size > 0:
                k = self.dilate_kernel_size
                pad = k // 2
                mask = F.max_pool3d(mask, kernel_size=k, stride=1, padding=pad)

            # 高斯模糊
            if self.gaussian_sigma > 0:
                if self._gaussian_kernel is None:
                    self._gaussian_kernel = self._build_gaussian_kernel(device, dtype)
                kernel = self._gaussian_kernel.to(device=device, dtype=dtype)
                k_size = kernel.shape[-1]
                pad = k_size // 2

                mask = F.pad(mask, (pad, pad, pad, pad, pad, pad), mode='replicate')
                mask = F.conv3d(mask, kernel)

            # 归一化到 [0, 1]
            mask = mask / mask.max().clamp_min(1e-6)

        # Expand to [B, C, D, H, W]
        mask = mask.expand(-1, num_channels, -1, -1, -1)

        return mask


@register_plugin("noise_coherent")
class NoiseCoherent:
    """
    Noise Coherent: UNet-based Complex P Optimization for 3D Segmentation UE

    核心流程:
      1. P-UNet 接收输入图像，输出 P_real 和 P_imag
      2. 应用频域 mask，IFFT 到空域
      3. tanh + epsilon 限制扰动，再 clamp
      4. 可选 ROI gate（软边缘或硬边缘）
      5. 生成扰动图像
      6. 最小化 surrogate 的分割损失
      7. 反向传播更新 P-UNet 参数

    Loss 计算:
      - 使用 DiceCELoss（Dice + CrossEntropy）
      - 目标: 最小化 surrogate 在扰动图像上的分割损失
      - 梯度回传到 P-UNet 参数

    软边缘梯度回传:
      - soft_edge=true: 软边缘，梯度平滑衰减
      - soft_edge=false: 硬边缘，梯度在边界截断
    """

    def __init__(self):
        self._seg_loss: DiceCELoss | None = None
        self._p_unet: PUNetWrapper | None = None
        self._opt_p_unet: torch.optim.Optimizer | None = None
        self._roi_mask_builder: SoftROIMask | None = None
        self._initialized: bool = False

        # 配置缓存
        self._spectral_mask_config: DictConfig | None = None
        self._spectral_mask: torch.Tensor | None = None
        self._epsilon: float = 0.0
        self._device: torch.device | None = None
        self._roi_aware: bool = True

        self.logger = get_logger()

    @staticmethod
    def _norm_inplace(x: torch.Tensor, mean, std):
        """原地归一化（按通道）。"""
        for c, (m, s) in enumerate(zip(mean, std)):
            x[:, c].sub_(float(m)).div_(float(s))
        return x

    def _init_components(self, trainer, in_channels: int, image_shape: tuple):
        """
        初始化组件（延迟初始化，仅一次）。

        Args:
            trainer: UETrainer 实例
            in_channels: 输入通道数
            image_shape: (D, H, W)
        """
        if self._initialized:
            return

        cfg = trainer.config
        device = trainer.device
        D, H, W = image_shape

        self._device = device

        # 获取配置
        algo = require_config(cfg, "ue.algorithm")
        params = require_config(algo, "params")
        self._epsilon = float(get_config(params, "epsilon", 8 / 255.0))
        self._spectral_mask_config = get_config(params, "spectral_mask", DictConfig({}))

        # 初始化频域 mask（固定）
        self._spectral_mask = SpectralMaskGenerator.build_mask(
            D, H, W, self._spectral_mask_config, device, torch.float32
        )

        # 初始化 P-UNet
        p_unet_cfg = get_config(cfg, "ue.p_unet", DictConfig({}))
        base_unet = _build_p_unet(p_unet_cfg, in_channels, spatial_dims=3)
        self._p_unet = PUNetWrapper(base_unet, in_channels)
        self._p_unet = self._p_unet.to(device)

        # 初始化 P-UNet 优化器
        opt_cfg = get_config(p_unet_cfg, "optimizer", DictConfig({}))
        lr = float(get_config(opt_cfg, "lr", 1e-4))
        weight_decay = float(get_config(opt_cfg, "weight_decay", 1e-5))
        betas = tuple(get_config(opt_cfg, "betas", (0.9, 0.999)))

        self._opt_p_unet = torch.optim.Adam(
            self._p_unet.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=betas,
        )

        # ROI mask 配置
        self._roi_aware = bool(get_config(params, "roi_aware", True))
        soft_edge = bool(get_config(params, "soft_edge", False))
        dilate_kernel_size = int(get_config(params, "dilate_kernel_size", 3))
        gaussian_sigma = float(get_config(params, "gaussian_sigma", 2.0))

        self._roi_mask_builder = SoftROIMask(
            soft_edge=soft_edge,
            dilate_kernel_size=dilate_kernel_size,
            gaussian_sigma=gaussian_sigma,
        )

        self._initialized = True
        self.logger.info(
            f"[NoiseCoherent] Initialized: in_channels={in_channels}, image_shape={image_shape}, "
            f"epsilon={self._epsilon:.6f}, roi_aware={self._roi_aware}, soft_edge={soft_edge}, lr={lr}"
        )

    def _generate_delta(self, x: torch.Tensor) -> torch.Tensor:
        """
        使用 P-UNet 生成扰动。

        Args:
            x: 输入图像 [B, C, D, H, W]

        Returns:
            delta: [B, C, D, H, W] 扰动（未应用 ROI mask）
        """
        # P-UNet 生成 P_real 和 P_imag
        P_real, P_imag = self._p_unet(x)  # [B, C, D, H, W] each

        # 构建复数 P
        P_complex = torch.complex(P_real, P_imag)  # [B, C, D, H, W]

        # 应用频域 mask
        # spectral_mask: [1, D, H, W] -> expand to [1, 1, D, H, W]
        mask = self._spectral_mask.unsqueeze(0)  # [1, 1, D, H, W]
        P_masked = P_complex * mask  # [B, C, D, H, W]

        # IFFT 到空域
        delta_raw = torch.fft.ifftn(P_masked, dim=(-3, -2, -1)).real  # [B, C, D, H, W]

        # tanh + epsilon
        delta = torch.tanh(delta_raw) * self._epsilon  # [B, C, D, H, W]

        # clamp
        delta = delta.clamp(-self._epsilon, self._epsilon)

        return delta

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
        仅更新 surrogate 参数，不更新 P-UNet。
        使用来自 backend 的 noise。
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
        image_shape = tuple(x.shape[2:])  # (D, H, W)

        self._init_components(trainer, C_in, image_shape)

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

    # ---------------- N-step: 更新 P-UNet ---------------- #
    def noise_step_batch(self, trainer, batch) -> Dict[str, float]:
        """
        更新 P-UNet 参数以最小化 surrogate 损失。

        流程:
          1. P-UNet 生成 P_real 和 P_imag
          2. 构建复数 P，应用频域 mask
          3. IFFT 到空域，tanh + epsilon，clamp
          4. 应用 ROI gate: delta = delta * ROI_mask
          5. 生成扰动图像: x_perturbed = clip(x + delta, 0, 1)
          6. 前向冻结的 surrogate: logits = surrogate(x_perturbed)
          7. 计算损失: loss = DiceCELoss(logits, label)
          8. 反向传播更新 P-UNet 参数
          9. 将最终 noise 存储到 backend

        软边缘梯度回传:
          - soft_edge=true: ROI_mask 通过高斯平滑生成，值在 [0, 1]
            梯度: dL/d(delta) = dL/d(delta_masked) * ROI_mask（平滑衰减）
          - soft_edge=false: ROI_mask 为 0 或 1（硬边缘）
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
        image_shape = tuple(x.shape[2:])  # (D, H, W)

        self._init_components(trainer, C_in, image_shape)

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
        if self._roi_aware:
            roi_mask = self._roi_mask_builder(y, C_in).to(device)  # [B, C, D, H, W]
        else:
            roi_mask = None

        # 训练 P-UNet
        self._p_unet.train()
        last_loss = torch.tensor(0.0, device=device)

        for _ in range(max(1, num_steps)):
            # 生成扰动
            delta_raw = self._generate_delta(x)  # [B, C, D, H, W]

            # 应用 ROI mask（如果启用）
            if roi_mask is not None:
                delta = delta_raw * roi_mask
            else:
                delta = delta_raw

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

            # 反向传播更新 P-UNet
            self._opt_p_unet.zero_grad(set_to_none=True)
            loss.backward()
            self._opt_p_unet.step()

        # 存储最终 noise 到 backend（与 noise_slice_frequence 一致）
        self._p_unet.eval()
        with torch.no_grad():
            final_delta_raw = self._generate_delta(x)
            if roi_mask is not None:
                final_delta = final_delta_raw * roi_mask
            else:
                final_delta = final_delta_raw
            final_delta = final_delta.clamp(-eps, eps)

        nb.commit_batch(keys_list, final_delta.detach().cpu())

        delta_linf = float(final_delta.detach().abs().max().cpu())
        return {
            "noise_loss": float(last_loss.cpu()),
            "delta_linf": delta_linf,
        }
