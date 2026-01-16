# file: src/core/ue_algos/unet_noise_slice_grad.py
"""
Inter-Slice Diversity + Intra-Slice Gradient-Guided UNet Noise Generator.

结合两种策略:
  1. 层间差异性约束: 最大化相邻层之间的噪声差异 (利用层间低分辨率)
  2. 层内梯度引导: 根据图像层内(x-y)梯度调制噪声强度 (边缘区域噪声更强)

核心思想:
  - 层间(z方向): 差异大，破坏网络对层间连续性的推断能力
  - 层内(x-y方向): 梯度引导，在图像边缘/纹理变化处噪声更强，平滑区域噪声弱

重要设计:
  - 梯度计算只在H-W平面进行，忽略D(z)方向
  - 这样梯度引导只作用于层内，不干扰层间差异性损失

损失函数:
  L_total = L_seg - λ_inter × L_inter

  其中:
  - L_seg: 分割损失 (min-min攻击)
  - L_inter: 层间差异性损失 (最大化，用负号)

噪声生成:
  δ = UNet(x) × gradient_weight_xy × roi_mask

  其中 gradient_weight_xy 只包含H和W方向的梯度

ROI Mask软边缘:
  - soft_edge=True: 使用高斯平滑创建软边缘ROI mask
  - 流程: 二值化 → 膨胀(dilate_iter次) → 高斯模糊(sigma, kernel_size)
  - 效果: 噪声在ROI边界处渐变过渡，避免硬截断导致的边界伪影

Maximum noise bound: 8/255 ≈ 0.0313725 (L∞ constraint)
"""
from __future__ import annotations
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
from monai.losses import DiceCELoss
from monai.networks.nets import UNet as MonaiUNet

from ...registry import register_plugin
from ...utils.config import get_config, require_config
from ...utils.logger import get_logger


def _create_gaussian_kernel_3d(kernel_size: int, sigma: float, device: torch.device) -> torch.Tensor:
    """创建3D高斯卷积核."""
    coords = torch.arange(kernel_size, dtype=torch.float32, device=device)
    coords -= (kernel_size - 1) / 2.0

    # 1D高斯
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()

    # 3D高斯 = 外积
    kernel = g.view(-1, 1, 1) * g.view(1, -1, 1) * g.view(1, 1, -1)
    kernel = kernel / kernel.sum()

    return kernel.view(1, 1, kernel_size, kernel_size, kernel_size)


def _gaussian_blur_3d(x: torch.Tensor, kernel_size: int = 5, sigma: float = 1.5) -> torch.Tensor:
    """对3D tensor应用高斯模糊.

    Args:
        x: [B, C, D, H, W] tensor
        kernel_size: 高斯核大小 (奇数)
        sigma: 高斯标准差
    Returns:
        模糊后的tensor
    """
    device = x.device
    B, C = x.shape[:2]

    kernel = _create_gaussian_kernel_3d(kernel_size, sigma, device)
    pad = kernel_size // 2

    # 对每个通道分别卷积
    result = []
    for c in range(C):
        blurred = F.conv3d(x[:, c:c+1], kernel, padding=pad)
        result.append(blurred)

    return torch.cat(result, dim=1)


def _create_soft_roi_mask(
    label: torch.Tensor,
    num_channels: int,
    soft_edge: bool = True,
    dilate_iter: int = 2,
    gaussian_sigma: float = 2.0,
    gaussian_kernel_size: int = 7,
) -> torch.Tensor:
    """创建软边缘ROI mask.

    Args:
        label: 标签tensor [B, D, H, W] 或 [B, 1, D, H, W]
        num_channels: 输出通道数
        soft_edge: 是否使用软边缘 (高斯平滑)
        dilate_iter: 膨胀迭代次数 (先膨胀再平滑，扩大ROI范围)
        gaussian_sigma: 高斯平滑的标准差
        gaussian_kernel_size: 高斯核大小
    Returns:
        soft_mask: [B, num_channels, D, H, W] 软边缘mask, 值在[0, 1]
    """
    device = label.device

    if label.dim() == 5:
        label = label.squeeze(1)

    # 基础二值mask
    mask = (label > 0).float().unsqueeze(1)  # [B, 1, D, H, W]

    if soft_edge:
        # 可选: 先膨胀扩大ROI边界
        if dilate_iter > 0:
            # 使用max pooling模拟膨胀
            for _ in range(dilate_iter):
                mask = F.max_pool3d(mask, kernel_size=3, stride=1, padding=1)

        # 高斯平滑创建软边缘
        mask = _gaussian_blur_3d(mask, kernel_size=gaussian_kernel_size, sigma=gaussian_sigma)

        # 确保值在[0, 1]范围内
        mask = mask.clamp(0.0, 1.0)

    # 扩展到目标通道数
    mask = mask.expand(-1, num_channels, -1, -1, -1)

    return mask.to(device)


def _build_noise_unet(cfg: DictConfig, in_channels: int, spatial_dims: int = 3) -> nn.Module:
    """Build a small U-Net for noise generation."""
    channels = list(get_config(cfg, "channels", [16, 32, 64, 128]))
    strides = list(get_config(cfg, "strides", [2, 2, 2]))
    num_res_units = int(get_config(cfg, "num_res_units", 1))
    act = get_config(cfg, "act", "LEAKYRELU")
    norm = get_config(cfg, "norm", "INSTANCE")
    dropout = float(get_config(cfg, "dropout", 0.0))

    unet = MonaiUNet(
        spatial_dims=spatial_dims,
        in_channels=in_channels,
        out_channels=in_channels,
        channels=channels,
        strides=strides,
        num_res_units=num_res_units,
        act=act,
        norm=norm,
        dropout=dropout,
    )
    return unet


class NoiseUNetWrapper(nn.Module):
    """Wrapper for noise U-Net that applies tanh and scales output to [-eps, eps]."""
    def __init__(self, unet: nn.Module, epsilon: float = 8/255):
        super().__init__()
        self.unet = unet
        self.epsilon = epsilon

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raw_noise = self.unet(x)
        noise = torch.tanh(raw_noise) * self.epsilon
        return noise


class GradientComputer3D:
    """
    计算图像的层内(x-y)空间梯度，用于引导噪声强度分布。

    注意：只计算H和W方向的梯度，忽略D(z)方向。
    这样梯度引导只作用于层内，不干扰层间差异性损失。
    """

    def __init__(self):
        self._sobel_kernels: Optional[Tuple[torch.Tensor, ...]] = None

    def _build_sobel_kernels(self, device: torch.device) -> Tuple[torch.Tensor, ...]:
        """Build 3D Sobel kernels for H and W gradient computation only."""
        if self._sobel_kernels is not None:
            return tuple(k.to(device) for k in self._sobel_kernels)

        # 3D Sobel kernel for H direction (忽略D方向)
        # 在D方向上kernel size=1，只在H-W平面计算
        sobel_h = torch.tensor([
            [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
            [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        ], dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        # 3D Sobel kernel for W direction (忽略D方向)
        sobel_w = torch.tensor([
            [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
            [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        ], dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        self._sobel_kernels = (sobel_h, sobel_w)
        return tuple(k.to(device) for k in self._sobel_kernels)

    def compute_gradient_weight(self, image: torch.Tensor) -> torch.Tensor:
        """
        计算图像的层内(H-W)梯度幅值，归一化到[0, 1]作为噪声权重。

        只计算H和W方向的梯度，忽略D(z)方向，这样：
        - 梯度引导只作用于层内
        - 不干扰层间差异性损失

        Args:
            image: [B, C, D, H, W] float image tensor
        Returns:
            gradient_weight: [B, 1, D, H, W] normalized gradient magnitude in [0, 1]
        """
        device = image.device
        B, C = image.shape[:2]

        sobel_h, sobel_w = self._build_sobel_kernels(device)

        grad_mag_sum = torch.zeros(B, 1, *image.shape[2:], device=device)

        for c in range(C):
            img_c = image[:, c:c+1, ...]

            pad = 1
            # 只计算H和W方向的梯度，忽略D方向
            grad_h = F.conv3d(img_c, sobel_h, padding=pad)
            grad_w = F.conv3d(img_c, sobel_w, padding=pad)

            # 梯度幅值只包含H和W方向
            grad_mag_c = torch.sqrt(grad_h ** 2 + grad_w ** 2 + 1e-8)
            grad_mag_sum += grad_mag_c

        grad_mag = grad_mag_sum / C

        # Normalize to [0, 1] per sample
        for b in range(B):
            max_val = grad_mag[b].max()
            if max_val > 0:
                grad_mag[b] = grad_mag[b] / max_val

        return grad_mag


class InterSliceDiversityLoss(nn.Module):
    """
    层间差异性损失。

    鼓励相邻层之间的噪声差异最大化，利用医学图像层间低分辨率的特性。

    L_inter = (1/(D-1)HW) × Σ_d Σ_{h,w} [δ(d+1,h,w) - δ(d,h,w)]²

    此损失用负权重，实现最大化。
    """
    def __init__(
        self,
        roi_aware: bool = True,
        depth_dim: int = 2,  # [B, C, D, H, W]
    ):
        super().__init__()
        self.roi_aware = roi_aware
        self.depth_dim = depth_dim

    def forward(
        self,
        delta: torch.Tensor,
        label: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            delta: Noise tensor [B, C, D, H, W]
            label: Segmentation label [B, D, H, W] (for ROI mask)
        Returns:
            Scalar diversity loss (to be MAXIMIZED, use with negative weight)
        """
        d = self.depth_dim
        # 计算相邻层之间的差异
        delta_diff = torch.diff(delta, n=1, dim=d)  # [B, C, D-1, H, W]
        diff_sq = delta_diff ** 2

        # ROI感知: 只对前景区域求平均
        if self.roi_aware and label is not None:
            roi_mask = self._create_roi_mask(label, diff_sq.shape)

            # 只对ROI内像素求平均
            masked_sum = (diff_sq * roi_mask).sum()
            count = roi_mask.sum().clamp(min=1.0)

            return masked_sum / count
        else:
            return diff_sq.mean()

    def _create_roi_mask(self, label: torch.Tensor, target_shape: tuple) -> torch.Tensor:
        """Create ROI mask from label tensor."""
        if label.dim() == 5:
            label = label.squeeze(1)
        mask = (label > 0).float()
        mask = mask.unsqueeze(1)
        # 调整深度维度: D -> D-1, 使用minimum确保两层都在ROI内
        if mask.shape[2] > target_shape[2]:
            mask = torch.minimum(mask[:, :, 1:, :, :], mask[:, :, :-1, :, :])
        mask = mask.expand(-1, target_shape[1], -1, -1, -1)
        return mask.to(label.device)


@register_plugin("unet_noise_slice_grad")
class UNetNoiseSliceGradUE:
    """
    Inter-Slice Diversity + Gradient-Guided UNet Noise Generator.

    结合两种策略:
      1. 层间差异性约束: 最大化相邻层噪声差异
      2. 梯度引导: 根据图像梯度调制噪声强度

    损失函数:
      L_total = L_seg - λ_inter × L_inter

    噪声生成:
      δ = UNet(x) × gradient_weight × roi_mask

    Assumptions:
      - batch["image"]: FloatTensor [B, C, D, H, W]
      - batch["label"]: LongTensor  [B, D, H, W]
      - batch["key"]:   sample-wise key
    """

    def __init__(self):
        self._seg_loss: DiceCELoss | None = None
        self._noise_unet: NoiseUNetWrapper | None = None
        self._opt_noise_unet: torch.optim.Optimizer | None = None
        self._inter_diversity_loss: InterSliceDiversityLoss | None = None
        self._gradient_computer: GradientComputer3D | None = None
        self._noise_unet_device: torch.device | None = None
        self._initialized: bool = False
        self.logger = get_logger()

    @staticmethod
    def _norm_inplace(x: torch.Tensor, mean, std):
        """In-place per-channel normalize for ND volume."""
        for c, (m, s) in enumerate(zip(mean, std)):
            x[:, c].sub_(float(m)).div_(float(s))
        return x

    @staticmethod
    def _create_roi_mask(
        label: torch.Tensor,
        num_channels: int,
        soft_edge: bool = False,
        dilate_iter: int = 2,
        gaussian_sigma: float = 2.0,
        gaussian_kernel_size: int = 7,
    ) -> torch.Tensor:
        """Create ROI mask from label.

        Args:
            label: 标签tensor
            num_channels: 输出通道数
            soft_edge: 是否使用软边缘 (高斯平滑)
            dilate_iter: 膨胀迭代次数
            gaussian_sigma: 高斯标准差
            gaussian_kernel_size: 高斯核大小
        """
        return _create_soft_roi_mask(
            label=label,
            num_channels=num_channels,
            soft_edge=soft_edge,
            dilate_iter=dilate_iter,
            gaussian_sigma=gaussian_sigma,
            gaussian_kernel_size=gaussian_kernel_size,
        )

    def _get_seg_loss(self, trainer) -> DiceCELoss:
        """Build DiceCELoss consistent with SegTrainer configuration."""
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

    def _init_noise_unet(self, trainer, in_channels: int, spatial_dims: int = 3):
        """Initialize noise U-Net, optimizer, gradient computer, and inter-slice loss."""
        if self._initialized:
            return

        cfg = trainer.config
        device = trainer.device

        noise_unet_cfg = get_config(cfg, "ue.noise_unet", DictConfig({}))
        params = get_config(cfg, "ue.algorithm.params", DictConfig({}))
        eps = float(get_config(params, "epsilon", 8/255))

        # Build noise UNet
        base_unet = _build_noise_unet(noise_unet_cfg, in_channels, spatial_dims)
        self._noise_unet = NoiseUNetWrapper(base_unet, epsilon=eps)
        self._noise_unet = self._noise_unet.to(device)
        self._noise_unet_device = device

        # Build optimizer
        opt_cfg = get_config(noise_unet_cfg, "optimizer", DictConfig({}))
        lr = float(get_config(opt_cfg, "lr", 1e-4))
        weight_decay = float(get_config(opt_cfg, "weight_decay", 1e-5))
        betas = tuple(get_config(opt_cfg, "betas", (0.9, 0.999)))

        self._opt_noise_unet = torch.optim.Adam(
            self._noise_unet.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=betas,
        )

        # Build gradient computer
        self._gradient_computer = GradientComputer3D()

        # Build inter-slice diversity loss
        roi_aware = bool(get_config(params, "roi_aware", True))
        soft_edge = bool(get_config(params, "soft_edge", False))
        self._inter_diversity_loss = InterSliceDiversityLoss(
            roi_aware=roi_aware,
            depth_dim=2,
        )

        self._initialized = True
        self.logger.info(
            f"[UNetSliceGrad] Initialized: in_ch={in_channels}, spatial_dims={spatial_dims}, "
            f"eps={eps:.6f}, lr={lr}, roi_aware={roi_aware}, soft_edge={soft_edge}"
        )

    # ---------------- Surrogate-step: Update surrogate ---------------- #
    def surrogate_step_batch(self, trainer, batch) -> Dict[str, float]:
        """Update surrogate parameters only, do not update noise."""
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
        keys: Iterable[int] = batch["key"]

        B, C_in = x.shape[:2]
        spatial_dims = len(x.shape) - 2

        self._init_noise_unet(trainer, C_in, spatial_dims)

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

    # ---------------- N-step: Update noise with Inter-Slice + Gradient ---------------- #
    def noise_step_batch(self, trainer, batch) -> Dict[str, float]:
        """
        使用UNet生成噪声，结合层间差异性约束和梯度引导。

        损失: L_total = L_seg - λ_inter × L_inter
        噪声: δ = UNet(x) × gradient_weight × roi_mask
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
        spatial_dims = len(x.shape) - 2

        self._init_noise_unet(trainer, C_in, spatial_dims)

        algo = require_config(cfg, "ue.algorithm")
        params = require_config(algo, "params")
        eps = float(get_config(params, "epsilon", 8 / 255.0))
        num_steps = int(get_config(params, "noise_step", 1))
        lambda_inter = float(get_config(params, "lambda_inter", 0.5))
        roi_aware = bool(get_config(params, "roi_aware", True))
        use_gradient = bool(get_config(params, "use_gradient", True))

        # 软边缘ROI mask参数
        soft_edge = bool(get_config(params, "soft_edge", False))
        dilate_iter = int(get_config(params, "dilate_iter", 2))
        gaussian_sigma = float(get_config(params, "gaussian_sigma", 2.0))
        gaussian_kernel_size = int(get_config(params, "gaussian_kernel_size", 7))

        mean = tuple(get_config(cfg, "training.data.transforms.mean", (0.0,) * C_in))
        std = tuple(get_config(cfg, "training.data.transforms.std", (1.0,) * C_in))

        seg_loss_fn = self._get_seg_loss(trainer)

        # Freeze surrogate
        if not trainer.surrogates:
            raise RuntimeError("[UE] No surrogate bound.")
        _, s_model = next(iter(trainer.surrogates.items()))
        s_model.eval()
        for p in s_model.parameters():
            p.requires_grad = False

        # 计算图像梯度权重 [B, 1, D, H, W] -> [B, C, D, H, W]
        if use_gradient:
            gradient_weight = self._gradient_computer.compute_gradient_weight(x).to(device)
            gradient_weight = gradient_weight.expand(-1, C_in, -1, -1, -1)
        else:
            gradient_weight = torch.ones(B, C_in, *x.shape[2:], device=device)

        # 创建ROI掩码 [B, C, D, H, W] (支持软边缘)
        if roi_aware:
            roi_mask = self._create_roi_mask(
                y, C_in,
                soft_edge=soft_edge,
                dilate_iter=dilate_iter,
                gaussian_sigma=gaussian_sigma,
                gaussian_kernel_size=gaussian_kernel_size,
            ).to(device)
        else:
            roi_mask = torch.ones(B, C_in, *x.shape[2:], device=device)

        # 训练noise UNet
        self._noise_unet.train()
        last_seg_loss = torch.tensor(0.0, device=device)
        last_inter_loss = torch.tensor(0.0, device=device)

        for _ in range(max(1, num_steps)):
            # UNet生成基础噪声 [-eps, eps]
            delta_raw = self._noise_unet(x)

            # 计算层间差异性损失 (在应用mask之前)
            inter_loss = self._inter_diversity_loss(delta_raw, label=y if roi_aware else None)
            last_inter_loss = inter_loss.detach()

            # 噪声 = 基础噪声 × 梯度权重 × ROI掩码
            delta = delta_raw * gradient_weight * roi_mask

            perturb_img = (x + delta).clamp(0.0, 1.0)
            xn = perturb_img.clone()
            self._norm_inplace(xn, mean, std)

            out = s_model(xn)
            logits = out[0] if isinstance(out, (tuple, list)) else out

            seg_loss = seg_loss_fn(logits, y.unsqueeze(1))
            last_seg_loss = seg_loss.detach()

            # 总损失: L_seg - λ_inter × L_inter (最大化层间差异)
            total_loss = seg_loss - lambda_inter * inter_loss

            self._opt_noise_unet.zero_grad(set_to_none=True)
            total_loss.backward()
            self._opt_noise_unet.step()

        # 存储最终噪声
        self._noise_unet.eval()
        with torch.no_grad():
            final_delta_raw = self._noise_unet(x)
            final_delta = final_delta_raw * gradient_weight * roi_mask
            final_delta = final_delta.clamp(-eps, eps)

        nb.commit_batch(keys_list, final_delta.detach().cpu())

        delta_linf = float(final_delta.detach().abs().max().cpu())

        return {
            "noise_loss": float(last_seg_loss.cpu()),
            "inter_loss": float(last_inter_loss.cpu()),
            "delta_linf": delta_linf,
        }