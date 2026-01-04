# file: src/core/ue_algos/unet_noise_grad.py
"""
Gradient-Guided UNet Noise Generator for 3D Medical Image Segmentation.

核心思想：
  - 在ROI区域内，根据**图像**的梯度来引导噪声强度
  - 图像梯度大的区域（边缘、纹理变化）噪声强
  - 图像梯度小的区域（平滑区域）噪声弱
  - 噪声连续分布，最大幅度为 8/255

数据流（避免噪声被梯度权重压缩）：
  1. raw = UNet(x)                         # 原始输出，无约束
  2. weighted = raw * grad_weight * roi_mask  # 梯度加权
  3. delta = tanh(weighted) * ε            # 最后才约束到[-ε, ε]

Training loop:
  1. Surrogate-step: 与MinMin相同，在加噪图像上训练代理模型
  2. Noise-step:
     - UNet生成原始输出（不加tanh）
     - 计算图像的3D梯度作为权重
     - 梯度加权后才做tanh约束
     - 反向传播更新UNet
"""
from __future__ import annotations
from typing import Dict, Iterable, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
from monai.losses import DiceCELoss
from monai.networks.nets import UNet as MonaiUNet

from ...registry import register_plugin
from ...utils.config import get_config, require_config
from ...utils.logger import get_logger


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
    """
    Wrapper for noise U-Net.

    注意：这里不应用tanh约束，输出原始值。
    tanh约束在梯度加权之后进行，避免噪声被压缩。
    """
    def __init__(self, unet: nn.Module, epsilon: float = 8/255):
        super().__init__()
        self.unet = unet
        self.epsilon = epsilon

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 输出原始值，不做tanh约束
        # tanh在noise_step_batch中梯度加权之后进行
        raw_noise = self.unet(x)
        return raw_noise


class GradientComputer3D:
    """计算图像的3D空间梯度，用于引导噪声强度分布。"""

    def __init__(self):
        self._sobel_kernels: Optional[Tuple[torch.Tensor, ...]] = None

    def _build_sobel_kernels(self, device: torch.device) -> Tuple[torch.Tensor, ...]:
        """Build 3D Sobel kernels for gradient computation."""
        if self._sobel_kernels is not None:
            return tuple(k.to(device) for k in self._sobel_kernels)

        # 3D Sobel kernel for D direction
        sobel_d = torch.tensor([
            [[-1, -2, -1], [-2, -4, -2], [-1, -2, -1]],
            [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            [[1, 2, 1], [2, 4, 2], [1, 2, 1]]
        ], dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        # 3D Sobel kernel for H direction
        sobel_h = torch.tensor([
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
            [[-2, -4, -2], [0, 0, 0], [2, 4, 2]],
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
        ], dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        # 3D Sobel kernel for W direction
        sobel_w = torch.tensor([
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
            [[-2, 0, 2], [-4, 0, 4], [-2, 0, 2]],
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
        ], dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        self._sobel_kernels = (sobel_d, sobel_h, sobel_w)
        return tuple(k.to(device) for k in self._sobel_kernels)

    def compute_gradient_weight(self, image: torch.Tensor) -> torch.Tensor:
        """
        计算图像的3D梯度幅值，归一化到[0, 1]作为噪声权重。

        Args:
            image: [B, C, D, H, W] float image tensor
        Returns:
            gradient_weight: [B, 1, D, H, W] normalized gradient magnitude in [0, 1]
        """
        device = image.device
        B, C = image.shape[:2]

        # Get Sobel kernels
        sobel_d, sobel_h, sobel_w = self._build_sobel_kernels(device)

        # 对每个通道计算梯度，然后取平均
        grad_mag_sum = torch.zeros(B, 1, *image.shape[2:], device=device)

        for c in range(C):
            img_c = image[:, c:c+1, ...]  # [B, 1, D, H, W]

            pad = 1
            grad_d = F.conv3d(img_c, sobel_d, padding=pad)
            grad_h = F.conv3d(img_c, sobel_h, padding=pad)
            grad_w = F.conv3d(img_c, sobel_w, padding=pad)

            # 计算该通道的梯度幅值
            grad_mag_c = torch.sqrt(grad_d ** 2 + grad_h ** 2 + grad_w ** 2 + 1e-8)
            grad_mag_sum += grad_mag_c

        # 取平均
        grad_mag = grad_mag_sum / C

        # Normalize to [0, 1] per sample
        for b in range(B):
            max_val = grad_mag[b].max()
            if max_val > 0:
                grad_mag[b] = grad_mag[b] / max_val

        return grad_mag


@register_plugin("unet_grad_noise")
class UNetGradNoiseUE:
    """
    Gradient-Guided UNet Noise Generator for 3D Medical Image Segmentation.

    在ROI区域内，根据图像梯度引导噪声强度：
      - 图像梯度大（边缘、纹理变化）→ 噪声强
      - 图像梯度小（平滑区域）→ 噪声弱
      - 噪声连续分布，最大 8/255
    """

    def __init__(self):
        self._seg_loss: DiceCELoss | None = None
        self._noise_unet: NoiseUNetWrapper | None = None
        self._opt_noise_unet: torch.optim.Optimizer | None = None
        self._noise_unet_device: torch.device | None = None
        self._gradient_computer: GradientComputer3D | None = None
        self._initialized: bool = False
        self.logger = get_logger()

    @staticmethod
    def _norm_inplace(x: torch.Tensor, mean, std):
        """In-place per-channel normalize for ND volume."""
        for c, (m, s) in enumerate(zip(mean, std)):
            x[:, c].sub_(float(m)).div_(float(s))
        return x

    @staticmethod
    def _create_roi_mask(label: torch.Tensor, num_channels: int) -> torch.Tensor:
        """Create ROI mask from label. ROI = label > 0."""
        if label.dim() == 5:
            label = label.squeeze(1)
        mask = (label > 0).float().unsqueeze(1)  # [B, 1, D, H, W]
        mask = mask.expand(-1, num_channels, -1, -1, -1)  # [B, C, D, H, W]
        return mask

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
        """Initialize the noise U-Net, optimizer, and gradient computer."""
        if self._initialized:
            return

        cfg = trainer.config
        device = trainer.device

        noise_unet_cfg = get_config(cfg, "ue.noise_unet", DictConfig({}))
        eps = float(get_config(cfg, "ue.algorithm.params.epsilon", 8/255))

        base_unet = _build_noise_unet(noise_unet_cfg, in_channels, spatial_dims)
        self._noise_unet = NoiseUNetWrapper(base_unet, epsilon=eps)
        self._noise_unet = self._noise_unet.to(device)
        self._noise_unet_device = device

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

        self._gradient_computer = GradientComputer3D()

        self._initialized = True
        self.logger.info(
            f"[UNetGradNoise] Initialized: in_channels={in_channels}, "
            f"spatial_dims={spatial_dims}, eps={eps:.6f}, lr={lr}"
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

    # ---------------- N-step: Update noise via Gradient-Guided UNet ---------------- #
    def noise_step_batch(self, trainer, batch) -> Dict[str, float]:
        """
        使用UNet生成噪声，根据图像梯度加权。

        数据流（避免噪声被压缩）：
          1. raw = UNet(x)                    # 原始输出，无约束
          2. weighted = raw * grad_weight * roi_mask  # 梯度加权
          3. delta = tanh(weighted) * ε       # 最后才约束到[-ε, ε]
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

        mean = tuple(get_config(cfg, "training.data.transforms.mean", (0.0,) * C_in))
        std = tuple(get_config(cfg, "training.data.transforms.std", (1.0,) * C_in))

        seg_loss_fn = self._get_seg_loss(trainer)

        # freeze surrogate
        if not trainer.surrogates:
            raise RuntimeError("[UE] No surrogate bound.")
        _, s_model = next(iter(trainer.surrogates.items()))
        s_model.eval()
        for p in s_model.parameters():
            p.requires_grad = False

        # 计算图像梯度权重 [B, 1, D, H, W]，归一化到[0, 1]
        gradient_weight = self._gradient_computer.compute_gradient_weight(x).to(device)
        # 扩展到所有通道 [B, C, D, H, W]
        gradient_weight = gradient_weight.expand(-1, C_in, -1, -1, -1)

        # 创建ROI掩码 [B, C, D, H, W]
        roi_mask = self._create_roi_mask(y, C_in).to(device)

        # 训练noise UNet
        self._noise_unet.train()
        last_loss = torch.tensor(0.0, device=device)

        for _ in range(max(1, num_steps)):
            # 1. UNet生成原始输出（无约束）
            raw = self._noise_unet(x)

            # 2. 梯度加权 + ROI掩码
            weighted = raw * gradient_weight * roi_mask

            # 3. 最后才做tanh约束到[-ε, ε]，避免噪声被压缩
            delta = torch.tanh(weighted) * eps

            perturb_img = (x + delta).clamp(0.0, 1.0)
            xn = perturb_img.clone()
            self._norm_inplace(xn, mean, std)

            out = s_model(xn)
            logits = out[0] if isinstance(out, (tuple, list)) else out

            loss = seg_loss_fn(logits, y.unsqueeze(1))
            last_loss = loss.detach()

            self._opt_noise_unet.zero_grad(set_to_none=True)
            loss.backward()
            self._opt_noise_unet.step()

        # 存储最终噪声
        self._noise_unet.eval()
        with torch.no_grad():
            raw = self._noise_unet(x)
            weighted = raw * gradient_weight * roi_mask
            final_delta = torch.tanh(weighted) * eps
            final_delta = final_delta.clamp(-eps, eps)  # 额外保险

        nb.commit_batch(keys_list, final_delta.detach().cpu())

        delta_linf = float(final_delta.detach().abs().max().cpu())

        return {
            "noise_loss": float(last_loss.cpu()),
            "delta_linf": delta_linf,
        }