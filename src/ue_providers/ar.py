# file: src/ue_providers/ar.py
"""
Autoregressive Perturbations Provider (Training-Free)

基于论文: "Autoregressive Perturbations for Data Poisoning" (Sandoval-Segura et al., NeurIPS 2022)
源代码: https://github.com/psandovalsegura/autoregressive-poisoning

重要说明:
    原论文的AR方法是**纯粹的Training-Free方法**，不涉及任何PGD或梯度优化！
    噪声通过自回归卷积过程一次性生成，然后直接应用到图像上。

ROI感知模式:
    基于 "Safeguarding Medical Image Segmentation Datasets against Unauthorized
    Training via Contour- and Texture-Aware Perturbations" 论文:
    "To adapt LSP and AR for MIS tasks, we treat pixels inside and outside
    the ROI as two distinct classes."
"""
from __future__ import annotations
from typing import ClassVar, Dict, Hashable, Iterable, List, Optional, Tuple, Any

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from ..registry import register_provider


# ============================================================================
# AR系数模式 (来自原论文)
# ============================================================================

# 几何序列系数
GEO_A1_R12 = torch.tensor([
    [1/2, 1/4, 1/8],
    [1/16, 1/32, 1/64],
    [1/128, 1/256, 0.0]
], dtype=torch.float32)

GEO_A2_R13 = torch.tensor([
    [2/3, 2/9, 2/27],
    [2/81, 2/243, 2/729],
    [2/2187, 2/6561, 0.0]
], dtype=torch.float32)

# 斐波那契序列系数
FIBONACCI = torch.tensor([
    [1/1, 1/1, 1/2],
    [1/3, 1/5, 1/8],
    [1/13, 1/21, 0.0]
], dtype=torch.float32)

# 均匀权重
UNIFORM = torch.tensor([
    [1.0, 1.0, 1.0],
    [1.0, 1.0, 1.0],
    [1.0, 1.0, 0.0]
], dtype=torch.float32)

# 边缘聚焦
EDGE_FOCUS = torch.tensor([
    [0.5, 1.0, 0.5],
    [1.0, 2.0, 1.0],
    [0.5, 1.0, 0.0]
], dtype=torch.float32)

AR_COEFFICIENTS = {
    "geo_a1_r12": GEO_A1_R12,
    "geo_a2_r13": GEO_A2_R13,
    "fibonacci": FIBONACCI,
    "uniform": UNIFORM,
    "edge_focus": EDGE_FOCUS,
}


def _canon_key(k: Any) -> Hashable:
    """规范化key为可哈希类型"""
    if torch.is_tensor(k):
        if k.ndim == 0:
            return k.item()
        return tuple(np.asarray(k.cpu()).reshape(-1).tolist())
    if isinstance(k, (np.integer,)):
        return int(k.item())
    if isinstance(k, (np.floating,)):
        return float(k.item())
    return k


def _make_key_index(keys: Iterable[Hashable]) -> Tuple[Dict[Hashable, int], List[Hashable]]:
    """创建key到索引的映射"""
    canon_keys = [_canon_key(k) for k in keys]
    uniq_list: List[Hashable] = list(dict.fromkeys(canon_keys))
    k2i = {k: i for i, k in enumerate(uniq_list)}
    return k2i, uniq_list


def normalize_linf_(x: torch.Tensor, eps: float, tiny: float = 1e-12) -> torch.Tensor:
    """就地将张量的L∞范数缩放到eps"""
    amax = x.abs().amax()
    if (not torch.isfinite(amax)) or float(amax) <= tiny or eps <= 0:
        x.zero_()
        return x
    x.mul_(eps / float(amax))
    x.clamp_(-eps, +eps)
    return x


class ARNoiseGenerator:
    """
    自回归噪声生成器 (Training-Free) - 优化版本

    严格按照原论文实现：
    1. 初始化随机高斯噪声
    2. 通过AR系数卷积迭代更新
    3. 归一化到epsilon范围

    不涉及任何梯度优化或PGD！
    """

    def __init__(
        self,
        coeffs: str = "geo_a1_r12",
        epsilon: float = 8/255,
        p_norm: float = float('inf'),
        spatial_dims: int = 3,
        seed: Optional[int] = None,
    ):
        self.epsilon = epsilon
        self.p_norm = p_norm
        self.spatial_dims = spatial_dims
        self.seed = seed

        # 加载AR系数
        if coeffs not in AR_COEFFICIENTS:
            raise ValueError(f"未知的AR系数: {coeffs}. 可用: {list(AR_COEFFICIENTS.keys())}")

        # 归一化系数 (右下角为0，其余归一化使和为1)
        raw_coeffs = AR_COEFFICIENTS[coeffs].clone()
        raw_coeffs[-1, -1] = 0.0
        total = raw_coeffs.sum()
        if total > 0:
            raw_coeffs = raw_coeffs / total
        self.coeffs = raw_coeffs

    def generate(
        self,
        shape: Tuple[int, ...],
        device: torch.device = None,
    ) -> torch.Tensor:
        """
        生成AR扰动 (优化版本 - 批量处理)

        Args:
            shape: 输出形状 [C, H, W] 或 [C, D, H, W]
            device: 目标设备

        Returns:
            扰动张量，范围在[-epsilon, epsilon]
        """
        if device is None:
            device = torch.device('cpu')

        # 设置随机种子
        if self.seed is not None:
            torch.manual_seed(self.seed)

        # 解析形状
        if len(shape) == 3:  # 2D: [C, H, W]
            C, H, W = shape
            D = 1
            is_3d = False
        elif len(shape) == 4:  # 3D: [C, D, H, W]
            C, D, H, W = shape
            is_3d = True
        else:
            raise ValueError(f"不支持的形状: {shape}")

        # AR系数转换为卷积核 [out_ch, in_ch, kH, kW]
        b = self.coeffs.to(device)
        kernel = b.unsqueeze(0).unsqueeze(0)  # [1, 1, 3, 3]

        if is_3d:
            # 3D情况：批量生成所有深度切片
            # 初始化随机信号 [C*D, 1, H, W] 以便批量卷积
            signal = torch.randn(C * D, 1, H, W, device=device)

            # 批量AR迭代
            for _ in range(3):
                signal = F.conv2d(signal, kernel, padding=1)
                signal = signal + 0.1 * torch.randn_like(signal)

            # 重塑为 [C, D, H, W]
            noise = signal.view(C, D, H, W)

            # 深度方向平滑 (可选，对于连续性)
            if D > 2:
                kernel_d = torch.tensor([0.25, 0.5, 0.25], device=device).view(1, 1, 3, 1, 1)
                noise = noise.unsqueeze(0)  # [1, C, D, H, W]
                noise = F.pad(noise, (0, 0, 0, 0, 1, 1), mode='replicate')
                # 对所有通道一起处理
                noise = noise.view(1, C, D + 2, H, W)
                smoothed = []
                for c in range(C):
                    ch = noise[:, c:c+1]  # [1, 1, D+2, H, W]
                    ch = F.conv3d(ch, kernel_d, padding=0)
                    smoothed.append(ch)
                noise = torch.cat(smoothed, dim=1).squeeze(0)  # [C, D, H, W]

        else:
            # 2D情况：批量处理所有通道
            signal = torch.randn(C, 1, H, W, device=device)

            for _ in range(3):
                signal = F.conv2d(signal, kernel, padding=1)
                signal = signal + 0.1 * torch.randn_like(signal)

            noise = signal.squeeze(1)  # [C, H, W]

        # 归一化到epsilon范围
        noise = self._normalize(noise)

        return noise

    def _normalize(self, delta: torch.Tensor) -> torch.Tensor:
        """归一化扰动到epsilon范围"""
        if self.p_norm == float('inf'):
            max_val = delta.abs().max()
            if max_val > 1e-8:
                delta = delta / max_val * self.epsilon
        elif self.p_norm == 2:
            norm = delta.norm(p=2)
            if norm > 1e-8:
                delta = delta / norm * self.epsilon

        return delta.clamp(-self.epsilon, self.epsilon)


@register_provider("ar")
class ARProvider:
    """
    AR噪声提供器 (Training-Free)

    这是原论文的正确实现方式：
    - 不需要训练代理模型
    - 不需要PGD优化
    - 噪声在初始化时一次性生成

    用法:
        provider = ARProvider(keys=all_keys, image_size=(4,128,128,128), epsilon=8/255)
        noise = provider.get_noise(key)

    ROI模式:
        provider = ARProvider(..., roi_mode="binary")
        noise = provider.get_noise_with_mask(key, segmentation_mask)
    """
    REQUIRES_KEYS_AT_INIT: ClassVar[bool] = True

    def __init__(
        self,
        *,
        keys: Iterable[Hashable],
        image_size: Tuple[int, ...],
        epsilon: float = 8/255,
        ar_coeffs: str = "geo_a1_r12",
        ar_coeffs_roi: str = "fibonacci",
        seed: int = 0,
        roi_mode: Optional[str] = None,
        **kwargs,
    ):
        """
        Args:
            keys: 所有样本的唯一标识符
            image_size: 噪声形状 (C,H,W) 或 (C,D,H,W)
            epsilon: 扰动幅度上限
            ar_coeffs: 背景AR系数模式
            ar_coeffs_roi: ROI区域AR系数模式 (roi_mode="binary"时使用)
            seed: 随机种子
            roi_mode: None (标准模式) 或 "binary" (ROI感知模式)
        """
        self.key2idx, self.uniq_keys = _make_key_index(keys)

        # 解析形状
        if len(image_size) == 3:
            self.C_in, self.H, self.W = map(int, image_size)
            self.D = 1
            self.ndim = 2
        elif len(image_size) == 4:
            self.C_in, self.D, self.H, self.W = map(int, image_size)
            self.ndim = 3
        else:
            raise ValueError(f"image_size必须是(C,H,W)或(C,D,H,W), 得到 {image_size}")

        self.eps = float(epsilon)
        self.seed = int(seed)
        self.roi_mode = None if roi_mode is None else str(roi_mode).lower()

        if self.roi_mode is not None and self.roi_mode not in ("binary",):
            raise ValueError(f"roi_mode必须是None或'binary', 得到 {self.roi_mode!r}")

        N = len(self.uniq_keys)
        if N == 0:
            raise ValueError("keys不能为空")

        # 选择设备 - 优先使用GPU
        if torch.cuda.is_available():
            device = torch.device('cuda')
            print(f"[AR] 使用GPU加速噪声生成")
        else:
            device = torch.device('cpu')
            print(f"[AR] 使用CPU生成噪声 (可能较慢)")

        # 创建AR生成器
        gen_bg = ARNoiseGenerator(
            coeffs=ar_coeffs,
            epsilon=self.eps,
            spatial_dims=3 if self.ndim == 3 else 2,
            seed=self.seed,
        )

        # 生成噪声表
        if self.ndim == 3:
            shape = (self.C_in, self.D, self.H, self.W)
            self._table_bg = torch.empty((N, self.C_in, self.D, self.H, self.W), dtype=torch.float32)
        else:
            shape = (self.C_in, self.H, self.W)
            self._table_bg = torch.empty((N, self.C_in, self.H, self.W), dtype=torch.float32)

        # 为每个样本生成背景噪声 (带进度条)
        print(f"[AR] 生成背景噪声: {N} 样本, 形状={shape}, 系数={ar_coeffs}")
        for i in tqdm(range(N), desc="[AR] 生成背景噪声", unit="样本"):
            gen_bg.seed = self.seed + i
            noise = gen_bg.generate(shape, device=device)
            normalize_linf_(noise, self.eps)
            self._table_bg[i].copy_(noise.cpu())

        # ROI模式：生成第二套噪声
        if self.roi_mode == "binary":
            gen_roi = ARNoiseGenerator(
                coeffs=ar_coeffs_roi,
                epsilon=self.eps,
                spatial_dims=3 if self.ndim == 3 else 2,
                seed=self.seed + 100000,
            )

            if self.ndim == 3:
                self._table_roi = torch.empty((N, self.C_in, self.D, self.H, self.W), dtype=torch.float32)
            else:
                self._table_roi = torch.empty((N, self.C_in, self.H, self.W), dtype=torch.float32)

            print(f"[AR] 生成ROI噪声: {N} 样本, 系数={ar_coeffs_roi}")
            for i in tqdm(range(N), desc="[AR] 生成ROI噪声", unit="样本"):
                gen_roi.seed = self.seed + 100000 + i
                noise = gen_roi.generate(shape, device=device)
                normalize_linf_(noise, self.eps)
                self._table_roi[i].copy_(noise.cpu())
        else:
            self._table_roi = None

        print(f"[AR] 噪声生成完成!")

    @torch.no_grad()
    def get_noise(self, key_raw: Hashable, perturb_type: Optional[str] = None) -> torch.Tensor:
        """
        获取指定key的噪声

        在ROI模式下返回背景噪声。要获取ROI混合噪声，请使用get_noise_with_mask()。
        """
        k = _canon_key(key_raw)
        if k not in self.key2idx:
            raise KeyError(f"未知的key: {repr(key_raw)}")
        i = self.key2idx[k]
        return self._table_bg[i].clone().clamp_(-self.eps, +self.eps)

    @torch.no_grad()
    def get_noise_with_mask(self, key_raw: Hashable, mask: torch.Tensor) -> torch.Tensor:
        """
        ROI模式：根据分割掩膜返回混合噪声

        实现论文思想：
        "To adapt LSP and AR for MIS tasks, we treat pixels inside and outside
        the ROI as two distinct classes."

        Args:
            key_raw: 样本标识符
            mask: 分割掩膜 [D,H,W] 或 [H,W]，值为{0,1,...,num_labels-1}

        Returns:
            混合噪声：背景区域(mask==0)使用ar_coeffs噪声，
                     ROI区域(mask>0)使用ar_coeffs_roi噪声
        """
        if self.roi_mode != "binary":
            raise RuntimeError(
                f"get_noise_with_mask()需要roi_mode='binary'。当前roi_mode={self.roi_mode!r}"
            )

        k = _canon_key(key_raw)
        if k not in self.key2idx:
            raise KeyError(f"未知的key: {repr(key_raw)}")
        i = self.key2idx[k]

        noise_bg = self._table_bg[i].clone()
        noise_roi = self._table_roi[i].clone()

        # 创建ROI掩膜
        roi_mask = (mask > 0).float()

        # 扩展维度以匹配噪声形状
        if self.ndim == 3:
            roi_mask = roi_mask.unsqueeze(0)  # [1, D, H, W]
        else:
            roi_mask = roi_mask.unsqueeze(0)  # [1, H, W]

        # 混合：背景区域用noise_bg，ROI区域用noise_roi
        noise = (1 - roi_mask) * noise_bg + roi_mask * noise_roi

        return noise.clamp_(-self.eps, +self.eps)