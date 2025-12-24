# file: src/core/ue_algos/lsp.py
"""
LSP (Linear Separable Perturbation) - Training-Free Unlearnable Examples

基于 "Availability Attacks Create Shortcuts" 方法：
- 无需训练即可生成线性可分噪声模式
- 每个样本/类别获得唯一的线性可分噪声模式
- 对于医学图像分割 (MIS)：支持 ROI 模式（背景 vs 前景）

算法流程：
1. 读取标签数量 (num_labels)
2. 使用 sklearn.make_classification 生成线性可分特征向量
3. 将特征向量通过 patch 平铺转换为全分辨率噪声
4. L-无穷范数归一化到 [-epsilon, +epsilon]
5. ROI 模式：根据分割掩码混合背景和前景噪声
"""
from __future__ import annotations
from typing import Dict, Iterable, Tuple, Optional, Any, List, Hashable
import math
import numpy as np
import torch
from sklearn.datasets import make_classification
from typing import ClassVar

from ...registry import register_provider


# -------------------- key canonicalization -------------------- #
def _canon_key(k: Any) -> Hashable:
    """将 key 转换为可哈希类型，与 Learnable provider 一致"""
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
    """构建 key 到索引的映射，保持顺序并去重"""
    canon_keys = [_canon_key(k) for k in keys]
    uniq_list: List[Hashable] = list(dict.fromkeys(canon_keys))
    k2i = {k: i for i, k in enumerate(uniq_list)}
    return k2i, uniq_list


# -------------------- L-infinity normalization -------------------- #
def normalize_linf_(x: torch.Tensor, eps: float, tiny: float = 1e-12) -> torch.Tensor:
    """
    就地缩放张量 x 使其 L-无穷范数等于 eps

    确保噪声在 [-eps, +eps] 范围内，同时保持相对模式结构
    """
    amax = x.abs().amax()
    if (not torch.isfinite(amax)) or float(amax) <= tiny or eps <= 0:
        x.zero_()
        return x
    x.mul_(eps / float(amax))
    x.clamp_(-eps, +eps)
    return x


# -------------------- Feature vector to image/volume -------------------- #
def _reshape_repeat_crop_to_image(x_vec: np.ndarray, C: int, H: int, W: int, frame: int) -> torch.Tensor:
    """
    将 1D 特征向量重塑为 2D 图像 [C,H,W]，使用 patch 平铺

    LSP 核心：每个 patch (frame x frame) 共享相同的特征值，
    形成在特征空间中线性可分的块状模式
    """
    ph = math.ceil(H / frame)
    pw = math.ceil(W / frame)
    arr = x_vec.reshape(ph, pw, C)
    arr = np.repeat(np.repeat(arr, frame, axis=0), frame, axis=1)
    arr = arr[:H, :W, :]
    t = torch.from_numpy(arr.astype(np.float32)).permute(2, 0, 1).contiguous()
    return t


def _reshape_repeat_crop_to_volume(x_vec: np.ndarray, C: int, D: int, H: int, W: int, frame: int) -> torch.Tensor:
    """
    将 1D 特征向量重塑为 3D 体积 [C, D, H, W]，使用 patch 平铺

    3D 扩展的 LSP patch-repeat 策略，用于体积医学图像
    """
    pd = math.ceil(D / frame)
    ph = math.ceil(H / frame)
    pw = math.ceil(W / frame)
    arr = x_vec.reshape(pd, ph, pw, C)
    arr = np.repeat(np.repeat(np.repeat(arr, frame, axis=0), frame, axis=1), frame, axis=2)
    arr = arr[:D, :H, :W, :]
    t = torch.from_numpy(arr.astype(np.float32)).permute(3, 0, 1, 2).contiguous()
    return t


@register_provider("lsp")
class LSPProvider:
    """
    LSP (Linear Separable Perturbation) Provider - Training-Free 实现

    基于 "Availability Attacks Create Shortcuts" 方法：
    - 使用 sklearn.make_classification 生成线性可分噪声模式
    - 每个模式成为模型学习的"捷径"，而非真实特征
    - 对于 MIS 任务：使用 ROI 模式（背景 vs 前景噪声）

    工作流程：
    1. 在 __init__ 时，联合生成线性可分特征向量 X
    2. 将每个样本 key 映射到其分配的类别模式
    3. 通过 patch 平铺将 1D 特征向量转换为全分辨率噪声
    4. 应用 L-无穷归一化将噪声限制在 epsilon 范围内

    ROI 模式（医学图像分割）：
      当 roi_mode="binary" 时，为每个 key 生成两种噪声模式：
      - 模式 0: 背景噪声（应用于 label == 0 的像素）
      - 模式 1: 前景/ROI 噪声（应用于 label > 0 的像素）

    参数
    ----------
    keys : Iterable[Hashable]
        唯一样本标识符（如 BraTS 的 case_id）
    image_size : Tuple[int, ...]
        噪声形状：(C, H, W) 用于 2D 或 (C, D, H, W) 用于 3D
    epsilon : float
        L-无穷范数约束（如 8/255 = 0.0313725）
    noise_frame_size : int
        patch 大小。典型值：8 或 32
    seed : int
        随机种子
    class_sep : float
        make_classification 中的类别分离度（默认 10.0）
    roi_mode : Optional[str]
        None = 标准 LSP，"binary" = ROI 模式
    num_labels : int
        分割标签数量（BraTS: 4）
    **kwargs :
        忽略额外参数（如 tied_channels）以保持接口兼容性
    """
    REQUIRES_KEYS_AT_INIT: ClassVar[bool] = True

    def __init__(
        self,
        *,
        keys: Iterable[Hashable],
        image_size: Tuple[int, ...],
        epsilon: float,
        noise_frame_size: int = 32,
        seed: int = 0,
        class_sep: float = 10.0,
        oversample_ratio: float = 0.2,
        roi_mode: Optional[str] = None,
        num_labels: int = 4,
        **kwargs,  # 忽略额外参数（如 tied_channels）
    ):
        # 构建 key 索引
        self.key2idx, self.uniq_keys = _make_key_index(keys)

        # 解析图像维度（支持 2D 和 3D）
        if len(image_size) == 3:
            self.C_in, self.H, self.W = map(int, image_size)
            self.D = 1
            self.ndim = 2
        elif len(image_size) == 4:
            self.C_in, self.D, self.H, self.W = map(int, image_size)
            self.ndim = 3
        else:
            raise ValueError(f"[LSP] image_size 必须是 (C,H,W) 或 (C,D,H,W)，得到 {image_size}")

        self.eps: float = float(epsilon)
        self.frame: int = int(noise_frame_size)
        self.seed: int = int(seed)
        self.class_sep: float = float(class_sep)
        self.oversample_ratio: float = float(oversample_ratio)
        self.roi_mode: Optional[str] = None if roi_mode is None else str(roi_mode).lower()
        self.num_labels: int = int(num_labels)

        N = len(self.uniq_keys)
        if N == 0:
            raise ValueError("[LSP] 'keys' 为空")
        if self.C_in <= 0 or self.H <= 0 or self.W <= 0 or self.D <= 0:
            raise ValueError(f"[LSP] 无效的 image_size: {image_size}")
        if self.roi_mode is not None and self.roi_mode not in ("binary",):
            raise ValueError(f"[LSP] roi_mode 必须是 None 或 'binary'，得到 {self.roi_mode!r}")

        # 计算基于 patch 网格的特征维度
        if self.ndim == 3:
            pd = math.ceil(self.D / self.frame)
            ph = math.ceil(self.H / self.frame)
            pw = math.ceil(self.W / self.frame)
            self._n_features = pd * ph * pw * self.C_in
        else:
            ph = math.ceil(self.H / self.frame)
            pw = math.ceil(self.W / self.frame)
            self._n_features = ph * pw * self.C_in

        if self._n_features <= 0:
            raise RuntimeError("[LSP] n_features 计算为 0，检查 image_size / noise_frame_size")

        # 生成线性可分噪声模式
        if self.roi_mode == "binary":
            self._init_roi_mode(N)
        else:
            self._init_standard_mode(N)

    def _init_roi_mode(self, N: int):
        """
        初始化 ROI 模式：为每个样本生成 2 种噪声模式

        医学图像分割：
        - 类别 0: 背景噪声（应用于 mask == 0）
        - 类别 1: 前景/ROI 噪声（应用于 mask > 0）

        这创建了一个线性捷径，将背景与肿瘤区域分开
        """
        n_classes = 2
        X_per_class: Dict[int, List[np.ndarray]] = {0: [], 1: []}
        round_id = 0

        while len(X_per_class[0]) < N or len(X_per_class[1]) < N:
            need = max(N - len(X_per_class[0]), N - len(X_per_class[1]))
            n_draw = int(max(need, int(math.ceil(1.2 * N))))
            rs = self.seed + round_id

            X, y = make_classification(
                n_samples=n_draw,
                n_features=self._n_features,
                n_informative=self._n_features,
                n_redundant=0,
                n_repeated=0,
                n_classes=2,
                n_clusters_per_class=1,
                class_sep=self.class_sep,
                flip_y=0.0,
                random_state=rs,
            )

            for xi, yi in zip(X, y):
                yi_int = int(yi)
                if yi_int in (0, 1) and len(X_per_class[yi_int]) < N:
                    X_per_class[yi_int].append(xi.astype(np.float32, copy=False))

            round_id += 1
            if round_id > 64:
                raise RuntimeError("[LSP] 64 轮后仍无法生成足够的 ROI 样本")

        # 分配噪声表
        if self.ndim == 3:
            self._table_bg = torch.empty((N, self.C_in, self.D, self.H, self.W), dtype=torch.float32, device="cpu")
            self._table_roi = torch.empty((N, self.C_in, self.D, self.H, self.W), dtype=torch.float32, device="cpu")
        else:
            self._table_bg = torch.empty((N, self.C_in, self.H, self.W), dtype=torch.float32, device="cpu")
            self._table_roi = torch.empty((N, self.C_in, self.H, self.W), dtype=torch.float32, device="cpu")

        # 用平铺噪声模式填充表
        for key, i in self.key2idx.items():
            x_vec_bg = X_per_class[0][i]
            if self.ndim == 3:
                noise_bg = _reshape_repeat_crop_to_volume(x_vec_bg, self.C_in, self.D, self.H, self.W, self.frame)
            else:
                noise_bg = _reshape_repeat_crop_to_image(x_vec_bg, self.C_in, self.H, self.W, self.frame)
            normalize_linf_(noise_bg, self.eps)
            self._table_bg[i].copy_(noise_bg)

            x_vec_roi = X_per_class[1][i]
            if self.ndim == 3:
                noise_roi = _reshape_repeat_crop_to_volume(x_vec_roi, self.C_in, self.D, self.H, self.W, self.frame)
            else:
                noise_roi = _reshape_repeat_crop_to_image(x_vec_roi, self.C_in, self.H, self.W, self.frame)
            normalize_linf_(noise_roi, self.eps)
            self._table_roi[i].copy_(noise_roi)

        self._table = None

    def _init_standard_mode(self, N: int):
        """
        初始化标准 LSP：每个样本/类别一个唯一噪声模式

        每个样本从不同类别获得噪声模式，确保在特征空间中的线性可分性
        """
        X_per_class: Dict[int, np.ndarray] = {}
        round_id = 0

        while len(X_per_class) < N:
            need = N - len(X_per_class)
            base = max(need, int(math.ceil((1.0 + self.oversample_ratio) * N)))
            n_draw = int(base)
            rs = self.seed + round_id

            X, y = make_classification(
                n_samples=n_draw,
                n_features=self._n_features,
                n_informative=self._n_features,
                n_redundant=0,
                n_repeated=0,
                n_classes=N,
                n_clusters_per_class=1,
                class_sep=self.class_sep,
                flip_y=0.0,
                random_state=rs,
            )

            for xi, yi in zip(X, y):
                if yi not in X_per_class:
                    X_per_class[int(yi)] = xi.astype(np.float32, copy=False)
                    if len(X_per_class) == N:
                        break

            round_id += 1
            if round_id > 64:
                raise RuntimeError("[LSP] 64 轮后仍无法实现每类覆盖")

        if self.ndim == 3:
            self._table = torch.empty((N, self.C_in, self.D, self.H, self.W), dtype=torch.float32, device="cpu")
        else:
            self._table = torch.empty((N, self.C_in, self.H, self.W), dtype=torch.float32, device="cpu")

        for key, i in self.key2idx.items():
            x_vec = X_per_class[i]
            if self.ndim == 3:
                noise = _reshape_repeat_crop_to_volume(x_vec, self.C_in, self.D, self.H, self.W, self.frame)
            else:
                noise = _reshape_repeat_crop_to_image(x_vec, self.C_in, self.H, self.W, self.frame)
            normalize_linf_(noise, self.eps)
            self._table[i].copy_(noise)

        self._table_bg = None
        self._table_roi = None

    # ------------------- 公共 API ------------------- #
    @torch.no_grad()
    def get_noise(self, key_raw: Hashable, perturb_type: Optional[str] = None) -> torch.Tensor:
        """
        返回给定 key 的存储噪声

        在 ROI 模式下，返回背景噪声（用于兼容）

        参数:
            key_raw: 样本标识符（如 case_id）
            perturb_type: 忽略（保持 API 兼容）

        返回:
            torch.Tensor: 噪声体积 [C,H,W] 或 [C,D,H,W]，CPU float32
        """
        k = _canon_key(key_raw)
        if k not in self.key2idx:
            raise KeyError(f"[LSP] 未知 key: {repr(key_raw)}")
        i = self.key2idx[k]

        if self.roi_mode == "binary":
            return self._table_bg[i].clone().clamp_(-self.eps, +self.eps)
        else:
            return self._table[i].clone().clamp_(-self.eps, +self.eps)

    @torch.no_grad()
    def get_noise_with_mask(self, key_raw: Hashable, mask: torch.Tensor) -> torch.Tensor:
        """
        ROI 模式：根据分割掩码返回混合噪声

        实现 MIS 任务的 ROI 模式 LSP 适配：
        - 背景像素 (mask == 0) 接收背景噪声模式
        - 前景/ROI 像素 (mask > 0) 接收 ROI 噪声模式

        参数:
            key_raw: 样本标识符
            mask: 分割掩码 [H,W] 或 [D,H,W]，值在 {0, 1, ..., num_labels-1}

        返回:
            torch.Tensor: 混合噪声 [C,H,W] 或 [C,D,H,W]
        """
        if self.roi_mode != "binary":
            raise RuntimeError(
                f"[LSP] get_noise_with_mask() 需要 roi_mode='binary'。"
                f"当前 roi_mode={self.roi_mode!r}。标准模式请使用 get_noise()。"
            )

        k = _canon_key(key_raw)
        if k not in self.key2idx:
            raise KeyError(f"[LSP] 未知 key: {repr(key_raw)}")
        i = self.key2idx[k]

        noise_bg = self._table_bg[i].clone()
        noise_roi = self._table_roi[i].clone()

        roi_mask = (mask > 0).float()

        if self.ndim == 3:
            roi_mask = roi_mask.unsqueeze(0)
        else:
            roi_mask = roi_mask.unsqueeze(0)

        noise = (1 - roi_mask) * noise_bg + roi_mask * noise_roi

        return noise.clamp_(-self.eps, +self.eps)

    # ------------------- 批量 API（兼容性）------------------- #
    @torch.no_grad()
    def batch_noise(self, keys: List[Hashable]) -> torch.Tensor:
        """
        获取一批 key 的噪声

        提供与 Learnable provider 接口的兼容性
        """
        tensors = [self.get_noise(k) for k in keys]
        return torch.stack(tensors, dim=0)