# file: src/ue_providers/lsp_makecls.py
from __future__ import annotations
from typing import Dict, Iterable, Tuple, Optional, Any, List, Hashable
import math
import numpy as np
import torch
from sklearn.datasets import make_classification
from typing import ClassVar

from ..registry import register_provider


# -------------------- key canonicalization (same spirit as Learnable) -------------------- #
def _canon_key(k: Any) -> Hashable:
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
    canon_keys = [_canon_key(k) for k in keys]
    uniq_list: List[Hashable] = list(dict.fromkeys(canon_keys))  # stable de-dup, preserves order
    k2i = {k: i for i, k in enumerate(uniq_list)}
    return k2i, uniq_list


# -------------------- strength control: L2 then per-pixel clamp (faithful to your script) -------------------- #
def normalize_linf_(x: torch.Tensor, eps: float, tiny: float = 1e-12) -> torch.Tensor:
    """
    就地把张量 x 的 L∞ 范数缩放到 eps：max(|x|) -> eps
    保持 dtype/device，不改变形状分布，只做全局等比缩放。
    """
    amax = x.abs().amax()
    if (not torch.isfinite(amax)) or float(amax) <= tiny or eps <= 0:
        x.zero_()
        return x
    x.mul_(eps / float(amax))
    # 可选：再夹一下，防极少数数值毛刺
    x.clamp_(-eps, +eps)
    return x



def _compute_linf_coefficient(epsilon: float, frame: int, C_in: int) -> float:
    """
    Faithfully replicate your constants:
    - grayscale-like (C=1):  frame=8 -> sqrt(16/224)*eps,  frame=32 -> sqrt(94/224)*eps
    - RGB-like      (C=3):  frame=8 -> sqrt(11/224)*eps,  frame=32 -> sqrt(32/224)*eps
    Keep the denominator 224 to mirror original behavior, regardless of actual H,W.
    """
    if frame not in (8, 32):
        raise ValueError(f"[LSP-MC] Only noise_frame_size in {{8, 32}} is supported for faithful reproduction, got {frame}.")
    if C_in == 3:
        num = 11.0 if frame == 8 else 32.0
    else:
        num = 16.0 if frame == 8 else 94.0
    return math.sqrt(num / 224.0) * float(epsilon)


def _reshape_repeat_crop_to_image(x_vec: np.ndarray, C: int, H: int, W: int, frame: int) -> torch.Tensor:
    """
    x_vec: shape [n_features] = [ceil(H/frame)*ceil(W/frame)*C]
    Returns torch.float32 tensor [C,H,W] on CPU.
    """
    # Determine patch grid
    ph = math.ceil(H / frame)
    pw = math.ceil(W / frame)
    # Reshape into [ph, pw, C]
    arr = x_vec.reshape(ph, pw, C)  # numpy
    # Repeat patches spatially by 'frame'
    arr = np.repeat(np.repeat(arr, frame, axis=0), frame, axis=1)  # [ph*frame, pw*frame, C]
    # Crop to exactly [H, W, C]
    arr = arr[:H, :W, :]
    # To torch [C,H,W]
    t = torch.from_numpy(arr.astype(np.float32)).permute(2, 0, 1).contiguous()
    return t


def _reshape_repeat_crop_to_volume(x_vec: np.ndarray, C: int, D: int, H: int, W: int, frame: int) -> torch.Tensor:
    """
    3D version: Reshape 1D feature vector to 3D volume [C, D, H, W].
    x_vec: shape [n_features] = [ceil(D/frame)*ceil(H/frame)*ceil(W/frame)*C]
    Returns torch.float32 tensor [C, D, H, W] on CPU.
    """
    # Determine patch grid
    pd = math.ceil(D / frame)
    ph = math.ceil(H / frame)
    pw = math.ceil(W / frame)
    # Reshape into [pd, ph, pw, C]
    arr = x_vec.reshape(pd, ph, pw, C)  # numpy
    # Repeat patches spatially by 'frame'
    arr = np.repeat(np.repeat(np.repeat(arr, frame, axis=0), frame, axis=1), frame, axis=2)
    # [pd*frame, ph*frame, pw*frame, C]
    # Crop to exactly [D, H, W, C]
    arr = arr[:D, :H, :W, :]
    # To torch [C, D, H, W]
    t = torch.from_numpy(arr.astype(np.float32)).permute(3, 0, 1, 2).contiguous()
    return t


@register_provider("lsp")
class LSPProvider:
    """
    Linear-Separable noise via make_classification (joint generation at init).

    Intent:
      - At __init__, jointly generate a pool of linearly separable feature vectors X with n_classes = #keys.
      - Ensure each class label y in [0..N-1] appears at least once (coverage).
      - Deterministically map each key (stable order) to its class index i and take the first sample with y==i.
      - Turn that 1D vector into a full-size noise image by patch-repeat tiling, matching [C_in,H,W] or [C_in,D,H,W].
      - Strength control: first L2-normalize to target (computed from "linf coefficient" + feature_dim),
        then per-pixel clamp to [-epsilon, +epsilon].
      - Store everything on CPU, float32. Unknown key -> raise.

    ROI Mode (for Medical Image Segmentation):
      - When roi_mode="binary", generates TWO noise patterns per key:
        * Pattern 0: Background noise (for pixels with label == 0)
        * Pattern 1: Foreground noise (for pixels with label > 0)
      - Use get_noise_with_mask(key, mask) to blend patterns based on segmentation mask.
      - This adapts LSP for MIS tasks by treating pixels inside and outside ROI as two distinct classes.

    Parameters
    ----------
    keys: Iterable[Hashable]
        Unique identifiers (identity_id or image_id). One noise per unique key.
    image_size: (C_in, H, W) or (C_in, D, H, W)
        Expected noise shape. 3-tuple for 2D, 4-tuple for 3D volumes.
    epsilon: float
        Per-pixel clamp bound in input (pre-normalize) space.
    noise_frame_size: int
        Size of the "block" that will be repeated into the final image. Support {8, 32} to mirror your constants.
    seed: int
        Global seed to make the joint generation deterministic.
    class_sep: float
        Passed to make_classification (default 10.0, mirroring your script).
    oversample_ratio: float
        Extra fraction over N to draw in one shot, to reduce the chance of class-coverage gaps.
        We will also loop with seed offsets until full coverage is achieved.
    roi_mode: Optional[str]
        ROI classification mode. None (default) = standard LSP, "binary" = ROI-based with 2 patterns per key.
    num_labels: int
        Number of segmentation labels (e.g., 4 for BraTS). Only used when roi_mode="binary".

    Methods
    -------
    get_noise(key, perturb_type) -> torch.FloatTensor [C_in,H,W] or [C_in,D,H,W] (CPU, float32)
        Returns the stored noise. 'perturb_type' is ignored (kept for API parity).
        In ROI mode, returns background noise (pattern 0).
    get_noise_with_mask(key, mask) -> torch.FloatTensor (CPU, float32)
        ROI mode only. Returns blended noise based on segmentation mask.
        mask: [H,W] or [D,H,W], values in {0, 1, ..., num_labels-1}
    """
    REQUIRES_KEYS_AT_INIT: ClassVar[bool] = True

    def __init__(
        self,
        *,
        keys: Iterable[Hashable],
        image_size: Tuple[int, ...],
        epsilon: float,
        noise_frame_size: int,
        seed: int = 0,
        class_sep: float = 10.0,
        oversample_ratio: float = 0.2,
        roi_mode: Optional[str] = None,
        num_labels: int = 4,
    ):
        # Index and shapes
        self.key2idx, self.uniq_keys = _make_key_index(keys)

        # Support both 2D (C,H,W) and 3D (C,D,H,W)
        if len(image_size) == 3:
            self.C_in, self.H, self.W = map(int, image_size)
            self.D = 1
            self.ndim = 2
        elif len(image_size) == 4:
            self.C_in, self.D, self.H, self.W = map(int, image_size)
            self.ndim = 3
        else:
            raise ValueError(f"[LSP] image_size must be (C,H,W) or (C,D,H,W), got {image_size}")

        self.eps: float = float(epsilon)
        self.frame: int = int(noise_frame_size)
        self.seed: int = int(seed)
        self.class_sep: float = float(class_sep)
        self.oversample_ratio: float = float(oversample_ratio)
        self.roi_mode: Optional[str] = None if roi_mode is None else str(roi_mode).lower()
        self.num_labels: int = int(num_labels)

        N = len(self.uniq_keys)
        if N == 0:
            raise ValueError("[LSP] 'keys' is empty.")
        if self.C_in <= 0 or self.H <= 0 or self.W <= 0 or self.D <= 0:
            raise ValueError(f"[LSP] Bad image_size: {image_size}")

        if self.roi_mode is not None and self.roi_mode not in ("binary",):
            raise ValueError(f"[LSP] roi_mode must be None or 'binary', got {self.roi_mode!r}")

        # 1) Decide patch grid & n_features
        if self.ndim == 3:
            pd = math.ceil(self.D / self.frame)
            ph = math.ceil(self.H / self.frame)
            pw = math.ceil(self.W / self.frame)
            self._n_features = pd * ph * pw * self.C_in
        else:  # 2D
            ph = math.ceil(self.H / self.frame)
            pw = math.ceil(self.W / self.frame)
            self._n_features = ph * pw * self.C_in

        if self._n_features <= 0:
            raise RuntimeError("[LSP] n_features computed as 0. Check image_size / noise_frame_size.")

        # 2) Generate linearly separable features
        #    Standard mode: n_classes = N (one per key)
        #    ROI mode: n_classes = 2 (background vs foreground), generate N samples for each class
        if self.roi_mode == "binary":
            # ROI mode: generate TWO noise patterns per key
            n_classes = 2
            # Need N samples from each class
            X_per_class: Dict[int, List[np.ndarray]] = {0: [], 1: []}
            round_id = 0

            while len(X_per_class[0]) < N or len(X_per_class[1]) < N:
                need = max(N - len(X_per_class[0]), N - len(X_per_class[1]))
                base = max(need, int(math.ceil(1.2 * N)))
                n_draw = int(base)
                rs = self.seed + round_id

                X, y = make_classification(
                    n_samples=n_draw,
                    n_features=self._n_features,
                    n_informative=self._n_features,
                    n_redundant=0,
                    n_repeated=0,
                    n_classes=2,  # Binary: background vs ROI
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
                    raise RuntimeError("[LSP] Could not generate sufficient ROI samples after 64 rounds.")

            # Allocate two tables: background and foreground
            if self.ndim == 3:
                self._table_bg = torch.empty((N, self.C_in, self.D, self.H, self.W), dtype=torch.float32, device="cpu")
                self._table_roi = torch.empty((N, self.C_in, self.D, self.H, self.W), dtype=torch.float32, device="cpu")
            else:
                self._table_bg = torch.empty((N, self.C_in, self.H, self.W), dtype=torch.float32, device="cpu")
                self._table_roi = torch.empty((N, self.C_in, self.H, self.W), dtype=torch.float32, device="cpu")

            # Populate tables
            for key, i in self.key2idx.items():
                # Background noise (class 0)
                x_vec_bg = X_per_class[0][i]
                if self.ndim == 3:
                    noise_bg = _reshape_repeat_crop_to_volume(x_vec_bg, self.C_in, self.D, self.H, self.W, self.frame)
                else:
                    noise_bg = _reshape_repeat_crop_to_image(x_vec_bg, self.C_in, self.H, self.W, self.frame)
                normalize_linf_(noise_bg, self.eps)
                self._table_bg[i].copy_(noise_bg)

                # ROI noise (class 1)
                x_vec_roi = X_per_class[1][i]
                if self.ndim == 3:
                    noise_roi = _reshape_repeat_crop_to_volume(x_vec_roi, self.C_in, self.D, self.H, self.W, self.frame)
                else:
                    noise_roi = _reshape_repeat_crop_to_image(x_vec_roi, self.C_in, self.H, self.W, self.frame)
                normalize_linf_(noise_roi, self.eps)
                self._table_roi[i].copy_(noise_roi)

            self._table = None  # Not used in ROI mode

        else:
            # Standard mode: one noise per key
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
                    n_classes=N,  # One class per key
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
                    raise RuntimeError("[LSP] Could not achieve per-class coverage after 64 rounds.")

            # Allocate table
            if self.ndim == 3:
                self._table = torch.empty((N, self.C_in, self.D, self.H, self.W), dtype=torch.float32, device="cpu")
            else:
                self._table = torch.empty((N, self.C_in, self.H, self.W), dtype=torch.float32, device="cpu")

            # Populate table
            for key, i in self.key2idx.items():
                x_vec = X_per_class[i]
                if self.ndim == 3:
                    noise = _reshape_repeat_crop_to_volume(x_vec, self.C_in, self.D, self.H, self.W, self.frame)
                else:
                    noise = _reshape_repeat_crop_to_image(x_vec, self.C_in, self.H, self.W, self.frame)
                normalize_linf_(noise, self.eps)
                self._table[i].copy_(noise)

            self._table_bg = None  # Not used in standard mode
            self._table_roi = None

    # ------------------- export-time API ------------------- #
    @torch.no_grad()
    def get_noise(self, key_raw: Hashable, perturb_type: Optional[str] = None) -> torch.Tensor:
        """
        Return stored noise (CPU, float32). Unknown key -> raise.
        'perturb_type' is ignored (kept for API parity with other providers).

        In ROI mode, returns background noise (pattern 0).
        For ROI-aware noise, use get_noise_with_mask() instead.

        Returns
        -------
        torch.Tensor
            [C_in,H,W] for 2D or [C_in,D,H,W] for 3D
        """
        k = _canon_key(key_raw)
        if k not in self.key2idx:
            raise KeyError(f"[LSP] Unknown key: {repr(key_raw)}")
        i = self.key2idx[k]

        if self.roi_mode == "binary":
            # In ROI mode, return background noise for compatibility
            return self._table_bg[i].clone().clamp_(-self.eps, +self.eps)
        else:
            return self._table[i].clone().clamp_(-self.eps, +self.eps)

    @torch.no_grad()
    def get_noise_with_mask(self, key_raw: Hashable, mask: torch.Tensor) -> torch.Tensor:
        """
        ROI mode only: Return blended noise based on segmentation mask.

        This method implements the ROI-based LSP adaptation for MIS tasks:
        "To adapt LSP and AR for MIS tasks, we treat pixels inside and outside
        the ROI as two distinct classes."

        Parameters
        ----------
        key_raw: Hashable
            Sample identifier
        mask: torch.Tensor
            Segmentation mask, shape [H,W] or [D,H,W]
            Values in {0, 1, ..., num_labels-1}

        Returns
        -------
        noise: torch.Tensor
            Blended noise [C,H,W] or [C,D,H,W]
            Background noise where mask==0, ROI noise where mask>0

        Raises
        ------
        RuntimeError
            If not in ROI mode
        KeyError
            If key not found
        """
        if self.roi_mode != "binary":
            raise RuntimeError(
                "[LSP] get_noise_with_mask() requires roi_mode='binary'. "
                f"Current roi_mode={self.roi_mode!r}. Use get_noise() for standard mode."
            )

        k = _canon_key(key_raw)
        if k not in self.key2idx:
            raise KeyError(f"[LSP] Unknown key: {repr(key_raw)}")
        i = self.key2idx[k]

        # Get both noise patterns
        noise_bg = self._table_bg[i].clone()   # [C,D,H,W] or [C,H,W]
        noise_roi = self._table_roi[i].clone()

        # Create ROI mask: ROI = (label > 0)
        roi_mask = (mask > 0).float()  # [D,H,W] or [H,W]

        # Expand to match noise shape
        if self.ndim == 3:
            roi_mask = roi_mask.unsqueeze(0)  # [1,D,H,W]
        else:
            roi_mask = roi_mask.unsqueeze(0)  # [1,H,W]

        # Blend: background where mask==0, ROI where mask>0
        noise = (1 - roi_mask) * noise_bg + roi_mask * noise_roi

        return noise.clamp_(-self.eps, +self.eps)