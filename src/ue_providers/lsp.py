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


@register_provider("lsp")
class LSPProvider:
    """
    Linear-Separable noise via make_classification (joint generation at init).

    Intent:
      - At __init__, jointly generate a pool of linearly separable feature vectors X with n_classes = #keys.
      - Ensure each class label y in [0..N-1] appears at least once (coverage).
      - Deterministically map each key (stable order) to its class index i and take the first sample with y==i.
      - Turn that 1D vector into a full-size noise image by patch-repeat tiling, matching [C_in,H,W].
      - Strength control: first L2-normalize to target (computed from "linf coefficient" + feature_dim),
        then per-pixel clamp to [-epsilon, +epsilon].
      - Store everything on CPU, float32. Unknown key -> raise.

    Parameters
    ----------
    keys: Iterable[Hashable]
        Unique identifiers (identity_id or image_id). One noise per unique key.
    image_size: (C_in, H, W)
        Expected noise shape (exactly matches your model input channels & resolution).
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

    Methods
    -------
    get_noise(key, perturb_type) -> torch.FloatTensor [C_in,H,W] (CPU, float32)
        Returns the stored noise. 'perturb_type' is ignored (kept for API parity).
    """
    REQUIRES_KEYS_AT_INIT: ClassVar[bool] = True

    def __init__(
        self,
        *,
        keys: Iterable[Hashable],
        image_size: Tuple[int, int, int],
        epsilon: float,
        noise_frame_size: int,
        seed: int = 0,
        class_sep: float = 10.0,
        oversample_ratio: float = 0.2,
    ):
        # Index and shapes
        self.key2idx, self.uniq_keys = _make_key_index(keys)
        self.C_in, self.H, self.W = map(int, image_size)
        self.eps: float = float(epsilon)
        self.frame: int = int(noise_frame_size)
        self.seed: int = int(seed)
        self.class_sep: float = float(class_sep)
        self.oversample_ratio: float = float(oversample_ratio)

        N = len(self.uniq_keys)
        if N == 0:
            raise ValueError("[LSP-MC] 'keys' is empty.")
        if self.C_in <= 0 or self.H <= 0 or self.W <= 0:
            raise ValueError(f"[LSP-MC] Bad image_size: {image_size}")

        # 1) Decide patch grid & n_features (per your script logic)
        ph = math.ceil(self.H / self.frame)
        pw = math.ceil(self.W / self.frame)
        self._n_features = ph * pw * self.C_in
        if self._n_features <= 0:
            raise RuntimeError("[LSP-MC] n_features computed as 0. Check image_size / noise_frame_size.")

        # 2) Jointly generate a large enough batch so that every class label in [0..N-1] appears at least once.
        #    We do deterministic rounds with seed offsets until coverage is satisfied.
        X_per_class: Dict[int, np.ndarray] = {}
        round_id = 0
        rng = np.random.RandomState(self.seed)  # only used for fallback sizing; sklearn uses the round seed
        while len(X_per_class) < N:
            # Draw this round
            need = N - len(X_per_class)
            base = max(need, int(math.ceil((1.0 + self.oversample_ratio) * N)))
            n_draw = int(base)
            rs = self.seed + round_id  # deterministic across runs
            X, y = make_classification(
                n_samples=n_draw,
                n_features=self._n_features,
                n_informative=self._n_features,
                n_redundant=0,
                n_repeated=0,
                n_classes=N,                   # <-- one class per key
                n_clusters_per_class=1,
                class_sep=self.class_sep,
                flip_y=0.0,
                random_state=rs,
            )
            # Fill missing classes with the first occurrence from this round
            for xi, yi in zip(X, y):
                if yi not in X_per_class:
                    X_per_class[int(yi)] = xi.astype(np.float32, copy=False)
                    if len(X_per_class) == N:
                        break
            round_id += 1
            if round_id > 64:
                # Extremely unlikely unless parameters are pathological
                raise RuntimeError("[LSP-MC] Could not achieve per-class coverage after 64 rounds. Check n_features / N.")

        # 3) Allocate table and populate per key (stable order)
        self._table = torch.empty((N, self.C_in, self.H, self.W), dtype=torch.float32, device="cpu")

        for key, i in self.key2idx.items():
            # Take the class-equal vector
            x_vec = X_per_class[i]  # np.ndarray [n_features]
            # Turn into full-size image noise (faithful reshape -> repeat -> crop)
            noise = _reshape_repeat_crop_to_image(x_vec, self.C_in, self.H, self.W, self.frame)  # torch [C,H,W] float32 CPU
            # Strength control: L2 then per-pixel clamp (same order as your script)
            normalize_linf_(noise, self.eps)
            self._table[i].copy_(noise)

    # ------------------- export-time API ------------------- #
    @torch.no_grad()
    def get_noise(self, key_raw: Hashable, perturb_type: Optional[str] = None) -> torch.Tensor:
        """
        Return stored noise [C_in,H,W] (CPU, float32). Unknown key -> raise.
        'perturb_type' is ignored (kept for API parity with other providers).
        """
        k = _canon_key(key_raw)
        if k not in self.key2idx:
            raise KeyError(f"[LSP-MC] Unknown key: {repr(key_raw)}")
        i = self.key2idx[k]
        # Defensive clamp (harmless, ensures boundary if table modified elsewhere):
        return self._table[i].clone().clamp_(-self.eps, +self.eps)
