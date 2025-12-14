from __future__ import annotations
from typing import Optional, Tuple, Dict, Any
import torch
import numpy as np
import hashlib
from ..registry import register_provider


@register_provider("random_noise")
class RandomNoiseProvider:
    """
    RandomNoiseProvider — training-free, key-deterministic random perturbations.

    Supported modes (mode):
      - "uniform":   U(-1, 1) then L∞ normalization -> [-eps, eps]
      - "gaussian":  N(0,1) then L∞ normalization -> [-eps, eps]
      - "rademacher": with 0.5 probability, set to +eps or -eps (equivalent to sign noise)
      - "saltpepper": with probability p, set pixel to +eps or -eps, others are 0
                      params example: {"p": 0.01, "pepper_prob": 0.5}
      - "sparse":    sparse additive noise: with sparsity q, sample U(-1,1), others are 0
                      params example: {"q": 0.05}

    Common parameters
    ----------
    epsilon : float
        L∞ bound (in pre-normalized space), e.g., 8/255.
    image_size : (C, H, W)
        Input channels, height, width (before Normalize).
    seed : int
        Global seed; local seed is derived from (seed, key, key_type).
    mode : str
        See the mode descriptions above.
    params : dict
        Mode-specific parameters, see comments for each mode.
    tied_channels : bool
        If True, reuse the same channel noise for all channels (useful for grayscale/achromatic scenarios).
    device : Optional[torch.device]
        Computation device; default is CPU.
    dtype : torch.dtype
        Output dtype (internal float32, converted to this dtype at the end).

    Methods
    ----
    get_noise(key, key_type) -> FloatTensor [C,H,W] ∈ [-eps, +eps] (CPU)
        key_type ∈ {"classwise", "samplewise"}
    """

    def __init__(
        self,
        *,
        epsilon: float,
        image_size: Tuple[int, int, int],
        seed: int = 0,
        mode: str = "uniform",
        params: Optional[Dict[str, Any]] = None,
        tied_channels: bool = False,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ):
        self.eps = float(epsilon)
        self.C, self.H, self.W = map(int, image_size)
        self.seed = int(seed)
        self.mode = str(mode).lower()
        self.params = params or {}
        self.tied_channels = bool(tied_channels)
        self.device = device
        self.out_dtype = dtype

        _valid = {"uniform", "gaussian", "rademacher", "saltpepper", "sparse"}
        if self.mode not in _valid:
            raise ValueError(f"[RandomNoise] Unsupported mode: {self.mode}. "
                             f"Choose from {_valid}.")

    # ---------- public API ----------

    @torch.no_grad()
    def get_noise(self, key, key_type: str) -> torch.Tensor:
        if key_type not in ("classwise", "samplewise"):
            raise ValueError(f"[RandomNoise] Unknown key_type: {key_type}")

        kid = self._hashable_int(key)
        if key_type == "classwise":
            lseed = (self.seed * 1315423911 + kid) & 0x7FFFFFFF
        else:
            lseed = (self.seed * 2654435761 + kid) & 0x7FFFFFFF

        if self.tied_channels:
            one = self._synthesize_one(lseed)  # [1,H,W]
            noise = one.expand(self.C, self.H, self.W).contiguous()
        else:
            noise_list = []
            for c in range(self.C):
                lseed_c = (lseed + (c + 1) * 104729) & 0x7FFFFFFF
                noise_list.append(self._synthesize_one(lseed_c))  # [1,H,W]
            noise = torch.cat(noise_list, dim=0)  # [C,H,W]

        noise = noise.clamp(min=-self.eps, max=self.eps).to(self.out_dtype)
        return noise.cpu()

    # ---------- internals ----------

    @staticmethod
    def _hashable_int(x) -> int:
        if isinstance(x, (int, np.integer)):
            return int(x)
        if isinstance(x, torch.Tensor):
            if x.numel() != 1:
                raise ValueError("Key tensor must be scalar.")
            return int(x.item())
        if isinstance(x, (str, bytes, bytearray)):
            h = hashlib.sha1(str(x).encode("utf-8")).hexdigest()
            return int(h[:16], 16)
        try:
            return int(x)
        except Exception:
            h = hashlib.sha1(repr(x).encode("utf-8")).hexdigest()
            return int(h[:16], 16)

    @torch.no_grad()
    def _synthesize_one(self, local_seed: int) -> torch.Tensor:
        """
        Generate one-channel noise [1,H,W], deterministic with respect to local_seed.
        All modes will finally normalize/clip to [-eps, eps].
        """
        gen = torch.Generator(device="cpu")
        gen.manual_seed(int(local_seed))

        d = self.device if self.device is not None else "cpu"

        if self.mode == "uniform":
            arr = torch.empty((1, self.H, self.W), dtype=torch.float32, device="cpu")
            arr.uniform_(-1.0, 1.0, generator=gen)
            arr = arr.to(d)
            # directly scale to [-eps, eps]
            arr = arr * self.eps
            return arr

        elif self.mode == "gaussian":
            arr = torch.randn((1, self.H, self.W), generator=gen, dtype=torch.float32, device="cpu")
            arr = arr.to(d)
            # L∞ normalization to avoid Gaussian tail exceeding eps
            max_abs = arr.abs().amax(dim=(-1, -2), keepdim=True) + 1e-12
            arr = arr / max_abs * self.eps
            return arr

        elif self.mode == "rademacher":
            # with 0.5 probability, set to +eps or -eps
            bern = torch.rand((1, self.H, self.W), generator=gen, dtype=torch.float32, device="cpu")
            sign = torch.where(bern < 0.5, -1.0, 1.0)
            arr = (sign * self.eps).to(d)
            return arr

        elif self.mode == "saltpepper":
            # sparse pulse noise: set to ±eps with probability p, others are 0
            p = float(self.params.get("p", 0.01))  # 单像素被扰动的概率
            pepper_prob = float(self.params.get("pepper_prob", 0.5))  # 负脉冲比例
            r = torch.rand((1, self.H, self.W), generator=gen, dtype=torch.float32, device="cpu")
            rp = torch.rand((1, self.H, self.W), generator=gen, dtype=torch.float32, device="cpu")

            zeros = torch.zeros((1, self.H, self.W), dtype=torch.float32)
            pos = torch.full((1, self.H, self.W), self.eps, dtype=torch.float32)
            neg = torch.full((1, self.H, self.W), -self.eps, dtype=torch.float32)

            arr = torch.where(r > p, zeros, torch.where(rp < pepper_prob, neg, pos))
            arr = arr.to(d)
            return arr

        elif self.mode == "sparse":
            # sparse additive: sample U(-1,1) with probability q, others are 0, then normalize/clip to eps
            q = float(self.params.get("q", 0.05))
            mask = (torch.rand((1, self.H, self.W), generator=gen, dtype=torch.float32, device="cpu") < q).float()
            vals = torch.empty((1, self.H, self.W), dtype=torch.float32, device="cpu").uniform_(-1.0, 1.0, generator=gen)
            arr = (mask * vals).to(d) * self.eps
            return arr

        else:
            raise RuntimeError(f"[RandomNoise] Unknown mode at runtime: {self.mode}")
