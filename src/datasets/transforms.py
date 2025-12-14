# file: src/utils/transforms.py
from __future__ import annotations
from typing import Callable, Tuple, Any, List, Sequence, Dict

import torch
from monai.transforms import (
    Compose,
    RandAxisFlipd,
    RandRotate90d,
    RandScaleIntensityd,
    RandShiftIntensityd,
)


def _build_3d_seg_transforms(
    split: str,
    *,
    normalize: bool = True,
    geom_aug: bool = True,
    intensity_aug: bool = True,
    mean: Sequence[float] | None = None,
    std: Sequence[float] | None = None,
) -> Callable[[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
    """
    3D segmentation 通用 transform：

      输入:  image [C,D,H,W] float32（BraTS 预处理后约在 [0,1]）
            label [D,H,W]    long/int
      输出:  image [C,D,H,W] float32
            label [D,H,W]    long

    - geom_aug / intensity_aug 只在 train split 生效
    - normalize=True 时，最后一步做 per-channel (x - mean) / std
    """
    split = str(split).lower()
    is_train = split == "train"

    if not is_train:
        # 非 train 默认不做增广，只做 normalize（如果 normalize=True）
        geom_aug = False
        intensity_aug = False

    xforms: List[Any] = []

    # ---------- 几何增强（3D aware） ----------
    if geom_aug:
        xforms.extend(
            [
                # 随机沿任一空间轴翻转（避免手写 0/1/2 导致 IndexError）
                RandAxisFlipd(
                    keys=["image", "label"],
                    prob=0.5,
                ),
                # 随机 90° 旋转（在任意一对空间轴上）
                RandRotate90d(
                    keys=["image", "label"],
                    prob=0.3,
                    max_k=3,
                ),
            ]
        )

    # ---------- 强度增强 ----------
    if intensity_aug:
        xforms.extend(
            [
                RandScaleIntensityd(
                    keys=["image"],
                    factors=0.1,
                    prob=0.5,
                ),
                RandShiftIntensityd(
                    keys=["image"],
                    offsets=0.1,
                    prob=0.5,
                ),
            ]
        )

    monai_compose = Compose(xforms)

    def _apply(image: torch.Tensor, label: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # image: [C, D, H, W]
        if image.ndim != 4:
            raise ValueError(f"[3DTransforms] expect image [C,D,H,W], got {tuple(image.shape)}")

        # label: [D, H, W] -> [1, D, H, W]，避免 RandAxisFlipd 维度不一致
        if label.ndim == 3:
            label_in = label.unsqueeze(0)  # [1, D, H, W]
        elif label.ndim == 4:
            label_in = label
        else:
            raise ValueError(f"[3DTransforms] expect label [D,H,W] or [1,D,H,W], got {tuple(label.shape)}")

        data = {"image": image, "label": label_in}
        out = monai_compose(data)

        img: torch.Tensor = out["image"]          # [C, D, H, W]
        lbl_out: torch.Tensor = out["label"]      # [1, D, H, W] 或 [K, D, H, W]

        # 如果只是我们自己加的那 1 个 channel，就再 squeeze 回去，保持和原先 pipeline 一致
        if lbl_out.ndim == 4 and lbl_out.shape[0] == 1:
            lbl_out = lbl_out[0]                  # [D, H, W]

        lbl_out = lbl_out.long()

        # -------- normalize 放在最后 --------
        if normalize:
            c = img.shape[0]
            if mean is None:
                mean_t = torch.zeros(c, dtype=img.dtype, device=img.device)
            else:
                mean_t = torch.as_tensor(mean, dtype=img.dtype, device=img.device)
            if mean_t.numel() == 1:
                mean_t = mean_t.repeat(c)
            if mean_t.numel() != c:
                raise RuntimeError(f"[3DTransforms] len(mean)={mean_t.numel()} != C={c}")

            if std is None:
                std_t = torch.ones(c, dtype=img.dtype, device=img.device)
            else:
                std_t = torch.as_tensor(std, dtype=img.dtype, device=img.device)
            if std_t.numel() == 1:
                std_t = std_t.repeat(c)
            if std_t.numel() != c:
                raise RuntimeError(f"[3DTransforms] len(std)={std_t.numel()} != C={c}")

            view_shape = (c,) + (1,) * (img.ndim - 1)  # [C,1,1,1]
            img = (img - mean_t.view(view_shape)) / std_t.view(view_shape)

        return img, lbl_out

    return _apply


def get_seg_transforms(
    *,
    ndim: int,
    split: str,
    normalize: bool = True,
    geom_aug: bool = True,
    intensity_aug: bool = True,
    mean: Sequence[float] | None = None,
    std: Sequence[float] | None = None,
) -> Callable[[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
    """
    统一入口：
      - 目前只实现 ndim == 3
    """
    if ndim == 3:
        return _build_3d_seg_transforms(
            split=split,
            normalize=normalize,
            geom_aug=geom_aug,
            intensity_aug=intensity_aug,
            mean=mean,
            std=std,
        )
    else:
        raise ValueError(
            f"get_seg_transforms currently only supports 3D (ndim=3). Got ndim={ndim}"
        )
