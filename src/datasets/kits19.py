# file: src/datasets/kits19.py
from __future__ import annotations

import os
from typing import Optional, Callable, Any, List, Union, Dict

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from omegaconf import DictConfig

from ..utils.logger import get_logger
from ..utils.config import require_config, get_config
from ..registry import register_dataset_builder
from .base_builder import BaseDatasetBuilder, BaseUEBuilder
from .transforms import get_seg_transforms


# ======================================================================
#   KiTS19 3D Volume Dataset
# ======================================================================

class KiTS19VolumeDataset(Dataset):
    """
    KiTS19 3D 肾脏 CT 数据集（用于 3D 分割 / UE 基任务）。

    依赖预处理脚本生成的「每个 split 一个 CSV」，例如：
      - train_csv_path: train.csv
      - val_csv_path:   val.csv
      - test_csv_path:  test.csv

    每个 CSV 必须至少包含以下列：
      - case_id:     KiTS19 病例 ID (e.g., case_00000)
      - grade:       空字符串（KiTS19 没有 grade 信息，保留字段以保持兼容）
      - volume_path: 指向 .h5 文件（里边存 image + label）

    .h5 文件内部结构（预处理脚本写入）：
      - dataset:
          - "image": float32, shape (C, H, W, D)，范围 [0,1]，C=1 for CT
          - "label": uint8,   shape (H, W, D)，值 ∈ {0:background, 1:kidney, 2:tumor}
      - attrs:
          - "case_id": str（可选，用于一致性检查）
    """

    def __init__(
        self,
        csv_path: str,
        split: str = "train",
        grades: Optional[Union[str, List[str]]] = None,
        transform: Optional[Callable[[torch.Tensor, torch.Tensor], Any]] = None,
        logger=None,
    ):
        super().__init__()
        self.logger = logger or get_logger()
        self.csv_path = csv_path
        self.split = str(split).lower()
        self.transform = transform

        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"[KiTS19] CSV not found: {csv_path}")

        df = pd.read_csv(csv_path)
        required_cols = ["case_id", "grade", "volume_path"]
        for c in required_cols:
            if c not in df.columns:
                raise ValueError(f"[KiTS19] CSV missing required column: {c}")

        # KiTS19 没有 grade 过滤，但保留接口以保持兼容
        if grades is not None:
            if isinstance(grades, str):
                grades = [grades]
            grades_upper = [g.upper() for g in grades]
            df["grade"] = df["grade"].astype(str).str.upper()
            df = df[df["grade"].isin(grades_upper)].reset_index(drop=True)

        if len(df) == 0:
            raise ValueError(
                f"[KiTS19] No samples in CSV after filtering: "
                f"csv_path={csv_path}, split={self.split}, grades={grades}"
            )

        self.df = df.reset_index(drop=True)

        self.logger.info(
            f"[KiTS19] Loaded split='{self.split}' from {csv_path}: "
            f"{len(self.df)} cases"
        )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        h5_path = row["volume_path"]

        if not os.path.exists(h5_path):
            raise FileNotFoundError(f"[KiTS19] h5 file not found: {h5_path}")

        # 读取 h5
        with h5py.File(h5_path, "r") as f:
            image_np = f["image"][()]  # (C, H, W, D), float32, [0,1], C=1 for CT
            label_np = f["label"][()]  # (H, W, D), uint8
            case_id_attr = f.attrs.get("case_id", None)

        if image_np.ndim != 4:
            raise ValueError(f"[KiTS19] image ndim={image_np.ndim}, expected 4 (C,H,W,D)")
        if label_np.ndim != 3:
            raise ValueError(f"[KiTS19] label ndim={label_np.ndim}, expected 3 (H,W,D)")

        # numpy -> torch，重排到 [C, D, H, W] / [D, H, W]
        image = torch.from_numpy(image_np).float().permute(0, 3, 1, 2)  # [C,D,H,W]
        label = torch.from_numpy(label_np.astype(np.int64)).long().permute(2, 0, 1)  # [D,H,W]

        if self.transform is not None:
            out = self.transform(image, label)
            if isinstance(out, (tuple, list)) and len(out) == 2:
                image, label = out
            else:
                raise RuntimeError(
                    "[KiTS19] transform must return (image, label), "
                    f"got type={type(out)}"
                )

        case_id = str(row["case_id"])
        grade = str(row["grade"])

        # 可选一致性检查
        if case_id_attr is not None and str(case_id_attr) != case_id:
            self.logger.warning(
                f"[KiTS19] case_id mismatch: CSV={case_id}, h5.attr={case_id_attr}"
            )

        return {
            "image": image,       # [C,D,H,W], C=1 for CT
            "label": label,       # [D,H,W]
            "case_id": case_id,
            "grade": grade,
            "index": int(idx),    # for UE noise indexing
            "h5_path": h5_path,
        }


# ======================================================================
#   KiTS19 Builder（继承 BaseDatasetBuilder）
# ======================================================================

class Kits19Builder(BaseDatasetBuilder):
    """
    通用 KiTS19 Builder（3D 分割基任务）。

    配置示例：

    dataset:
      name: kits19_seg
      train_csv_path: /.../train.csv
      val_csv_path:   /.../val.csv
      test_csv_path:  /.../test.csv
      grades: null
    """

    def __init__(self, config: DictConfig):
        super().__init__(config)
        dcfg: DictConfig = require_config(config, "dataset")

        train_csv = require_config(dcfg, "train_csv_path", type_=str)
        val_csv   = require_config(dcfg, "val_csv_path", type_=str)
        test_csv  = require_config(dcfg, "test_csv_path", type_=str)
        self.csv_paths = {
            "train": train_csv,
            "val":   val_csv,
            "test":  test_csv,
        }

        self.grades = get_config(dcfg, "grades", None)

    def build_dataset(self, split: str, **overrides) -> Dataset:
        """
        根据 split 构建对应的 KiTS19VolumeDataset。
        """
        split_norm = self._normalize_split(split)

        csv_path = overrides.get("csv_path", self.csv_paths.get(split_norm))
        if csv_path is None:
            raise ValueError(f"[KiTS19] No CSV path configured for split '{split_norm}'.")

        grades = overrides.get("grades", self.grades)

        transform = overrides.get("transform", None)

        if transform is None:
            dcfg: DictConfig = require_config(self.config, "training.data")
            tcfg: DictConfig = get_config(dcfg, "transforms", DictConfig({}))

            normalize = bool(require_config(tcfg, "normalize"))
            geom_aug = bool(require_config(tcfg, "geom_aug"))
            intensity_aug = bool(require_config(tcfg, "intensity_aug"))
            # KiTS19 单通道
            mean = get_config(tcfg, "mean", [0.0])
            std = get_config(tcfg, "std", [1.0])

            transform = get_seg_transforms(
                ndim=3,
                split=split_norm,
                normalize=normalize,
                geom_aug=geom_aug,
                intensity_aug=intensity_aug,
                mean=mean,
                std=std,
            )

        ds = KiTS19VolumeDataset(
            csv_path=csv_path,
            split=split_norm,
            grades=grades,
            transform=transform,
            logger=self.logger,
        )
        return ds


# ======================================================================
#   Registry 注册：Seg 基任务 + UE 任务
# ======================================================================

@register_dataset_builder("kits19_seg")
class Kits19SegBuilder(Kits19Builder):
    """
    对应 segmentation 基任务：task.name = 'kits19_seg'
    """
    def __init__(self, config: DictConfig):
        super().__init__(config)


@register_dataset_builder("kits19_ue")
class Kits19UEBuilder(BaseUEBuilder):
    """
    对应 UE 训练任务：task.name = 'kits19_ue'

    沿用 BaseUEBuilder 的策略：
      - train: ConcatDataset(UEKey(train_clean), UEKey(val_clean))
      - val  : None
      - test : 复用 base_task_builder 的 test split
    """
    def __init__(self, config: DictConfig):
        super().__init__(config)
        self._base_builder_name = "kits19_seg"