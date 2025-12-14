from __future__ import annotations
from typing import Dict, Any, List

import torch
from torch.utils.data import Dataset, ConcatDataset
from omegaconf import DictConfig

from ..utils.config import get_config, require_config
from ..core.ue_keys import extract_key


class UEConcatDataset(ConcatDataset):
    """
    ConcatDataset with labels_for_sampling support.
    所有子数据集都必须实现 labels_for_sampling(kind)，然后在这里按顺序拼接。
    """

    def labels_for_sampling(self, kind: str = "reid") -> torch.Tensor:
        parts: List[torch.Tensor] = []
        for ds in self.datasets:
            if not hasattr(ds, "labels_for_sampling"):
                raise RuntimeError(
                    "UEConcatDataset requires child datasets to implement labels_for_sampling(...)"
                )
            lab = ds.labels_for_sampling(kind)
            lab = lab.to(torch.long) if isinstance(lab, torch.Tensor) else torch.as_tensor(lab, dtype=torch.long)
            parts.append(lab)

        if not parts:
            return torch.empty(0, dtype=torch.long)
        return torch.cat(parts, dim=0)


class UEKeyDataset(Dataset):
    """
    Wrap 一个 base dataset，为每个 sample 额外加上 'key' 字段。

    key_spec (DictConfig):
      - type:  "samplewise" | "classwise"          # 用于与 perturb_type 做一致性检查
      - from:  "index" | "field" | "filename"      # key 的来源
      - field: 当 from=="field" 时的路径（支持点路径，如 "targets.reid"）
      - lower: bool (默认 True)                    # string key 是否转小写
      - strip: bool (默认 True)                    # string key 是否 strip
      - namespace: str (可选)                      # 仅用于记录 / 导出，不参与 key 比较
    """

    def __init__(self, base: Dataset, key_spec: DictConfig):
        if not isinstance(key_spec, DictConfig):
            raise TypeError("UEKeyDataset only accepts DictConfig type for key_spec")

        self.base = base
        self._kspec: DictConfig = key_spec

        self._ktype: str = require_config(self._kspec, "type", type_=str)
        self._kfrom: str = get_config(self._kspec, "from", "field")
        self._ffield: str | None = get_config(self._kspec, "field", None, type_=str)

        self._lower: bool = bool(get_config(self._kspec, "lower", True))
        self._strip: bool = bool(get_config(self._kspec, "strip", True))

    def __len__(self) -> int:
        return len(self.base)

    def __getattr__(self, name: str):
        # 将未知属性/方法转发给 base dataset（例如 labels_for_sampling）
        return getattr(self.base, name)

    def labels_for_sampling(self, kind: str = "reid") -> torch.Tensor:
        if hasattr(self.base, "labels_for_sampling"):
            return self.base.labels_for_sampling(kind)
        raise RuntimeError("Underlying dataset does not implement labels_for_sampling(...)")

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        s = self.base[idx]
        key = extract_key(s, idx, self._kspec)

        ktype = str(self._ktype)
        if ktype not in ("classwise", "samplewise"):
            raise ValueError(f"Invalid ue.key.type: {ktype}")

        s["key"] = key
        return s
