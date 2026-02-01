# file: src/core/ue_orchestrator.py
from __future__ import annotations
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import os

import torch
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from ..utils.config import get_config, require_config
from ..registry import PROVIDERS
from ..datasets.poisoned_dataset import PoisonedDataset
from .ue_artifacts import write_shards, write_files
from .ue_keys import collect_keys  # Unified key extraction and collection interface


# =======================================================================
# 1. 构造共享 provider（在线噪声提供者，通常用于 UE 生成 / 2D ReID 等）
# =======================================================================

def build_unlearnable_provider_instance(
    config: DictConfig,
    train_dataset,
    val_dataset=None,
) -> Optional[object]:
    """
    对于当前 3D 项目，victim training 不再依赖在线 provider。
    但仍保留该函数，用于：
      - training-based UE 生成阶段（如 learnable noise backend）
      - 未来可能的 2D ReID / 其他子任务

    若 provider 类声明 REQUIRES_KEYS_AT_INIT = True，则会先收集
    train ∪ val 的 key（由 ue.key 控制）并以 keys=... 传入构造函数。
    """
    pcfg = get_config(config, "training.data.poison", OmegaConf.create({}))
    if not bool(get_config(pcfg, "enabled", False)):
        return None

    # 只有 source.type == "provider" 时才需要构造在线 provider
    stype = str(require_config(config, "training.data.poison.source.type")).lower()
    if stype != "provider":
        return None

    tcfg = require_config(config, "training.data.transforms")
    src = get_config(pcfg, "source", OmegaConf.create({}))
    prov = get_config(src, "provider", OmegaConf.create({}))

    name = str(get_config(prov, "name", "")).lower()
    params = dict(get_config(prov, "params", OmegaConf.create({})))

    # image_size: 直接视为 3D 体的尺寸 [C,D,H,W]
    if "image_size" not in params:
        params["image_size"] = tuple(require_config(tcfg, "image_size"))

    ProviderCls = PROVIDERS.get(name)
    if ProviderCls is None:
        raise ValueError(f"[UE] Unknown provider: {name!r}")

    requires_keys = bool(getattr(ProviderCls, "REQUIRES_KEYS_AT_INIT", False))

    if requires_keys:
        # key_spec: 统一从 ue.key 中取
        key_spec = get_config(
            config,
            "ue.key",
            OmegaConf.create({"type": "samplewise", "from": "index"}),
        )
        ktype = str(get_config(key_spec, "type", "samplewise")).lower()
        classwise = ktype == "classwise"

        union: List[Any] = []
        seen = set()
        for ds in (train_dataset, val_dataset):
            if ds is None:
                continue
            ks = collect_keys(ds, key_spec, classwise=classwise)
            if classwise:
                for k in ks:
                    if k not in seen:
                        union.append(k)
                        seen.add(k)
            else:
                union.extend(ks)

        params = dict(params)
        params["keys"] = union  # provider 内部知道如何处理 keys

    provider_instance = ProviderCls(**params)
    return provider_instance


# =======================================================================
# 2. 训练阶段：将干净 dataset 包装成 PoisonedDataset（3D seg）
#    -> 仅支持「从磁盘 manifest 读噪声」，不再支持在线 provider 打毒
# =======================================================================

def attach_unlearnable_noise(
    config: DictConfig,
    dataset,
    *,
    provider_instance: Optional[object] = None,  # 参数保留，但在 3D pipeline 中不再使用
):
    """
    3D segmentation 训练阶段，将干净 dataset 包装为 PoisonedDataset。

    配置约定（统一 3D 版）：

      training:
        data:
          transforms:
            mean: [0.0, 0.0, 0.0]      # Normalize 均值（3D 通道）
            std:  [1.0, 1.0, 1.0]      # Normalize 方差
          poison:
            enabled: true/false
            key: {...}                 # ue.key 的一个副本或子集
            perturb_type: "classwise" | "samplewise"
            clamp_min: 0.0
            clamp_max: 1.0
            apply_stage: "before_normalize"   # 目前只支持这个
            source:
              type: "files" | "shards" | "manifest"
              manifest_path: "/path/to/manifest.json"

    逻辑（3D 版约简）：
      - 若 poison.enabled=False：直接返回原 dataset。
      - 总是通过 manifest_path + UEShardsAccessor 读离线噪声，
        不再支持在线 provider 打毒，即使外部传了 provider_instance 也会被忽略。
    """
    pcfg = get_config(config, "training.data.poison", OmegaConf.create({}))
    if not bool(get_config(pcfg, "enabled", False)):
        return dataset

    if provider_instance is not None:
        # 明确提示：3D pipeline 不再用 provider 打毒
        print(
            "[UE] Note: provider_instance is ignored in attach_unlearnable_noise; "
            "3D pipeline only reads noise from offline manifest."
        )

    tcfg = get_config(config, "training.data.transforms", OmegaConf.create({}))
    key_spec = require_config(pcfg, "key")
    perturb_type = str(require_config(pcfg, "perturb_type"))

    mean = tuple(get_config(tcfg, "mean", [0.0, 0.0, 0.0]))
    std = tuple(get_config(tcfg, "std", [1.0, 1.0, 1.0]))

    src_cfg = require_config(pcfg, "source")
    stype = str(get_config(src_cfg, "type", "files")).lower()
    if stype not in {"files", "shards", "manifest"}:
        raise ValueError(
            f"[UE] 3D pipeline only supports poison.source.type in "
            f"{{'files','shards','manifest'}}, got {stype!r}"
        )

    manifest_path = require_config(src_cfg, "manifest_path", type_=str)

    clamp_min = float(get_config(pcfg, "clamp_min", 0.0))
    clamp_max = float(get_config(pcfg, "clamp_max", 1.0))
    apply_stage = str(get_config(pcfg, "apply_stage", "before_normalize"))

    # defense augmentation config (optional, for ablation experiments)
    defense_cfg = get_config(pcfg, "defense", None)
    if defense_cfg is not None:
        defense_cfg = OmegaConf.to_container(defense_cfg, resolve=True)

    # 统一走离线模式：PoisonedDataset 内部通过 UEShardsAccessor.from_manifest 读噪声
    return PoisonedDataset(
        base=dataset,
        perturb_type=perturb_type,
        key_spec=key_spec,
        source_cfg={"type": stype, "manifest_path": manifest_path},
        clamp=(clamp_min, clamp_max),
        apply_stage=apply_stage,
        mean=mean,
        std=std,
        defense_cfg=defense_cfg,
    )


# =======================================================================
# 3. 离线 / training-free UE 生成（3D volume）
# =======================================================================

def generate_training_free(
    config: DictConfig,
    datasets: Union[torch.utils.data.Dataset, Sequence[torch.utils.data.Dataset]],
) -> bool:
    """
    3D 版本的 training-free UE 生成（例如 3D LSP，无需 victim-training）。

    使用方式（示例）：

      ue:
        algorithm:
          kind: training_free
          name: lsp_3d
          params:
            epsilon: 0.0313725
            image_size: [C,D,H,W]
            seed: 0
        key:
          type: classwise
          from: field
          field: case_id
        store_dir: ./ue/brats19_lsp
        io:
          strategy: files      # 推荐：3DUE 用 files 即可
          shard_size: 1000     # 若 strategy=shards 时生效

    行为：
      1. 根据 ue.algorithm 选择 provider 类（从 PROVIDERS registry）。
      2. 用 ue.key 从给定 datasets 中收集所有 key（classwise 则去重）。
      3. 对每个 key 调用 provider.get_noise(key, perturb_type)，
         要求返回 [C,D,H,W] 的 3D 噪声。
      4. 调用 write_files 或 write_shards 将噪声写入磁盘，并生成 manifest.json。
    """
    ue_cfg = get_config(config, "ue", OmegaConf.create({}))
    alg = get_config(ue_cfg, "algorithm", OmegaConf.create({}))
    if str(get_config(alg, "kind", "")).lower() != "training_free":
        # 不是 training_free 算法则直接跳过
        return False

    prov_name = str(get_config(alg, "name", "")).lower()
    prov_cls = PROVIDERS.get(prov_name)
    if prov_cls is None:
        raise ValueError(f"[UE] Unknown training-free provider: {prov_name!r}")

    # provider params（不做假设，全部透传）
    params = dict(get_config(alg, "params", OmegaConf.create({})))
    epsilon = float(get_config(alg, "params.epsilon", params.get("epsilon", 0.0313725)))

    # ---- 统一整理 datasets → list ----
    ds_list = list(datasets) if isinstance(datasets, (list, tuple)) else [datasets]

    # ---- key 收集策略（3D、与维度无关）----
    key_spec = require_config(ue_cfg, "key")
    perturb_type = str(require_config(key_spec, "type"))
    classwise = perturb_type.lower() == "classwise"

    union: List[Any] = []
    seen = set()
    #@xinyao:增加dsnone检查
    for ds in ds_list:
        if ds is None:
            continue
        ks = collect_keys(ds, key_spec, classwise=classwise)
        if classwise:
            for k in ks:
                if k not in seen:
                    union.append(k)
                    seen.add(k)
        else:
            union.extend(ks)

    requires_keys = bool(getattr(prov_cls, "REQUIRES_KEYS_AT_INIT", False))
    prov_params = dict(params)
    if requires_keys:
        prov_params["keys"] = union

    provider = prov_cls(**prov_params)

    # ---- 真正生成 [key -> noise 3D volume] ----
    entries: List[Tuple[Any, torch.Tensor]] = []
    for k in tqdm(union, desc="UE gen (training-free 3D)", unit="key"):
        n = provider.get_noise(k, perturb_type)
        if not isinstance(n, torch.Tensor):
            raise TypeError(
                f"[UE] provider.get_noise must return torch.Tensor, got {type(n)}"
            )

        if n.ndim != 4:
            raise ValueError(
                f"[UE] 3D training-free UE expects noise shape [C,D,H,W], "
                f"got {tuple(n.shape)} for key={k!r}"
            )
        entries.append((k, n))

    # ---- 写入磁盘：files/shards（3D 版本）----
    store_dir = str(get_config(ue_cfg, "store_dir", os.path.join(".", "ue")))
    os.makedirs(store_dir, exist_ok=True)

    io_cfg = get_config(ue_cfg, "io", OmegaConf.create({}))
    strategy = str(get_config(io_cfg, "strategy", "files")).lower()

    if strategy == "files":
        manifest_path = write_files(
            store_dir=store_dir,
            entries=entries,
            eps=epsilon,
            perturb_type=perturb_type,
            key_spec=key_spec,
        )
    elif strategy == "shards":
        shard_size = int(get_config(io_cfg, "shard_size", 1000))
        manifest_path = write_shards(
            store_dir=store_dir,
            entries=entries,
            eps=epsilon,
            shard_size=shard_size,
            perturb_type=perturb_type,
            key_spec=key_spec,
        )
    else:
        raise ValueError(f"[UE] Unknown ue.io.strategy: {strategy!r}")

    if not os.path.exists(manifest_path):
        raise RuntimeError("[UE] training-free generation failed to write manifest.")

    print(f"[UE] training-free 3D manifest saved to: {manifest_path}")
    return True