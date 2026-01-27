# PUE (Provably Unlearnable Examples) 实现文档

## 概述

本文档描述了基于 **Provably Unlearnable Data Examples** (NeurIPS 2022) 方法的噪声生成实现，该实现适配于 BraTS19 3D 医学图像分割任务。

**论文参考**: [Provably Unlearnable Data Examples](https://arxiv.org/abs/2206.10278)

**参考代码库**: [NeuralSec/certified-data-learnability](https://github.com/NeuralSec/certified-data-learnability)

---

## 方法原理

### 1. 核心思想

PUE (Provably Unlearnable Examples) 的核心创新是 **Random Weight Perturbation (RWP)**：

- 在训练代理模型和优化噪声时，对模型权重添加**临时高斯噪声** $\mathcal{N}(0, \sigma^2)$
- 通过多次采样权重噪声，取平均梯度来更新噪声
- 这使得生成的噪声对**模型权重扰动具有鲁棒性**，从而提供理论上的不可学习性保证

### 2. 算法流程

```
输入: 干净数据集 D, 代理模型 θ, 噪声预算 ε
输出: 扰动噪声 δ

for epoch = 1 to T:
    # S-step: 更新代理模型（带 RWP）
    for batch in D:
        累积梯度 = 0
        for u = 1 to U_train:
            ε_w ~ N(0, σ²)           # 采样权重噪声
            θ' = θ + ε_w              # 临时添加噪声
            loss = L(θ', x + δ, y)    # 前向传播
            累积梯度 += ∇_θ loss       # 累积梯度
            θ' = θ                    # 恢复原始权重
        θ = θ - lr * (累积梯度 / U_train)  # 更新权重

    # N-step: 更新噪声（PGD + RWP）
    for batch in D:
        for step = 1 to noise_step:
            累积梯度 = 0
            for u = 1 to U_noise:
                ε_w ~ N(0, σ²)        # 采样权重噪声
                θ' = θ + ε_w           # 临时添加噪声
                loss = L(θ', x + δ, y)
                累积梯度 += ∇_δ loss    # 累积对噪声的梯度
                θ' = θ                 # 恢复原始权重
            g = 累积梯度 / U_noise
            δ = δ - step_size * sign(g)  # PGD 更新（min-min）
            δ = clamp(δ, -ε, ε)          # L∞ 投影
```

### 3. 与 Min-Min 的主要区别

| 特性 | Min-Min | PUE |
|------|---------|-----|
| 权重扰动 | 无 | RWP (随机权重扰动) |
| 采样次数 | 1 | U_train / U_noise |
| 理论保证 | 无 | 可证明不可学习性 |
| 鲁棒性 | 对模型初始化敏感 | 对初始化更鲁棒 |
| 计算开销 | 低 | 较高（U倍） |

---

## 代码实现

### 修改的文件

| 文件 | 路径 | 说明 |
|------|------|------|
| **pue.py** | `src/core/ue_algos/pue.py` | PUE 算法核心实现 |
| **pue.yaml** | `configs/method/pue.yaml` | 配置文件 |
| **ue_pue.sh** | `ue_pue.sh` | 训练脚本 |

### 1. pue.py 核心代码

**文件路径**: `src/core/ue_algos/pue.py`

```python
@register_plugin("pue")
class PUE:
    """
    PUE (Provably Unlearnable Examples) for 3D segmentation.

    核心方法:
    - _add_weight_noise_(): 对模型权重添加临时高斯噪声
    - _remove_weight_noise_(): 恢复原始权重
    - surrogate_step_batch(): 使用 RWP 训练代理模型
    - noise_step_batch(): 使用 PGD + RWP 优化噪声
    """

    @staticmethod
    def _add_weight_noise_(model, sigma):
        """添加权重噪声 N(0, σ²)"""
        noises = []
        with torch.no_grad():
            for p in model.parameters():
                eps = torch.randn_like(p) * sigma
                p.add_(eps)
                noises.append(eps)
        return noises

    @staticmethod
    def _remove_weight_noise_(model, noises):
        """移除权重噪声，恢复原始参数"""
        with torch.no_grad():
            for p, eps in zip(model.parameters(), noises):
                p.sub_(eps)
```

### 2. pue.yaml 配置说明

**文件路径**: `configs/method/pue.yaml`

```yaml
ue:
  # RWP 配置
  rwp:
    enabled: true     # 启用 RWP
    sigma: 0.05       # 权重噪声标准差
    U_train: 4        # S-step 采样次数
    U_noise: 8        # N-step 采样次数

  # PGD 配置
  algorithm:
    kind: training_based
    name: pue
    params:
      epsilon: 0.0313725      # 8/255 (L∞ 预算)
      noise_step: 10          # PGD 内层步数
      step_size: 0.0039215    # 1/255 (步长)
      surrogate_step: 20      # 每 epoch S-step 批次数

  # 代理模型配置
  surrogates:
    s_seg:
      backbone: unet
      in_channels: 4
      num_classes: 4
      spatial_dims: 3
      channels: [32, 64, 128, 256, 512]
      strides: [2, 2, 2, 2]
      ...

  # 噪声存储配置
  io:
    enabled: true
    include_manifest: true
    strategy: files           # 与 min_min 一致
    dtype: int8              # 量化为 int8
    save_from_epoch: 30
    save_every: 10
```

---

## 使用示例

### 1. 基础用法

```bash
# 使用默认配置运行 PUE
python ue_generate.py \
    dataset=brats19 \
    task.run_name=pue_noise \
    method=pue \
    task=brats19_ue \
    training.epochs=100 \
    ue.key.type=samplewise \
    ue.key.from=field \
    ue.key.field=case_id \
    training.batch_size=8 \
    training.gpu_ids=[0]
```

### 2. 调整 RWP 参数

```bash
# 增大权重噪声强度
python ue_generate.py \
    dataset=brats19 \
    method=pue \
    task=brats19_ue \
    training.epochs=100 \
    ue.rwp.sigma=0.1 \          # 增大 σ
    ue.rwp.U_train=8 \          # 增加采样次数
    ue.rwp.U_noise=16 \
    training.batch_size=4       # 显存不足时减小 batch
```

### 3. 禁用 RWP（退化为 Min-Min）

```bash
# 禁用 RWP，等价于 min_min
python ue_generate.py \
    dataset=brats19 \
    method=pue \
    task=brats19_ue \
    ue.rwp.enabled=false
```

### 4. 完整训练脚本 (ue_pue.sh)

```bash
#!/bin/bash
python ue_generate.py \
    dataset=brats19 \
    task.run_name=pue_noise_rwp \
    method=pue \
    task=brats19_ue \
    training.epochs=100 \
    ue.key.type=samplewise \
    ue.key.from=field \
    ue.key.field=case_id \
    ue.algorithm.params.step_size=5e-3 \
    training.batch_size=8 \
    training.gpu_ids=[0] \
    ue.algorithm.params.surrogate_step=10 \
    ue.rwp.enabled=true \
    ue.rwp.sigma=0.05 \
    ue.rwp.U_train=4 \
    ue.rwp.U_noise=8 \
    ue.io.save_from_epoch=0 \
    ue.io.save_every=10
```

---

## 输出格式

### 噪声存储结构

```
outputs/brats19_ue/pue_noise_rwp/<TIMESTAMP>/noise/
├── epoch_0000/
│   ├── manifest.json          # 噪声索引
│   ├── BraTS19_CBICA_AAB_1.pt # 样本噪声
│   ├── BraTS19_CBICA_AAG_1.pt
│   └── ...
├── epoch_0010/
│   ├── manifest.json
│   └── ...
└── epoch_0099/
    ├── manifest.json
    └── ...
```

### manifest.json 格式

```json
{
    "version": 1,
    "strategy": "files",
    "dtype": "int8",
    "scale": 0.0313725,
    "perturb_type": "samplewise",
    "keys": ["BraTS19_CBICA_AAB_1", "BraTS19_CBICA_AAG_1", ...],
    "entries": [
        {"key": "BraTS19_CBICA_AAB_1", "file": "BraTS19_CBICA_AAB_1.pt"},
        {"key": "BraTS19_CBICA_AAG_1", "file": "BraTS19_CBICA_AAG_1.pt"},
        ...
    ]
}
```

---

## 参数调优建议

### 1. RWP 参数

| 参数 | 默认值 | 建议范围 | 说明 |
|------|--------|----------|------|
| `sigma` | 0.05 | 0.01 ~ 0.2 | 权重噪声标准差，越大鲁棒性越强但收敛慢 |
| `U_train` | 4 | 2 ~ 8 | S-step 采样次数，增大提升稳定性 |
| `U_noise` | 8 | 4 ~ 16 | N-step 采样次数，增大提升噪声质量 |

### 2. PGD 参数

| 参数 | 默认值 | 建议范围 | 说明 |
|------|--------|----------|------|
| `epsilon` | 8/255 | 4/255 ~ 16/255 | L∞ 噪声预算 |
| `step_size` | 1/255 | 0.5/255 ~ 5e-3 | PGD 步长 |
| `noise_step` | 10 | 5 ~ 20 | 内层 PGD 迭代次数 |
| `surrogate_step` | 20 | 10 ~ 50 | 每 epoch S-step 批次数 |

### 3. 显存优化

由于 RWP 需要多次前向/反向传播，显存消耗较大：

```bash
# 减小 batch_size
training.batch_size=4

# 减少采样次数（牺牲一定鲁棒性）
ue.rwp.U_train=2
ue.rwp.U_noise=4

# 使用梯度累积（需修改代码）
```

---

## 与原 Min-Min 的兼容性

PUE 实现完全兼容原有的噪声存储和加载机制：

1. **噪声格式一致**: 使用相同的 `files` 策略和 `int8` 量化
2. **manifest 兼容**: 生成相同格式的 `manifest.json`
3. **Poisoning 流程**: 可直接用于 victim 训练

```bash
# 使用 PUE 生成的噪声训练 victim
python main.py \
    method=poison_files \
    dataset=brats19 \
    task=brats19_seg \
    training.data.poison.source.type=manifest \
    training.data.poison.source.manifest_path=<path_to_pue_manifest>/manifest.json
```

---

## 总结

PUE 方法通过 Random Weight Perturbation 提供了比 Min-Min 更强的理论保证：

1. **优势**:
   - 对模型初始化更加鲁棒
   - 提供可证明的不可学习性
   - 生成的噪声在不同模型架构间迁移性更好

2. **代价**:
   - 计算开销增加约 U_train * U_noise 倍
   - 需要更多显存

3. **适用场景**:
   - 需要更强保护的高敏感数据
   - 对噪声鲁棒性有较高要求的场景
   - 研究和验证不可学习性理论
