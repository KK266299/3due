# Z轴层间多样性损失

## 概述

`noise_slice_frequence_learnable` 算法现已支持可选的 **Z轴多样性损失**，用于增强生成噪声的层间多样性。该损失通过最大化相邻切片在频域中的 L2 差异来实现。

## 配置参数

添加以下参数来启用 z 轴多样性正则化：

```yaml
ue.algorithm.params.z_diversity_weight: 0.1  # 设为 0 禁用（默认值）
```

- `z_diversity_weight = 0.0`（默认）：禁用，不计算 z-diversity 损失
- `z_diversity_weight > 0`：启用，损失函数会鼓励层间变化

## 工作原理

z-diversity 损失的计算步骤：

1. 对噪声张量的每个切片（xy平面）应用 2D FFT
2. 计算每个切片的幅度谱
3. 计算沿 z 轴相邻切片之间的 L2 差异
4. 对所有切片对、通道和批次取平均

由于我们希望**最大化**多样性，因此将该值的**负值**加入总损失：

```
total_loss = seg_loss + z_diversity_weight * (-z_diversity)
```

## 命令行示例

### 基础用法（z_diversity 默认禁用）

```bash
python ue_generate.py \
    dataset=flare21 \
    task=flare21_ue \
    task.run_name=noise_slice_freq_learnable \
    method=noise_slice_frequence_learnable \
    training.epochs=100 \
    training.batch_size=8 \
    training.gpu_ids=[0] \
    ue.key.type=samplewise \
    ue.key.from=field \
    ue.key.field=case_id \
    ue.algorithm.params.epsilon=0.0156863 \
    ue.algorithm.params.noise_step=1 \
    ue.algorithm.params.surrogate_step=10 \
    ue.algorithm.params.roi_aware=true \
    ue.algorithm.params.soft_edge=false \
    ue.io.save_from_epoch=50 \
    ue.io.save_every=10 \
    ue.surrogates.s_seg.in_channels=1 \
    ue.surrogates.s_seg.num_classes=5
```

### 启用 Z-Diversity 损失

```bash
python ue_generate.py \
    dataset=flare21 \
    task=flare21_ue \
    task.run_name=noise_slice_freq_learnable_zdiv \
    method=noise_slice_frequence_learnable \
    training.epochs=100 \
    training.batch_size=8 \
    training.gpu_ids=[0] \
    ue.key.type=samplewise \
    ue.key.from=field \
    ue.key.field=case_id \
    ue.algorithm.params.epsilon=0.0156863 \
    ue.algorithm.params.noise_step=1 \
    ue.algorithm.params.surrogate_step=10 \
    ue.algorithm.params.roi_aware=true \
    ue.algorithm.params.soft_edge=false \
    ue.algorithm.params.z_diversity_weight=0.1 \
    ue.io.save_from_epoch=50 \
    ue.io.save_every=10 \
    ue.surrogates.s_seg.in_channels=1 \
    ue.surrogates.s_seg.num_classes=5
```

## 输出指标

启用 z-diversity 后，会记录以下额外指标：

| 指标 | 说明 |
|------|------|
| `z_diversity` | 频域中层间 L2 差异的平均值（始终记录） |
| `z_diversity_loss` | 加权后的 z-diversity 损失项（仅当 `z_diversity_weight > 0` 时记录） |

## 参数参考

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `z_diversity_weight` | float | 0.0 | z 轴多样性损失的权重，设为 0 禁用 |
| `epsilon` | float | 8/255 | 噪声扰动的 L-infinity 上界 |
| `noise_step` | int | 1 | 每个批次的噪声更新步数 |
| `surrogate_step` | int | 10 | 每个噪声步骤的代理模型更新步数 |
| `roi_aware` | bool | true | 仅在 ROI 区域应用噪声 |
| `soft_edge` | bool | true | ROI 掩码使用软边缘 |
| `z_cutoff_low` | float | 0.1 | z 轴高通滤波器的初始截止频率 |
| `xy_cutoff_high` | float | 0.3 | xy 平面低通滤波器的初始截止频率 |
| `cutoff_lr_scale` | float | 1.0 | 截止频率参数的学习率缩放因子 |
