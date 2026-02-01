# Z轴层间多样性损失 & Logits 差异损失

## 概述

`noise_slice_frequence_learnable` 算法支持两种可选的正则化损失：

1. **Z轴多样性损失** (`z_diversity_weight`)：增强生成噪声的层间多样性
2. **Logits 差异损失** (`logits_div_enabled`)：最大化干净图像和噪声图像预测之间的差异

## 配置参数

### Z轴多样性损失

```yaml
ue.algorithm.params.z_diversity_weight: 0.1  # 设为 0 禁用（默认值）
```

### Logits 差异损失

```yaml
ue.algorithm.params.logits_div_enabled: true    # 启用 logits 差异损失
ue.algorithm.params.logits_div_mode: fft_l1     # 差异计算模式
ue.algorithm.params.logits_div_weight: 1.0      # 损失权重
ue.algorithm.params.logits_div_temperature: 1.0 # KL 散度模式的温度参数
```

## Logits 差异损失模式

| 模式 | 说明 |
|------|------|
| `l1` | 直接计算 logits 差异的 L1 范数 |
| `l2` | 直接计算 logits 差异的 L2 范数 |
| `fft_l1` | 对 logits 差异做 FFT，然后计算 L1 范数（推荐） |
| `fft_l2` | 对 logits 差异做 FFT，然后计算 L2 范数 |
| `kl_div` | 计算 softmax 分布之间的 KL 散度 |

## 工作原理

### Z轴多样性损失

1. 对噪声张量的每个切片（xy平面）应用 2D FFT
2. 计算每个切片的幅度谱
3. 计算沿 z 轴相邻切片之间的 L2 差异
4. 对所有切片对取平均，然后取负值加入损失（最大化多样性）

### Logits 差异损失

1. 获取干净图像在 surrogate 模型上的预测 logits
2. 获取噪声图像在 surrogate 模型上的预测 logits
3. 计算两者之间的差异（根据选定的模式）
4. 取负值加入损失（最大化差异）

## 命令行示例

### 仅启用 Z-Diversity

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

### 仅启用 Logits Divergence

```bash
python ue_generate.py \
    dataset=flare21 \
    task=flare21_ue \
    task.run_name=noise_slice_freq_learnable_logits \
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
    ue.algorithm.params.logits_div_enabled=true \
    ue.algorithm.params.logits_div_mode=fft_l1 \
    ue.algorithm.params.logits_div_weight=1.0 \
    ue.io.save_from_epoch=50 \
    ue.io.save_every=10 \
    ue.surrogates.s_seg.in_channels=1 \
    ue.surrogates.s_seg.num_classes=5
```

### 同时启用两者

```bash
python ue_generate.py \
    dataset=flare21 \
    task=flare21_ue \
    task.run_name=noise_slice_freq_learnable_all \
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
    ue.algorithm.params.logits_div_enabled=true \
    ue.algorithm.params.logits_div_mode=fft_l1 \
    ue.algorithm.params.logits_div_weight=1.0 \
    ue.io.save_from_epoch=50 \
    ue.io.save_every=10 \
    ue.surrogates.s_seg.in_channels=1 \
    ue.surrogates.s_seg.num_classes=5
```

## 输出指标

| 指标 | 说明 |
|------|------|
| `z_diversity` | 频域中层间 L2 差异的平均值（始终记录） |
| `z_diversity_loss` | 加权后的 z-diversity 损失项（仅当 `z_diversity_weight > 0`） |
| `div_loss` | 加权后的 logits divergence 损失项（仅当 `logits_div_enabled=true`） |
| `logits_diff_l1` | 最终噪声的 logits L1 差异（仅当 `logits_div_enabled=true`） |

## 参数参考

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `z_diversity_weight` | float | 0.0 | z 轴多样性损失的权重，设为 0 禁用 |
| `logits_div_enabled` | bool | false | 是否启用 logits 差异损失 |
| `logits_div_mode` | str | fft_l1 | 差异计算模式：l1, l2, fft_l1, fft_l2, kl_div |
| `logits_div_weight` | float | 1.0 | logits 差异损失的权重 |
| `logits_div_temperature` | float | 1.0 | KL 散度模式的温度参数 |
| `epsilon` | float | 8/255 | 噪声扰动的 L-infinity 上界 |
| `noise_step` | int | 1 | 每个批次的噪声更新步数 |
| `surrogate_step` | int | 10 | 每个噪声步骤的代理模型更新步数 |
| `roi_aware` | bool | true | 仅在 ROI 区域应用噪声 |
| `soft_edge` | bool | true | ROI 掩码使用软边缘 |
| `z_cutoff_low` | float | 0.1 | z 轴高通滤波器的初始截止频率 |
| `xy_cutoff_high` | float | 0.3 | xy 平面低通滤波器的初始截止频率 |
| `cutoff_lr_scale` | float | 1.0 | 截止频率参数的学习率缩放因子 |
