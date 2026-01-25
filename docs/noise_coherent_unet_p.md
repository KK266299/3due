# UNet-based Complex P Optimization for 3D Segmentation UE

## 概述

本方法通过 UNet 直接生成频域参数 P（复数的实部和虚部），然后通过 IFFT 转换到空域生成扰动噪声。目标是**最小化 surrogate 在扰动图像上的分割损失（min-min 过程）**。

## 核心思想

1. **UNet 直接输出 P**：UNet 接收原始图像，输出 P 的实部和虚部（直接更新，非增量）
2. **频域约束**：通过频域 mask 约束噪声的频率特性
3. **tanh 限制**：在 IFFT 后对 delta 进行 tanh 限制，再 clamp 到 [-ε, ε]
4. **软边缘开关**：可配置开启/关闭软边缘 ROI mask

## 数据流

```
┌─────────────────────────────────────────────────────────────────┐
│                         数据流图                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   Input Image x                                                  │
│       │                                                          │
│       ▼                                                          │
│   ┌───────────┐                                                  │
│   │  P-UNet   │  输入: [B, C, D, H, W]                           │
│   │           │  输出: [B, 2*C, D, H, W] (P_real, P_imag)        │
│   └─────┬─────┘                                                  │
│         │                                                        │
│         ▼                                                        │
│   ┌─────────────────┐                                            │
│   │ Split Channels  │  P_real: [B, C, D, H, W]                   │
│   │                 │  P_imag: [B, C, D, H, W]                   │
│   └───────┬─────────┘                                            │
│           │                                                      │
│           ▼                                                      │
│   ┌─────────────────┐                                            │
│   │ Complex P =     │  P = P_real + i * P_imag                   │
│   │ P_real + i*P_imag│                                           │
│   └───────┬─────────┘                                            │
│           │                                                      │
│           ▼                                                      │
│   ┌─────────────────┐                                            │
│   │ Spectral Mask   │  P_masked = P * M                          │
│   │ (frequency      │  M: 固定频域 mask                          │
│   │  constraint)    │  - Z轴: 低频约束                            │
│   └───────┬─────────┘  - XY平面: 带通约束                         │
│           │                                                      │
│           ▼                                                      │
│   ┌─────────────────┐                                            │
│   │     IFFT        │  delta_raw = IFFT(P_masked).real           │
│   │ (to spatial)    │                                            │
│   └───────┬─────────┘                                            │
│           │                                                      │
│           ▼                                                      │
│   ┌─────────────────┐                                            │
│   │ tanh + epsilon  │  delta = tanh(delta_raw) * epsilon         │
│   │                 │                                            │
│   └───────┬─────────┘                                            │
│           │                                                      │
│           ▼                                                      │
│   ┌─────────────────┐                                            │
│   │     clamp       │  delta = clamp(delta, -eps, eps)           │
│   │                 │                                            │
│   └───────┬─────────┘                                            │
│           │                                                      │
│           ▼                                                      │
│   ┌─────────────────┐    soft_edge=true: 高斯平滑边缘            │
│   │ ROI Gate        │    soft_edge=false: 硬边缘 (0/1)           │
│   │ (optional)      │    delta = delta * ROI_mask                │
│   └───────┬─────────┘                                            │
│           │                                                      │
│           ▼                                                      │
│   ┌─────────────────┐                                            │
│   │ Perturbed Image │  x_perturbed = clip(x + delta, 0, 1)       │
│   │                 │                                            │
│   └───────┬─────────┘                                            │
│           │                                                      │
│           ▼                                                      │
│   ┌─────────────────┐                                            │
│   │   Surrogate     │  logits = surrogate(x_perturbed)           │
│   │   (frozen)      │                                            │
│   └───────┬─────────┘                                            │
│           │                                                      │
│           ▼                                                      │
│   ┌─────────────────┐                                            │
│   │  DiceCE Loss    │  loss = DiceCELoss(logits, label)          │
│   │  (minimize)     │                                            │
│   └───────┬─────────┘                                            │
│           │                                                      │
│           ▼                                                      │
│   ┌─────────────────┐                                            │
│   │   Backprop      │  更新 P-UNet 参数                          │
│   │                 │                                            │
│   └─────────────────┘                                            │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Loss 计算方法

### 目标函数

```
min_{θ_UNet} L_seg(surrogate(x + δ), y)
```

其中：
- `θ_UNet`: P-UNet 的参数
- `δ = tanh(IFFT(P * M).real) * ε`，`P = UNet(x)`
- `L_seg`: DiceCE Loss（Dice Loss + Cross-Entropy Loss）

### Loss 组成

```python
L_seg = λ_dice * L_dice + λ_ce * L_ce
```

- **Dice Loss**: 衡量预测与标签的重叠度
- **Cross-Entropy Loss**: 逐像素分类损失
- **λ_dice, λ_ce**: 权重系数（默认均为 1.0）

### Min-Min 过程

1. **N-step（噪声更新）**：冻结 surrogate，更新 P-UNet 以最小化 surrogate 的分割损失
2. **S-step（surrogate 更新）**：冻结噪声，更新 surrogate 以最小化在扰动图像上的分割损失

## 软边缘梯度回传

### 软边缘机制

当 `soft_edge=true` 时：
1. **二值化**：ROI_mask = (label > 0)
2. **膨胀**：通过 max_pool3d 扩展 ROI 区域
3. **高斯模糊**：平滑边缘，使 ROI_mask 值在 [0, 1] 连续

### 梯度回传分析

```
δ_masked = δ * ROI_mask

∂L/∂δ = ∂L/∂δ_masked * ROI_mask
```

**关键点**：
- **soft_edge=true**：`ROI_mask ∈ [0, 1]`，边缘区域梯度平滑衰减
  - ROI 中心：`ROI_mask ≈ 1`，梯度完全回传
  - ROI 边缘：`0 < ROI_mask < 1`，梯度按比例衰减
  - ROI 外部：`ROI_mask ≈ 0`，梯度趋近于 0

- **soft_edge=false**：`ROI_mask ∈ {0, 1}`，硬边缘
  - ROI 内部：梯度完全回传
  - ROI 外部：梯度为 0（截断）

### 梯度流图

```
┌────────────────────────────────────────────────────────┐
│                  梯度回传路径                           │
├────────────────────────────────────────────────────────┤
│                                                        │
│   Loss                                                 │
│     │                                                  │
│     ▼                                                  │
│   ∂L/∂logits                                          │
│     │                                                  │
│     ▼ (surrogate frozen, no grad)                     │
│   ∂L/∂x_perturbed                                     │
│     │                                                  │
│     ▼                                                  │
│   ∂L/∂δ_masked = ∂L/∂x_perturbed                      │
│     │                                                  │
│     ▼                                                  │
│   ∂L/∂δ = ∂L/∂δ_masked * ROI_mask  ← 软边缘影响        │
│     │                                                  │
│     ▼                                                  │
│   ∂L/∂δ_raw (through tanh * epsilon)                  │
│     │                                                  │
│     ▼                                                  │
│   ∂L/∂P_masked (through IFFT)                         │
│     │                                                  │
│     ▼                                                  │
│   ∂L/∂P (through spectral mask)                       │
│     │                                                  │
│     ▼                                                  │
│   ∂L/∂θ_UNet (通过 P_real, P_imag 的生成)              │
│                                                        │
└────────────────────────────────────────────────────────┘
```

## 噪声存储

与 `noise_slice_frequence` 方法一致：

```python
# 存储最终噪声到 backend
nb.commit_batch(keys_list, final_delta.detach().cpu())
```

噪声存储格式：
- 每个 sample 独立存储
- 通过 `noise_backend` 统一管理
- 支持 `int8` 量化存储以节省空间

## 使用方法

```bash
python ue_generate.py \
    dataset=flare21 \
    task=flare21_ue \
    task.run_name=noise_coherent_unet_p \
    method=noise_coherent \
    training.epochs=100 \
    training.batch_size=8 \
    training.gpu_ids=[0] \
    ue.key.type=samplewise \
    ue.key.from=field \
    ue.key.field=case_id \
    ue.algorithm.params.epsilon=0.0156863 \
    ue.algorithm.params.noise_step=1 \
    ue.algorithm.params.surrogate_step=10 \
    ue.algorithm.params.soft_edge=false \
    ue.io.save_from_epoch=50 \
    ue.io.save_every=10 \
    ue.surrogates.s_seg.in_channels=1 \
    ue.surrogates.s_seg.num_classes=5
```

### 关键参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `epsilon` | 8/255 | 扰动范围 L∞ bound |
| `noise_step` | 1 | 每 batch 更新 P-UNet 的迭代次数 |
| `surrogate_step` | 10 | 每 epoch 更新 surrogate 的 batch 数 |
| `soft_edge` | false | 是否启用软边缘 ROI mask |
| `dilate_kernel_size` | 3 | ROI 膨胀卷积核大小 |
| `gaussian_sigma` | 2.0 | 高斯模糊 sigma |

### 频域 Mask 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `z_max` | 0.125 | Z 轴低频截止 |
| `xy_min` | 0.05 | XY 平面带通下限 |
| `xy_max` | 0.3 | XY 平面带通上限 |

## P-UNet 架构

```yaml
p_unet:
  channels: [16, 32, 64, 128]
  strides: [2, 2, 2]
  num_res_units: 1
  act: LEAKYRELU
  norm: INSTANCE
  dropout: 0.0
  optimizer:
    lr: 1.0e-4
    weight_decay: 1.0e-5
    betas: [0.9, 0.999]
```

输入：原始图像 `[B, C, D, H, W]`
输出：P 的实部和虚部 `[B, 2*C, D, H, W]`

## 与其他方法的对比

| 特性 | noise_coherent (原) | noise_coherent (UNet-P) | noise_slice_frequence |
|------|---------------------|-------------------------|----------------------|
| P 更新方式 | 直接优化 P 参数 | UNet 生成 P | UNet 生成空域噪声 |
| 频域约束 | 有 | 有 | 有 |
| 软边缘开关 | 有 | 有 | 有 |
| Sample-wise | 是 | 否（共享 UNet） | 否（共享 UNet） |
| 参数量 | O(C×D×H×W×N_samples) | O(UNet) | O(UNet) |
