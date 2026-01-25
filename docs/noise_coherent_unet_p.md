# UNet-based Samplewise Complex P Optimization for 3D Segmentation UE

## 概述

本方法与 `unet_roi_noise.py` 的流程完全一致，核心区别在于噪声生成方式：

| 方法 | 噪声生成 |
|------|----------|
| `unet_roi_noise` | `delta = tanh(UNet(x)) * ε`（直接输出空域噪声）|
| `noise_coherent` | `delta = tanh(IFFT(P * M)) * ε`，其中 `P = UNet(x)`（输出频域参数，IFFT 到空域）|

## Samplewise 特性

本方法是 **samplewise** 的：

```
每个样本有自己的 P: P = UNet(x)
不同的输入图像 x 产生不同的 P
```

这与 `unet_roi_noise.py` 中每个样本有自己的 `delta = UNet(x)` 完全一致。

## P 的初始化

P 的初始化方式与 `unet_roi_noise.py` 中 delta 的初始化方式完全一致：

```
┌─────────────────────────────────────────────────────────────────┐
│                    P 的初始化说明                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. P 由 UNet 直接从输入图像 x 生成:                             │
│     P = UNet(x)                                                  │
│                                                                  │
│  2. UNet 权重使用 MONAI 默认初始化（Kaiming initialization）     │
│                                                                  │
│  3. 初始时 P 接近零:                                             │
│     - UNet 输出层权重随机初始化                                  │
│     - 输出层偏置初始化为 0                                       │
│     - 因此初始 P 接近零                                          │
│                                                                  │
│  4. 训练过程:                                                    │
│     - 通过最小化 surrogate 损失来更新 UNet 权重                  │
│     - UNet 权重更新 → 每个样本的 P 更新                          │
│     - 这与 unet_roi_noise.py 中 delta = UNet(x) 完全一致         │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## 与 unet_roi_noise.py 的对比

```
┌─────────────────────────────────────────────────────────────────┐
│           unet_roi_noise.py          noise_coherent.py          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  输入: x (图像)                      输入: x (图像)              │
│       │                                   │                      │
│       ▼                                   ▼                      │
│  ┌─────────┐                        ┌─────────┐                  │
│  │  UNet   │                        │ P-UNet  │                  │
│  └────┬────┘                        └────┬────┘                  │
│       │                                   │                      │
│       ▼                                   ▼                      │
│  raw_noise                          P_real, P_imag               │
│       │                                   │                      │
│       │                                   ▼                      │
│       │                            P = P_real + i*P_imag         │
│       │                                   │                      │
│       │                                   ▼                      │
│       │                              P_masked = P * M            │
│       │                                   │                      │
│       │                                   ▼                      │
│       │                            delta_raw = IFFT(P_masked)    │
│       │                                   │                      │
│       ▼                                   ▼                      │
│  delta = tanh(raw_noise) * ε       delta = tanh(delta_raw) * ε  │
│       │                                   │                      │
│       ▼                                   ▼                      │
│  delta = delta * ROI_mask           delta = delta * ROI_mask     │
│       │                                   │                      │
│       ▼                                   ▼                      │
│  存储 delta (samplewise)            存储 delta (samplewise)      │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## 数据流

```
┌─────────────────────────────────────────────────────────────────┐
│                         数据流图                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   Input Image x [B, C, D, H, W]                                  │
│       │                                                          │
│       ▼                                                          │
│   ┌───────────┐                                                  │
│   │  P-UNet   │  输入: [B, C, D, H, W]                           │
│   │           │  输出: [B, 2*C, D, H, W] (P_real, P_imag)        │
│   └─────┬─────┘                                                  │
│         │                                                        │
│         │  每个样本通过 UNet(x) 得到自己独特的 P                  │
│         │  这就是 samplewise 的含义                              │
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
│   │ (frequency      │  M: 固定频域 mask（不参与训练）            │
│   │  constraint)    │  - Z轴: 低频约束 (|f_z| <= z_max)          │
│   └───────┬─────────┘  - XY平面: 带通约束                        │
│           │                                                      │
│           ▼                                                      │
│   ┌─────────────────┐                                            │
│   │     IFFT        │  delta_raw = IFFT(P_masked).real           │
│   │ (to spatial)    │                                            │
│   └───────┬─────────┘                                            │
│           │                                                      │
│           ▼                                                      │
│   ┌─────────────────┐                                            │
│   │ Per-sample      │  delta_norm = delta_raw / max * tanh_scale │
│   │ Normalization   │  tanh_scale = 3.0 使 tanh 输出接近 ±1      │
│   │ (关键步骤!)     │                                            │
│   └───────┬─────────┘                                            │
│           │                                                      │
│           │  为什么需要归一化？                                   │
│           │  - 频域 mask 过滤掉 95%+ 的频率分量                   │
│           │  - IFFT 后能量很小，delta_raw ≈ 0.001                │
│           │  - 如果不归一化：tanh(0.001) ≈ 0.001                 │
│           │  - 结果：delta ≈ 0.001 * 0.0157 ≈ 0.00001 (太小!)   │
│           │  - 归一化后：tanh(3.0) ≈ 0.995                       │
│           │  - 结果：delta ≈ 0.995 * 0.0157 ≈ 0.0156 (正确!)    │
│           │                                                      │
│           ▼                                                      │
│   ┌─────────────────┐                                            │
│   │ tanh + epsilon  │  delta = tanh(delta_norm) * epsilon        │
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
│   └───────┬─────────┘    (与 unet_roi_noise 一致)                │
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
│   │  (minimize)     │  目标: 最小化 surrogate 分割损失            │
│   └───────┬─────────┘  (min-min 过程)                            │
│           │                                                      │
│           ▼                                                      │
│   ┌─────────────────┐                                            │
│   │   Backprop      │  更新 P-UNet 参数                          │
│   │                 │  (与 unet_roi_noise 更新方式一致)          │
│   └───────┬─────────┘                                            │
│           │                                                      │
│           ▼                                                      │
│   ┌─────────────────┐                                            │
│   │  Store delta    │  nb.commit_batch(keys, delta)              │
│   │  to backend     │  (与 noise_slice_frequence 一致)           │
│   └─────────────────┘                                            │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## 为什么需要 Per-sample 归一化？

```
┌─────────────────────────────────────────────────────────────────┐
│                    噪声过小问题分析                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  问题：频域 mask 过滤掉大部分频率分量                            │
│                                                                  │
│  频域 mask 条件：                                                │
│    mask_z = (|f_z| <= 0.125)        # Z 轴低频                  │
│    mask_xy = (0.05 <= |f_xy| <= 0.3) # XY 带通                  │
│    M = mask_z & mask_xy              # 两者交集                  │
│                                                                  │
│  假设图像大小 D=32, H=W=256:                                     │
│    - 总频率点: 32 × 256 × 256 = 2,097,152                       │
│    - mask=1 的点: 约 1-5%                                        │
│    - 95-99% 的 P 被置零                                          │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │ 无归一化:                                                   ││
│  │   P ∈ [-1, 1]                                               ││
│  │   P_masked = P * M (95% 为 0)                               ││
│  │   delta_raw = IFFT(P_masked) ≈ 0.001                        ││
│  │   tanh(0.001) ≈ 0.001                                       ││
│  │   delta = 0.001 * 0.0157 ≈ 0.00001 ❌                       ││
│  └─────────────────────────────────────────────────────────────┘│
│                                                                  │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │ 有归一化:                                                   ││
│  │   delta_raw = IFFT(P_masked) ≈ 0.001                        ││
│  │   delta_norm = delta_raw / 0.001 * 3.0 = 3.0                ││
│  │   tanh(3.0) ≈ 0.995                                         ││
│  │   delta = 0.995 * 0.0157 ≈ 0.0156 ✓                        ││
│  └─────────────────────────────────────────────────────────────┘│
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Loss 计算方法

### 目标函数（Min-Min 过程）

```
min_{θ_UNet} L_seg(surrogate(x + δ), y)
```

其中：
- `θ_UNet`: P-UNet 的参数
- `δ = tanh(IFFT(P * M).real) * ε`
- `P = UNet(x)`（samplewise）
- `L_seg`: DiceCE Loss

### Loss 组成

```python
L_seg = λ_dice * L_dice + λ_ce * L_ce
```

- **Dice Loss**: 衡量预测与标签的重叠度
- **Cross-Entropy Loss**: 逐像素分类损失
- **λ_dice, λ_ce**: 权重系数（默认均为 1.0）

## 软边缘梯度回传

### 软边缘机制

```
soft_edge=false (默认):
  ROI_mask = (label > 0).float()  # 硬边缘，值为 0 或 1
  与 unet_roi_noise.py 一致

soft_edge=true:
  1. ROI_mask = (label > 0).float()
  2. ROI_mask = 膨胀(ROI_mask)
  3. ROI_mask = 高斯模糊(ROI_mask)
  结果: ROI_mask ∈ [0, 1]，边缘平滑过渡
```

### 梯度回传分析

```
δ_masked = δ * ROI_mask

∂L/∂δ = ∂L/∂δ_masked * ROI_mask
```

**软边缘 (soft_edge=true)**:
- `ROI_mask ∈ [0, 1]`
- ROI 中心：`ROI_mask ≈ 1`，梯度完全回传
- ROI 边缘：`0 < ROI_mask < 1`，梯度按比例衰减
- ROI 外部：`ROI_mask ≈ 0`，梯度趋近于 0
- **软边缘区域可以参与梯度更新**

**硬边缘 (soft_edge=false)**:
- `ROI_mask ∈ {0, 1}`
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
# 存储最终空域噪声 delta 到 backend
nb.commit_batch(keys_list, final_delta.detach().cpu())
```

噪声存储格式：
- 存储的是 **delta（空域噪声）**，不是 P（频域参数）
- 每个 sample 独立存储（samplewise）
- 通过 `noise_backend` 统一管理
- 支持 `int8` 量化存储以节省空间

## 使用方法

```bash
python ue_generate.py \
    dataset=flare21 \
    task=flare21_ue \
    task.run_name=noise_coherent_4_255_hard \
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
    ue.algorithm.params.roi_aware=true \
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
| `roi_aware` | true | 是否启用 ROI mask |
| `soft_edge` | false | 软边缘开关: false=硬边缘(0/1), true=软边缘 |
| `dilate_kernel_size` | 3 | ROI 膨胀卷积核大小（仅 soft_edge=true） |
| `gaussian_sigma` | 2.0 | 高斯模糊 sigma（仅 soft_edge=true） |

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
    lr: 1.0e-4             # P-UNet 学习率
    weight_decay: 1.0e-5   # 权重衰减
    betas: [0.9, 0.999]    # Adam betas
```

输入：原始图像 `[B, C, D, H, W]`
输出：P 的实部和虚部 `[B, 2*C, D, H, W]`

## 与其他方法的对比

| 特性 | unet_roi_noise | noise_coherent | noise_slice_frequence |
|------|----------------|----------------|----------------------|
| 噪声生成 | UNet(x) → delta | UNet(x) → P → IFFT → delta | UNet(x) → delta + 频域滤波 |
| 频域约束 | 无 | 有（通过 P * M） | 有（后处理滤波） |
| P 初始化 | N/A | UNet 权重初始化 | N/A |
| Samplewise | 是 | 是 | 是 |
| 软边缘开关 | 无 | 有 | 有 |
| 存储格式 | delta | delta | delta |
