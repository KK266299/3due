# UNet Noise Slice — 逐层切片噪声生成

## 方法概述

`unet_noise_slice` 方法使用 **2D UNet** 逐层（slice-by-slice）为 3D 医学影像体积生成噪声，替代原有的 3D UNet 整体生成方式。

### 核心思想

传统 `unet_noise` 方法使用 3D UNet 一次性为整个体积 `[B, C, D, H, W]` 生成噪声。
本方法改为：

1. 将 3D 体积沿深度轴拆分为 2D 切片：`[B, C, D, H, W]` → `[B*D, C, H, W]`
2. 使用 **2D UNet** 为每个切片独立生成噪声
3. 将切片噪声重组为 3D 体积：`[B*D, C, H, W]` → `[B, C, D, H, W]`
4. 将 3D 噪声加到原始图像上，通过 3D surrogate 模型计算分割损失
5. 反向传播更新 2D UNet 参数

### 优势

| 特性 | 3D UNet (unet_noise) | 2D UNet (unet_noise_slice) |
|------|----------------------|----------------------------|
| 显存占用 | 较高 | 较低（2D卷积参数少） |
| 切片噪声独立性 | 隐式（受3D卷积感受野约束） | 完全独立（每层独立生成） |
| 适配各向异性数据 | 一般 | 好（匹配CT/MRI层间采集特性） |
| 训练速度 | 较慢 | 可能更快（支持切片分批处理） |

### 架构图

```
Input Volume [B, C, D, H, W]
        │
        ▼
   Reshape to 2D slices
   [B*D, C, H, W]
        │
        ▼
   ┌─────────────┐
   │  2D UNet    │  (共享参数，逐切片处理)
   │  + tanh     │
   │  × epsilon  │
   └─────────────┘
        │
        ▼
   Per-slice noise [B*D, C, H, W]
        │
        ▼
   Reshape back to 3D
   [B, C, D, H, W]
        │
        ▼
   Add to original image → Noisy image
        │
        ▼
   ┌─────────────────────┐
   │  3D Surrogate UNet  │  (frozen, 用于计算分割损失)
   └─────────────────────┘
        │
        ▼
   Segmentation Loss (DiceCE)
        │
        ▼
   Backprop → Update 2D Noise UNet
```

## 配置参数

### 算法参数 (`ue.algorithm.params`)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `epsilon` | 0.0313725 (8/255) | L∞ 噪声约束 |
| `noise_step` | 1 | 每个 batch 的 UNet 训练迭代次数 |
| `surrogate_step` | 2 | 每个 epoch 中 surrogate 训练的 batch 数 |
| `slice_batch_size` | 0 | 切片分批大小，0=一次处理所有切片。如遇 OOM 设为 64 或 128 |

### 2D Noise UNet 配置 (`ue.noise_unet`)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `channels` | [16, 32, 64, 128] | UNet 各层通道数 |
| `strides` | [2, 2, 2] | 下采样步长 |
| `num_res_units` | 1 | 残差单元数 |
| `act` | LEAKYRELU | 激活函数 |
| `norm` | INSTANCE | 归一化方式 |
| `optimizer.lr` | 1e-4 | 学习率 |

## 使用方法

### 基础运行命令

```bash
python ue_generate.py \
    dataset=brats19 \
    task.run_name=unet_noise_slice \
    method=unet_noise_slice \
    task=brats19_ue \
    training.epochs=100 \
    ue.key.type=samplewise \
    ue.key.from=field \
    ue.key.field=case_id \
    training.batch_size=8 \
    training.gpu_ids=[1] \
    ue.algorithm.params.surrogate_step=10 \
    ue.io.save_from_epoch=50 \
    ue.io.save_every=10
```

### 使用切片分批（显存不足时）

```bash
python ue_generate.py \
    dataset=brats19 \
    task.run_name=unet_noise_slice_chunked \
    method=unet_noise_slice \
    task=brats19_ue \
    training.epochs=100 \
    ue.key.type=samplewise \
    ue.key.from=field \
    ue.key.field=case_id \
    training.batch_size=4 \
    training.gpu_ids=[0] \
    ue.algorithm.params.surrogate_step=10 \
    ue.algorithm.params.slice_batch_size=64 \
    ue.io.save_from_epoch=50 \
    ue.io.save_every=10
```

### 调整噪声强度

```bash
# 使用 4/255 的较小噪声
python ue_generate.py \
    dataset=brats19 \
    task.run_name=unet_noise_slice_4_255 \
    method=unet_noise_slice \
    task=brats19_ue \
    training.epochs=100 \
    ue.key.type=samplewise \
    ue.key.from=field \
    ue.key.field=case_id \
    training.batch_size=8 \
    training.gpu_ids=[1] \
    ue.algorithm.params.epsilon=0.0156863 \
    ue.algorithm.params.surrogate_step=10 \
    ue.io.save_from_epoch=50 \
    ue.io.save_every=10
```

## 文件结构

```
3due/
├── src/core/ue_algos/
│   └── unet_noise_slice.py          # 算法实现
├── configs/method/
│   └── unet_noise_slice.yaml        # Hydra 配置文件
├── docs/
│   └── unet_noise_slice.md          # 本文档
└── ue_noise_slice.sh                # 运行脚本
```

## 与其他方法的关系

- **unet_noise**: 基础 3D UNet 噪声生成，本方法的参考实现
- **unet_boundary_noise**: 带边界加权的 3D 噪声生成
- **unet_noise_slice_grad**: 层间梯度引导（基于旧版 3D 切片一致性实现）
- **unet_noise_slice_in_out**: 各向异性感知噪声（基于旧版 3D 切片一致性实现）

## 保存路径

噪声保存路径与其他方法保持一致：
```
${task.save_dir}/${task.run_name}/${timestamp}/noise/
```
