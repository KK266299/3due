# Noise Coherent: 技术说明文档

## 概述

Noise Coherent 是一种基于 UNet 的频域扰动生成方法，其核心思想是：
1. **使用 UNet 生成 P 的实部和虚部增量**
2. **维护可学习的 P 参数，通过 UNet 输出进行更新**
3. **应用频域 mask，然后 IFFT 到空域得到扰动**
4. **支持 ROI 软边缘 mask，梯度可以回传更新 UNet 和 P**

---

## 1. P 参数更新机制

### 1.1 初始化
```python
# P_real 和 P_imag 初始化（可选）
P_real = nn.Parameter(torch.randn(C, D, H, W) * init_scale)  # 实部
P_imag = nn.Parameter(torch.randn(C, D, H, W) * init_scale)  # 虚部
```

配置参数：
- `p_init.enabled`: 是否初始化 P（如果为 false，P 从零开始）
- `p_init.init_scale`: 初始化尺度（默认 0.01）
- `p_init.learnable`: P 是否作为可学习参数（默认 true）

### 1.2 UNet 生成增量
```python
# UNet 输入图像 x，输出 P 的实部和虚部增量
delta_P_real, delta_P_imag = UNet(x)  # [B, C, D, H, W]
```

### 1.3 更新策略

#### Additive 模式（默认）
```python
P_real_new = P_real + update_scale * delta_P_real
P_imag_new = P_imag + update_scale * delta_P_imag
```
- 保留历史信息，逐步调整 P
- `update_scale` 控制更新步长（默认 0.1）

#### Replace 模式
```python
P_real_new = delta_P_real
P_imag_new = delta_P_imag
```
- 完全由 UNet 输出决定 P
- 适合图像依赖的扰动生成

---

## 2. Loss 计算方法

### 2.1 完整流程

```
输入: x (图像), y (标签)

步骤 1: UNet 生成 P 增量
    delta_P_real, delta_P_imag = UNet(x)

步骤 2: 更新 P
    P_real_new = P_real + scale * delta_P_real
    P_imag_new = P_imag + scale * delta_P_imag

步骤 3: 构建复数 P
    P_complex = P_real_new + i * P_imag_new

步骤 4: 应用频域 mask
    P_masked = P_complex * M

    其中 M 是固定的频域 mask:
    - Z 轴（层间）: 低频 (|f_z| <= z_max)
    - XY 平面（层内）: 带通 (xy_min <= |f_xy| <= xy_max)

步骤 5: IFFT 到空域
    delta_spatial = IFFT(P_masked)  # 实部

步骤 6: 应用 ROI 软边缘 mask
    ROI_mask = GaussianSmooth(label > 0)  # [0, 1] 范围
    delta = delta_spatial * ROI_mask

步骤 7: Clip 到 epsilon
    delta = clamp(delta, -eps, eps)

步骤 8: 生成扰动图像
    x_perturbed = clamp(x + delta, 0, 1)

步骤 9: 前向 surrogate（冻结）
    logits = surrogate(normalize(x_perturbed))

步骤 10: 计算分割损失
    loss = DiceCELoss(logits, y)

    其中 DiceCELoss = lambda_dice * Dice + lambda_ce * CrossEntropy

步骤 11: 反向传播
    loss.backward()  # 梯度回传到 UNet 和 P

步骤 12: 优化器更新
    optimizer.step()  # 更新 UNet 和 P 参数
```

### 2.2 Loss 函数详解

**DiceCELoss** 结合了 Dice Loss 和 Cross-Entropy Loss：

```python
DiceCELoss = lambda_dice * DiceLoss + lambda_ce * CrossEntropyLoss

DiceLoss = 1 - (2 * sum(pred * target) + smooth) / (sum(pred) + sum(target) + smooth)

CrossEntropyLoss = -sum(target * log(softmax(pred)))
```

**目标**：最大化 surrogate 的分割损失，使其在扰动图像上的预测更差。

---

## 3. 软边缘梯度回传机制

### 3.1 ROI 软边缘生成

```python
# 1. 创建二值 mask
mask = (label > 0).float()  # [B, D, H, W]

# 2. 可选：形态学膨胀
mask = MaxPool3D(mask, kernel_size=3)

# 3. 高斯平滑（关键步骤）
kernel = GaussianKernel3D(sigma=2.0)
mask_smooth = Conv3D(mask, kernel)  # [B, 1, D, H, W]

# 4. 归一化到 [0, 1]
ROI_mask = mask_smooth / max(mask_smooth)
```

### 3.2 软边缘的作用

**硬边缘**（无高斯平滑）：
```
ROI 内部: mask = 1.0
ROI 外部: mask = 0.0
边界: 突变（梯度不连续）
```

**软边缘**（有高斯平滑）：
```
ROI 内部: mask ≈ 1.0
ROI 外部: mask ≈ 0.0
边界: 平滑过渡（mask ∈ (0, 1)）
```

### 3.3 梯度回传机制

#### 前向传播
```python
# 扰动被 ROI mask 缩放
delta_masked = delta * ROI_mask

# 扰动图像
x_perturbed = x + delta_masked

# Surrogate 前向
logits = surrogate(x_perturbed)

# 损失
loss = DiceCELoss(logits, y)
```

#### 反向传播

根据链式法则：
```
∂loss/∂delta = ∂loss/∂x_perturbed * ∂x_perturbed/∂delta_masked * ∂delta_masked/∂delta

= ∂loss/∂x_perturbed * 1 * ROI_mask

= ROI_mask * ∂loss/∂x_perturbed
```

**关键点**：
- **软边缘区域**（0 < ROI_mask < 1）的梯度会被**缩放**（衰减），但**不会完全消失**
- **ROI 内部**（ROI_mask ≈ 1）的梯度**几乎不受影响**
- **ROI 外部**（ROI_mask ≈ 0）的梯度**接近零**

### 3.4 软边缘的优势

| 对比项 | 硬边缘 | 软边缘 |
|--------|--------|--------|
| 边界梯度 | 不连续，可能梯度消失 | 平滑连续，梯度平滑衰减 |
| 视觉效果 | 可能出现明显边界伪影 | 自然过渡，无明显伪影 |
| 学习能力 | 边界区域无法学习 | 边界区域可以学习（衰减） |
| 训练稳定性 | 较低 | 较高 |

### 3.5 参数调优

**gaussian_sigma**（高斯平滑标准差）：
- **小值**（如 0.5-1.0）：边缘较硬，过渡区域窄
- **中值**（如 2.0-3.0，推荐）：边缘适中，平衡效果和梯度
- **大值**（如 5.0+）：边缘很软，过渡区域宽，但可能过度平滑

**dilate_kernel_size**（膨胀卷积核）：
- **0**：无膨胀，ROI 区域仅限于标签
- **3-5**（推荐）：轻微膨胀，覆盖边界附近区域
- **7+**：大幅膨胀，扩大 ROI 范围

---

## 4. 完整梯度流

```
Loss (DiceCELoss)
  ↓ ∂loss/∂logits
Logits (Surrogate output)
  ↓ ∂logits/∂x_perturbed  [Surrogate 冻结，梯度不更新 surrogate]
x_perturbed = clip(x + delta_masked)
  ↓ ∂x_perturbed/∂delta_masked
delta_masked = delta * ROI_mask
  ↓ ∂delta_masked/∂delta = ROI_mask  [软边缘梯度缩放]
delta = IFFT(P_masked).real
  ↓ ∂delta/∂P_masked  [IFFT 可微]
P_masked = P_complex * M
  ↓ ∂P_masked/∂P_complex = M
P_complex = P_real_new + i * P_imag_new
  ↓ ∂P_complex/∂P_real_new, ∂P_complex/∂P_imag_new
P_real_new = P_real + scale * delta_P_real
P_imag_new = P_imag + scale * delta_P_imag
  ↓ ∂P_new/∂delta_P_real = scale, ∂P_new/∂delta_P_imag = scale
  ↓ ∂P_new/∂P_real = 1, ∂P_new/∂P_imag = 1
delta_P_real, delta_P_imag = UNet(x)
  ↓ ∂delta_P/∂UNet_params
UNet 参数更新
P_real, P_imag 参数更新
```

---

## 5. 配置示例

```yaml
ue:
  algorithm:
    name: noise_coherent
    params:
      epsilon: 0.01568627  # 4/255

      # UNet 配置
      unet:
        in_channels: 4
        out_channels: 8  # 实部 + 虚部
        channels: [32, 64, 128, 256]
        strides: [2, 2, 2]

      # P 初始化
      p_init:
        enabled: true
        init_scale: 0.01
        learnable: true

      # P 更新策略
      p_update:
        mode: "additive"  # 或 "replace"
        update_scale: 0.1

      # 频域 mask
      spectral_mask:
        z_max: 0.125
        xy_min: 0.05
        xy_max: 0.3

      # ROI 软边缘
      roi_gate:
        enabled: true
        dilate_kernel_size: 3
        gaussian_sigma: 2.0  # 控制边缘软化程度

      # 优化器
      optimizer:
        lr: 1.0e-3
        weight_decay: 1.0e-5

      # 训练步数
      noise_step: 5
```

---

## 6. 使用方法

### 6.1 训练
```bash
python ue_generate.py \
  dataset=brats19 \
  task=brats19_ue \
  method=noise_coherent \
  task.run_name=noise_coherent_exp \
  training.epochs=100 \
  training.batch_size=4 \
  training.gpu_ids=[0]
```

### 6.2 调试软边缘效果

修改 `gaussian_sigma` 来观察不同的边缘软化效果：

```bash
# 硬边缘（不推荐）
ue.algorithm.params.roi_gate.gaussian_sigma=0.5

# 中等软边缘（推荐）
ue.algorithm.params.roi_gate.gaussian_sigma=2.0

# 很软的边缘
ue.algorithm.params.roi_gate.gaussian_sigma=5.0
```

---

## 7. 总结

### 核心特性
1. **UNet 生成 P 增量**：图像依赖的频域参数生成
2. **P 参数可学习**：结合全局和局部信息
3. **软边缘梯度回传**：边界区域平滑衰减，可参与学习
4. **频域约束**：通过固定 mask 保证频域一致性

### 软边缘的关键作用
- ✅ **梯度可以回传**：软边缘区域的梯度被缩放但不消失
- ✅ **视觉效果自然**：避免硬边界伪影
- ✅ **训练更稳定**：平滑的梯度流
- ✅ **可调节**：通过 `gaussian_sigma` 控制边缘软化程度

### 适用场景
- 医学图像分割的对抗扰动生成
- ROI 区域的针对性扰动
- 需要频域约束的扰动生成任务
