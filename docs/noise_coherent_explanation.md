# Noise Coherent: 技术说明文档

## 概述

Noise Coherent 是一种基于频域可学习参数 P 的扰动生成方法，其核心思想是：
1. **维护可学习的 P 参数（实部和虚部）**
2. **应用固定的频域 mask，然后 IFFT 到空域**
3. **tanh + epsilon 限制扰动范围**
4. **可选的 ROI 软边缘 mask（可开关）**
5. **最小化 surrogate 在扰动图像上的分割损失（min-min）**

---

## 1. P 参数机制

### 1.1 初始化
```python
# P_real 和 P_imag 初始化为可学习参数
P_real = nn.Parameter(torch.randn(C, D, H, W) * init_scale)  # 实部
P_imag = nn.Parameter(torch.randn(C, D, H, W) * init_scale)  # 虚部
```

配置参数：
- `p_init.init_scale`: 初始化尺度（默认 10.0）
  - **重要**：需要较大的初始值（如 10.0），因为经过 IFFT → tanh → epsilon 后值会被大幅缩小
  - 如果 init_scale 太小（如 0.01），初始 δ 会接近 0

### 1.2 为什么需要大的初始化尺度？

由于扰动经过多个变换步骤，初始值需要足够大：

```python
# 假设 init_scale = 0.01（太小）
P_real ∼ N(0, 0.01²)  # 很小的随机值

# 经过 IFFT
delta_raw = IFFT(P_complex * M).real  # 值进一步缩小（频域 mask 很稀疏）

# 经过 tanh
delta_tanh = tanh(delta_raw)  # 对小值，tanh(x) ≈ x

# 乘以 epsilon
delta = delta_tanh * epsilon  # 例如 0.0157

# 结果：delta ≈ 0.0001 * 0.0157 ≈ 0.0000016（接近 0！）
```

**解决方案**：使用较大的 `init_scale`（如 10.0）：
```python
# init_scale = 10.0（合理）
P_real ∼ N(0, 10²)  # 较大的随机值

# 最终 delta 会有合理的幅度（接近 epsilon 范围）
```

### 1.3 直接更新 P
**重要**：P 是直接通过梯度下降更新，**不是增量更新**。

每次迭代：
```python
# 构建复数 P
P_complex = P_real + i * P_imag

# 应用频域 mask
P_masked = P_complex * M

# IFFT 到空域
delta_raw = IFFT(P_masked).real

# tanh + epsilon
delta = tanh(delta_raw) * epsilon

# 反向传播更新 P_real 和 P_imag
loss.backward()  # 梯度流向 P_real 和 P_imag
optimizer.step()  # 直接更新 P
```

---

## 2. Loss 计算方法

### 2.1 完整流程

```
输入: x (图像), y (标签)

步骤 1: 构建复数 P
    P_complex = P_real + i * P_imag  # [C, D, H, W]

步骤 2: 应用频域 mask
    P_masked = P_complex * M

    其中 M 是固定的频域 mask:
    - Z 轴（层间）: 低频 (|f_z| <= z_max)
    - XY 平面（层内）: 带通 (xy_min <= |f_xy| <= xy_max)

步骤 3: IFFT 到空域
    delta_raw = IFFT(P_masked).real  # [C, D, H, W]

步骤 4: tanh + epsilon
    delta = tanh(delta_raw) * epsilon  # [C, D, H, W]

步骤 5: 扩展到 batch
    delta = delta.unsqueeze(0).expand(B, -1, -1, -1, -1)  # [B, C, D, H, W]

步骤 6: 应用 ROI gate（可选）
    ROI_mask = create_roi_mask(label)  # [B, 1, D, H, W]

    if roi_gate.enabled == true:
        # 软边缘: 高斯平滑，值在 [0, 1]
        ROI_mask = GaussianSmooth(ROI_mask)
    else:
        # 硬边缘: 值为 0 或 1
        ROI_mask = ROI_mask

    delta = delta * ROI_mask

步骤 7: 生成扰动图像
    x_perturbed = clip(x + delta, 0, 1)

步骤 8: 前向 surrogate（冻结）
    logits = surrogate(normalize(x_perturbed))

步骤 9: 计算分割损失
    loss = DiceCELoss(logits, y)

    其中 DiceCELoss = lambda_dice * Dice + lambda_ce * CrossEntropy

步骤 10: 反向传播（最小化损失）
    loss.backward()  # 梯度回传到 P_real 和 P_imag

步骤 11: 优化器更新
    optimizer.step()  # 更新 P_real 和 P_imag
```

### 2.2 Loss 函数详解

**DiceCELoss** 结合了 Dice Loss 和 Cross-Entropy Loss：

```python
DiceCELoss = lambda_dice * DiceLoss + lambda_ce * CrossEntropyLoss

DiceLoss = 1 - (2 * sum(pred * target) + smooth) / (sum(pred) + sum(target) + smooth)

CrossEntropyLoss = -sum(target * log(softmax(pred)))
```

**目标**：**最小化** surrogate 的分割损失（min-min），使 surrogate 在扰动图像上的预测**更准确**。

这与传统对抗攻击不同，这里是训练 surrogate 和扰动同时最小化损失，达到 min-min 平衡。

---

## 3. 软边缘梯度回传机制

### 3.1 ROI 软边缘生成

```python
# 1. 创建二值 mask
mask = (label > 0).float()  # [B, D, H, W]

# 2. 可选：形态学膨胀
mask = MaxPool3D(mask, kernel_size=3)

# 3. 添加通道维度
mask = mask.unsqueeze(1)  # [B, 1, D, H, W]

# 4. 高斯平滑（仅当 roi_gate.enabled=true）
if roi_gate.enabled and gaussian_sigma > 0:
    kernel = GaussianKernel3D(sigma=gaussian_sigma)
    mask = Conv3D(mask, kernel)  # [B, 1, D, H, W]

    # 归一化到 [0, 1]
    ROI_mask = mask / max(mask)
else:
    # 硬边缘
    ROI_mask = mask  # 值为 0 或 1
```

### 3.2 软边缘开关

通过配置 `roi_gate.enabled` 来控制：

| `roi_gate.enabled` | 边缘类型 | ROI_mask 值范围 | 梯度特性 |
|-------------------|---------|----------------|---------|
| **false** | 硬边缘 | {0, 1} | 边界截断 |
| **true** | 软边缘 | [0, 1] | 平滑衰减 |

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

**硬边缘**（`enabled=false`）：
- ROI 内部（mask=1）：梯度完全保留
- ROI 外部（mask=0）：梯度完全截断
- 边界：梯度不连续

**软边缘**（`enabled=true`）：
- ROI 内部（mask≈1）：梯度几乎不受影响
- ROI 外部（mask≈0）：梯度接近零
- **边界区域**（0 < mask < 1）：**梯度平滑衰减**，可参与学习

### 3.4 软边缘的优势

| 对比项 | 硬边缘（enabled=false） | 软边缘（enabled=true） |
|--------|----------------------|---------------------|
| 边界梯度 | 不连续，可能梯度消失 | 平滑连续，梯度平滑衰减 |
| 视觉效果 | 可能出现明显边界伪影 | 自然过渡，无明显伪影 |
| 学习能力 | 边界区域无法学习 | 边界区域可以学习（衰减） |
| 训练稳定性 | 较低 | 较高 |
| 计算开销 | 低 | 稍高（高斯卷积） |

### 3.5 参数调优

**roi_gate.enabled**（软边缘开关）：
- **false**：硬边缘，训练更快但可能不稳定
- **true**：软边缘，训练更稳定但计算稍慢

**gaussian_sigma**（高斯平滑标准差，仅当 enabled=true 时有效）：
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
Loss (DiceCELoss) - 最小化
  ↓ ∂loss/∂logits
Logits (Surrogate output)
  ↓ ∂logits/∂x_perturbed  [Surrogate 冻结，梯度不更新 surrogate]
x_perturbed = clip(x + delta_masked)
  ↓ ∂x_perturbed/∂delta_masked
delta_masked = delta * ROI_mask
  ↓ ∂delta_masked/∂delta = ROI_mask  [软边缘: 梯度缩放; 硬边缘: 梯度截断]
delta = tanh(delta_raw) * epsilon
  ↓ ∂delta/∂delta_raw = epsilon * (1 - tanh²(delta_raw))
delta_raw = IFFT(P_masked).real
  ↓ ∂delta_raw/∂P_masked  [IFFT 可微]
P_masked = P_complex * M
  ↓ ∂P_masked/∂P_complex = M
P_complex = P_real + i * P_imag
  ↓ ∂P_complex/∂P_real, ∂P_complex/∂P_imag
P_real, P_imag 参数更新
```

---

## 5. 配置示例

### 5.1 硬边缘（默认，类似 unet_roi_noise）
```yaml
ue:
  algorithm:
    name: noise_coherent
    params:
      epsilon: 0.0156863  # 4/255

      # P 初始化
      p_init:
        init_scale: 10.0

      # 频域 mask
      spectral_mask:
        z_max: 0.125
        xy_min: 0.05
        xy_max: 0.3

      # ROI 硬边缘
      roi_gate:
        enabled: false      # 硬边缘
        dilate_kernel_size: 3

      # 优化器
      optimizer:
        lr: 1.0e-3
        weight_decay: 1.0e-5

      # 训练步数
      noise_step: 1
      surrogate_step: 2
```

### 5.2 软边缘（实验性）
```yaml
ue:
  algorithm:
    name: noise_coherent
    params:
      epsilon: 0.0156863

      p_init:
        init_scale: 0.01

      spectral_mask:
        z_max: 0.125
        xy_min: 0.05
        xy_max: 0.3

      # ROI 软边缘
      roi_gate:
        enabled: true       # 软边缘
        dilate_kernel_size: 3
        gaussian_sigma: 2.0  # 控制边缘软化程度

      optimizer:
        lr: 1.0e-3
        weight_decay: 1.0e-5

      noise_step: 1
      surrogate_step: 2
```

---

## 6. 使用方法

### 6.1 训练（FLARE21 数据集）
```bash
python ue_generate.py \
  dataset=flare21 \
  task.run_name=noise_coherent_4_255 \
  method=noise_coherent \
  task=flare21_ue \
  training.epochs=100 \
  ue.key.type=samplewise \
  ue.key.from=field \
  ue.key.field=case_id \
  training.batch_size=8 \
  training.gpu_ids=[0] \
  ue.algorithm.params.epsilon=0.0156863 \
  ue.algorithm.params.surrogate_step=10 \
  ue.io.save_from_epoch=50 \
  ue.io.save_every=10
```

### 6.2 调试软边缘效果

```bash
# 硬边缘（默认）
python ue_generate.py ... \
  ue.algorithm.params.roi_gate.enabled=false

# 软边缘
python ue_generate.py ... \
  ue.algorithm.params.roi_gate.enabled=true \
  ue.algorithm.params.roi_gate.gaussian_sigma=2.0

# 很软的边缘
python ue_generate.py ... \
  ue.algorithm.params.roi_gate.enabled=true \
  ue.algorithm.params.roi_gate.gaussian_sigma=5.0
```

### 6.3 调整频域 mask

```bash
# 更宽的频域范围
python ue_generate.py ... \
  ue.algorithm.params.spectral_mask.z_max=0.25 \
  ue.algorithm.params.spectral_mask.xy_min=0.03 \
  ue.algorithm.params.spectral_mask.xy_max=0.5

# 更窄的频域范围
python ue_generate.py ... \
  ue.algorithm.params.spectral_mask.z_max=0.1 \
  ue.algorithm.params.spectral_mask.xy_min=0.1 \
  ue.algorithm.params.spectral_mask.xy_max=0.2
```

---

## 7. 与 unet_roi_noise 的对比

| 特性 | unet_roi_noise | noise_coherent |
|------|---------------|---------------|
| 扰动生成 | UNet 直接输出空域噪声 | P 参数 → 频域 mask → IFFT |
| 参数量 | UNet 参数（较多） | P 参数（较少，C×D×H×W×2） |
| 频域约束 | 无 | 固定频域 mask |
| 软边缘 | 无 | 可选（roi_gate.enabled） |
| 训练目标 | 最小化损失 | 最小化损失 |
| 噪声存储 | 一致 | 一致 |
| 适用场景 | 通用 ROI 扰动 | 需要频域约束的 ROI 扰动 |

---

## 8. 总结

### 核心特性
1. **P 参数直接更新**：通过梯度下降直接优化 P_real 和 P_imag
2. **频域约束**：通过固定 mask 保证频域一致性
3. **tanh + epsilon**：限制扰动范围
4. **软边缘可选**：通过 roi_gate.enabled 开关控制
5. **min-min 优化**：最小化 surrogate 损失

### 软边缘的关键作用
- ✅ **可开关**：roi_gate.enabled 控制硬边缘/软边缘
- ✅ **梯度可回传**：软边缘区域的梯度平滑衰减
- ✅ **视觉效果自然**：避免硬边界伪影
- ✅ **训练更稳定**：平滑的梯度流
- ✅ **可调节**：通过 gaussian_sigma 控制边缘软化程度

### 适用场景
- 医学图像分割的对抗扰动生成
- ROI 区域的针对性扰动（硬边缘或软边缘）
- 需要频域约束的扰动生成任务
- 需要可学习频域参数的场景

### 噪声存储
- 与 unet_roi_noise 等方法一致
- 使用 noise_backend 存储和管理噪声
- 支持 samplewise key 管理
