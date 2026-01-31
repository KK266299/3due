# 可学习截止频率 — 完整流程详解

## 目录

1. [背景：什么是频谱 mask 和截止频率](#1-背景)
2. [核心问题：为什么截止频率能被"学习"](#2-核心问题)
3. [完整计算图与梯度流](#3-完整计算图与梯度流)
4. [逐步数据流示例（带具体张量形状）](#4-逐步数据流示例)
5. [关键代码位置索引](#5-关键代码位置索引)
6. [训练命令参考](#6-训练命令参考)

---

## 1. 背景

### 1.1 频谱 mask 的作用

对 3D 医学图像噪声 `delta [B, C, D, H, W]`，我们在频域施加约束：

```
delta  ──→  FFT  ──→  delta_fft * M  ──→  IFFT  ──→  filtered_delta
           频域         频谱mask相乘          回到空域
```

mask `M` 由两个独立滤波器相乘构成：

| 滤波器 | 轴 | 类型 | 目的 |
|--------|------|------|------|
| M_z | Z轴（层间） | **高通** | 让不同 slice 之间的噪声差异大（保留高频） |
| M_xy | XY平面（层内） | **低通** | 让每个 slice 内部的噪声平滑（保留低频） |

### 1.2 截止频率的含义

频率坐标通过 `torch.fft.fftfreq` 归一化到 `[-0.5, 0.5)`：

```
Z轴高通:  |f_z| < z_cutoff_low  的频率被衰减  (默认 z_cutoff_low = 0.1)
XY低通:   r_xy  > xy_cutoff_high 的频率被衰减  (默认 xy_cutoff_high = 0.3)
```

**静态模式**下这两个值是 yaml 中写死的常量。**可学习模式**下它们由一个小网络根据每个输入样本预测。

---

## 2. 核心问题

### 2.1 为什么截止频率能被"学习"？

关键在于 **可微分性**。原版 mask 使用 `torch.where`（不可微的阶跃函数），可学习版本替换为 `sigmoid`（处处可微）：

**原版（静态，不可微）：**
```python
# torch.where 是分段函数，对 cutoff 没有梯度
M_z = torch.where(
    |f_z| >= z_cutoff,          # 条件
    1.0,                         # 通过
    exp(-(...) / 2σ²)           # 高斯衰减
)
```

**可学习版（可微）：**
```python
# sigmoid 对 z_c 处处有梯度
M_z = sigmoid((|f_z| - z_c) / σ_z)    # z_c 是 per-sample 的截止频率
M_xy = sigmoid((xy_c - r_xy) / σ_xy)  # xy_c 是 per-sample 的截止频率
```

`sigmoid` 的特性：
- 当 `|f_z| >> z_c` 时 → sigmoid ≈ 1（通过）
- 当 `|f_z| << z_c` 时 → sigmoid ≈ 0（衰减）
- 在 `|f_z| ≈ z_c` 附近平滑过渡
- **对 `z_c` 有明确梯度**：`∂sigmoid/∂z_c = -sigmoid(1-sigmoid)/σ`

### 2.2 截止频率如何产生？

```
输入图像 x [B,C,D,H,W]
    │
    ▼
AdaptiveAvgPool3d(4)     →  [B, C, 4, 4, 4]     # 压缩空间维度
    │
    ▼
Flatten                   →  [B, C×64]
    │
    ▼
Linear(C×64, 64) + LeakyReLU
    │
    ▼
Linear(64, 2)            →  [B, 2]               # 原始 logits
    │
    ▼
sigmoid                   →  [B, 2]  ∈ (0, 1)     # 归一化
    │
    ├──→ z_cutoff  = out[:,0] × (0.45-0.01) + 0.01  →  [B]  ∈ [0.01, 0.45]
    └──→ xy_cutoff = out[:,1] × (0.45-0.05) + 0.05  →  [B]  ∈ [0.05, 0.45]
```

**每个样本**得到自己的一对截止频率。不同的图像内容会产生不同的截止频率。

### 2.3 初始化保证

为了让开启 learnable 后**初始行为等同于静态版本**：

```python
# 最后一层 Linear:
weight = 0                  # 初始时输出不依赖输入，所有样本输出相同
bias = logit(...)           # 反推使得 sigmoid(bias) 恰好映射到默认值 0.1 / 0.3
```

也就是说训练刚开始时，无论输入什么图像，输出都是 `z_cutoff=0.1, xy_cutoff=0.3`。随着训练推进，网络逐渐学会为不同样本输出不同的截止频率。

---

## 3. 完整计算图与梯度流

```
                                 LearnableCutoffPredictor
                                 ┌────────────────────┐
                          ┌─────►│ Pool → MLP → σ     │
                          │      │                    │
                          │      │ 输出: z_c, xy_c    │──────────┐
                          │      └────────────────────┘          │
                          │       θ_cutoff (可学习参数)            │
                          │                                      ▼
 输入图像 x ──────────────┤                            FrequencyDomainConstraint
 [B,C,D,H,W]             │                            ┌──────────────────────┐
                          │      NoiseUNet             │                      │
                          │      ┌──────────┐          │  1. FFT(delta_raw)   │
                          └─────►│ UNet(x)  │─────────►│  2. 构建 M(z_c,xy_c) │
                                 │          │ delta_raw │  3. fft × M          │
                                 └──────────┘          │  4. IFFT → filtered  │
                                  θ_unet               └──────────┬───────────┘
                                                                  │
                                                                  ▼
                                                         ROI Mask (可选)
                                                                  │
                                                                  ▼
                                                         Clip to [-ε, ε]
                                                                  │
                                                                  ▼
                                                         x + delta → perturbed
                                                                  │
                                                                  ▼
                                                     Surrogate Model (冻结)
                                                                  │
                                                                  ▼
                                                          DiceCE Loss
                                                                  │
                                                   ┌──────────────┴──────────────┐
                                                   │          .backward()        │
                                                   ▼                             ▼
                                            ∂L/∂θ_unet                    ∂L/∂θ_cutoff
                                                   │                             │
                                                   └──────────┬──────────────────┘
                                                              ▼
                                                     Adam optimizer.step()
                                                   (同一个 optimizer 更新两组参数)
```

### 3.1 梯度链路详解（为什么 cutoff 能收到梯度）

从 loss 到 cutoff predictor 的梯度链路：

```
∂L/∂θ_cutoff = ∂L/∂logits                    # DiceCE loss 对 logits 的梯度
             × ∂logits/∂perturbed             # surrogate 的反传
             × ∂perturbed/∂delta_filtered     # x + delta, 梯度 = 1
             × ∂delta_filtered/∂M             # IFFT(FFT(delta)×M), 梯度 ∝ FFT(delta)
             × ∂M/∂z_c (或 ∂M/∂xy_c)        # sigmoid 的梯度
             × ∂z_c/∂θ_cutoff                 # MLP 的反传
```

每一步都是可微的，因此梯度可以一路回传到 cutoff predictor 的参数。

### 3.2 sigmoid 替换的直觉

以 Z 轴高通为例：

```
            静态 (torch.where)                   可学习 (sigmoid)
  M_z │                                   M_z │
  1.0 │          ┌─────────              1.0 │          ╱─────────
      │          │                            │        ╱
      │          │                            │      ╱
  0.0 │──────────┘                       0.0 │────╱
      └─────────────────── |f_z|             └─────────────────── |f_z|
              z_cutoff                               z_cutoff
        (不可微的阶跃)                         (可微的 S 形曲线)
```

sigmoid 版本中，移动 `z_cutoff` 会平滑地改变整个 mask 的形状，优化器可以通过梯度下降调节它。

---

## 4. 逐步数据流示例

假设 `batch_size=2, C=1, D=16, H=128, W=128`：

### Step 1: Cutoff Predictor 前向

```
x [2, 1, 16, 128, 128]
  │
  ▼ AdaptiveAvgPool3d(4)
x_pool [2, 1, 4, 4, 4]
  │
  ▼ Flatten
x_flat [2, 64]
  │
  ▼ Linear(64, 64) + LeakyReLU + Linear(64, 2)
logits [2, 2]
  │
  ▼ sigmoid + 线性映射
z_cutoff  = [0.08, 0.12]     # 样本0: 0.08, 样本1: 0.12
xy_cutoff = [0.25, 0.35]     # 样本0: 0.25, 样本1: 0.35
```

**注意：两个样本有不同的截止频率。**

### Step 2: Noise UNet 前向

```
x [2, 1, 16, 128, 128]
  │
  ▼ UNet → tanh × ε
delta_raw [2, 1, 16, 128, 128]    # 原始噪声 ∈ [-ε, ε]
```

### Step 3: 构建 per-sample 频谱 Mask

```
频率网格 (固定, 缓存):
  abs_k_z [1, 16, 128, 128]     # |f_z| 值, 范围 [0, 0.5)
  r_xy    [1, 16, 128, 128]     # sqrt(f_x² + f_y²)

z_cutoff → reshape [2, 1, 1, 1]:
  z_c = [[0.08], [0.12]]

Z轴高通 mask:
  M_z = sigmoid((abs_k_z - z_c) / 0.05)    # [2, 16, 128, 128]
  │
  │  样本0: z_c=0.08, 所以 |f_z|>0.08 的频率通过
  │  样本1: z_c=0.12, 所以 |f_z|>0.12 的频率通过（更严格的高通）

XY低通 mask:
  M_xy = sigmoid((xy_c - r_xy) / 0.1)      # [2, 16, 128, 128]
  │
  │  样本0: xy_c=0.25, 所以 r_xy<0.25 的频率通过
  │  样本1: xy_c=0.35, 所以 r_xy<0.35 的频率通过（更宽松的低通）

合并:
  M = M_z × M_xy              # [2, 16, 128, 128]
  M[:, 0, 0, 0] = 0.1         # DC 分量衰减
  M = M.unsqueeze(1)           # [2, 1, 16, 128, 128]
```

### Step 4: 频域滤波

```
delta_fft = FFT(delta_raw)              # [2, 1, 16, 128, 128] complex
delta_fft_filtered = delta_fft × M      # per-sample mask 相乘
delta_filtered = IFFT(delta_fft_filtered).real  # [2, 1, 16, 128, 128]
```

### Step 5: ROI Mask + Clip

```
delta = delta_filtered × roi_mask       # 如果 roi_aware=true
delta = clamp(delta, -ε, ε)            # [2, 1, 16, 128, 128]
```

### Step 6: 计算 Loss

```
perturbed = clamp(x + delta, 0, 1)     # [2, 1, 16, 128, 128]
perturbed_normed = normalize(perturbed)
logits = surrogate(perturbed_normed)    # [2, num_classes, 16, 128, 128]
loss = DiceCE(logits, labels)           # 标量
```

### Step 7: 反向传播

```
loss.backward()
  │
  ├──→ ∂L/∂θ_unet      # Noise UNet 的参数梯度
  │
  └──→ ∂L/∂θ_cutoff    # Cutoff Predictor 的参数梯度
       │
       │ 梯度路径: loss → logits → perturbed → delta → delta_filtered
       │         → (IFFT) → delta_fft_filtered → M → sigmoid → z_c/xy_c
       │         → MLP → θ_cutoff
       │
       │ 核心: ∂M/∂z_c = ∂sigmoid/∂z_c = -sigmoid(1-sigmoid)/σ ≠ 0
       │                                   ↑ 这就是为什么用 sigmoid

optimizer.step()    # 同一个 Adam 同时更新两组参数
```

### 直觉理解梯度方向

- 如果 loss 希望噪声在 Z 轴有**更多**高频变化 → 梯度会推动 `z_cutoff` **降低**（允许更多 Z 频率通过）
- 如果 loss 希望噪声在 XY 平面**更平滑** → 梯度会推动 `xy_cutoff` **降低**（截断更多 XY 高频）
- 对于不同的图像样本，最优截止频率可能不同（比如大器官 vs 小器官），网络可以学到这种差异

---

## 5. 关键代码位置索引

| 功能 | 文件 | 行号 |
|------|------|------|
| `LearnableCutoffPredictor` 定义 | `noise_slice_frequence_h_l_pass.py` | 80-133 |
| `_build_spectral_mask_batched` (可微 mask) | 同上 | 219-252 |
| `FrequencyDomainConstraint.forward` (分发逻辑) | 同上 | 254-294 |
| `_init_components` 中构建 predictor + optimizer | 同上 | 495-533 |
| `noise_step_batch` 训练循环 | 同上 | 676-741 |
| 返回 cutoff 统计量 | 同上 | 758-760 |
| YAML 配置 | `noise_slice_frequence_h_l_pass.yaml` | 65-67 |

---

## 6. 训练命令参考

### 开启可学习截止频率

```bash
python ue_generate.py \
    dataset=flare21 \
    task=flare21_ue \
    task.run_name=noise_slice_freq_h_l_learnable \
    method=noise_slice_frequence_h_l_pass \
    training.epochs=100 \
    training.batch_size=8 \
    training.gpu_ids=[2] \
    ue.key.type=samplewise \
    ue.key.from=field \
    ue.key.field=case_id \
    ue.algorithm.params.epsilon=0.0156863 \
    ue.algorithm.params.noise_step=1 \
    ue.algorithm.params.surrogate_step=10 \
    ue.algorithm.params.roi_aware=true \
    ue.algorithm.params.soft_edge=false \
    ue.algorithm.params.learnable_cutoff=true \
    ue.algorithm.params.cutoff_lr_scale=1.0 \
    ue.io.save_from_epoch=50 \
    ue.io.save_every=10 \
    ue.surrogates.s_seg.in_channels=1 \
    ue.surrogates.s_seg.num_classes=5
```

### 关闭（与原版完全一致）

```bash
python ue_generate.py \
    dataset=flare21 \
    task=flare21_ue \
    task.run_name=noise_slice_freq_h_l_static \
    method=noise_slice_frequence_h_l_pass \
    training.epochs=100 \
    training.batch_size=8 \
    training.gpu_ids=[2] \
    ue.key.type=samplewise \
    ue.key.from=field \
    ue.key.field=case_id \
    ue.algorithm.params.epsilon=0.0156863 \
    ue.algorithm.params.noise_step=1 \
    ue.algorithm.params.surrogate_step=10 \
    ue.algorithm.params.roi_aware=true \
    ue.algorithm.params.soft_edge=false \
    ue.algorithm.params.learnable_cutoff=false \
    ue.io.save_from_epoch=50 \
    ue.io.save_every=10 \
    ue.surrogates.s_seg.in_channels=1 \
    ue.surrogates.s_seg.num_classes=5
```

### 关键参数说明

| 参数 | 含义 | 建议值 |
|------|------|--------|
| `learnable_cutoff` | 开关 | `true` / `false` |
| `cutoff_lr_scale` | cutoff predictor 学习率倍率（相对于 noise_unet lr） | `0.1 ~ 1.0`，建议从 `1.0` 开始 |
| `z_cutoff_low` | 静态默认值 / learnable 初始值 | `0.1` |
| `xy_cutoff_high` | 静态默认值 / learnable 初始值 | `0.3` |
| `z_sigma` | Z 轴过渡带宽度（也控制 sigmoid 的陡峭程度） | `0.05` |
| `xy_sigma` | XY 过渡带宽度 | `0.1` |

### 日志输出

开启 `learnable_cutoff=true` 后，`noise_step_batch` 的返回字典会额外包含：

```
z_cutoff_mean:  0.0923    # 当前 batch 中 z 截止频率的均值
xy_cutoff_mean: 0.2814    # 当前 batch 中 xy 截止频率的均值
```

这两个值会随训练变化，可用于监控截止频率是否在合理范围内演化。
