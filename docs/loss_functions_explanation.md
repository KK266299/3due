# Loss Functions 详细解释

本文档详细解释 `noise_slice_frequence_learnable.py` 中的两个关键损失函数：
1. **Z-Diversity Loss** (Z轴多样性损失)
2. **Logits Divergence Loss** (Logits发散损失)

---

## 1. Z-Diversity Loss (Z轴多样性损失)

### 1.1 目的

Z-Diversity Loss 用于最大化噪声在**Z轴方向（切片间）的多样性**。

**动机**：3D医学图像由多个2D切片堆叠而成。如果噪声在相邻切片间过于相似，则噪声的攻击效果会被削弱。通过最大化切片间的差异，可以生成更有效的对抗噪声。

### 1.2 计算流程

```
输入: delta [B, C, D, H, W] 噪声张量
                │
                ▼
┌───────────────────────────────────┐
│  Step 1: 2D FFT (每个切片)         │
│  delta_fft_2d = FFT2D(delta)      │
│  形状: [B, C, D, H, W] (complex)   │
└───────────────────────────────────┘
                │
                ▼
┌───────────────────────────────────┐
│  Step 2: 取幅度谱                  │
│  delta_fft_mag = |delta_fft_2d|   │
│  形状: [B, C, D, H, W]             │
└───────────────────────────────────┘
                │
                ▼
┌───────────────────────────────────┐
│  Step 3: 计算相邻切片差异          │
│  slice_diff = mag[:,:,1:] -       │
│               mag[:,:,:-1]         │
│  形状: [B, C, D-1, H, W]           │
└───────────────────────────────────┘
                │
                ▼
┌───────────────────────────────────┐
│  Step 4: 计算L2范数                │
│  l2 = sqrt(sum(slice_diff²))      │
│  形状: [B, C, D-1]                 │
└───────────────────────────────────┘
                │
                ▼
┌───────────────────────────────────┐
│  Step 5: 取平均                    │
│  z_diversity = mean(l2)           │
│  形状: scalar                      │
└───────────────────────────────────┘
                │
                ▼
输出: z_diversity (标量，越大表示切片间差异越大)
```

### 1.3 代码实现

```python
def _compute_z_diversity(self, delta: torch.Tensor) -> torch.Tensor:
    """
    Compute z-axis inter-slice diversity in frequency domain.

    Args:
        delta: [B, C, D, H, W] noise tensor

    Returns:
        z_diversity: Scalar tensor representing mean inter-slice L2 difference
    """
    # Step 1: Apply 2D FFT on each slice (xy-plane)
    delta_fft_2d = torch.fft.fft2(delta, dim=(-2, -1))  # [B, C, D, H, W] complex

    # Step 2: Compute magnitude spectrum for each slice
    delta_fft_mag = delta_fft_2d.abs()  # [B, C, D, H, W]

    # Step 3: Compute L2 difference between adjacent slices along z-axis
    slice_diff = delta_fft_mag[:, :, 1:, :, :] - delta_fft_mag[:, :, :-1, :, :]  # [B, C, D-1, H, W]

    # Step 4: Compute L2 norm for each pair of slices
    l2_per_pair = torch.sqrt((slice_diff ** 2).sum(dim=(-2, -1)) + 1e-10)  # [B, C, D-1]

    # Step 5: Mean over all pairs, channels, and batches
    z_diversity = l2_per_pair.mean()

    return z_diversity
```

### 1.4 损失计算

在训练中，Z-Diversity 被**取负**以最大化（因为优化器是最小化loss）：

```python
z_diversity_loss = -self._z_diversity_weight * z_diversity
```

### 1.5 直观理解

```
切片 z=0:  ████████████████  (频谱A)
切片 z=1:  ████████████████  (频谱B)
切片 z=2:  ████████████████  (频谱C)
           ↓        ↓
       |A-B|²   |B-C|²
           ↓        ↓
         L2范数  L2范数
              ↓
           平均值 = z_diversity
```

**z_diversity越大** → 相邻切片的频谱差异越大 → 噪声在Z方向变化越剧烈

---

## 2. Logits Divergence Loss (Logits发散损失)

### 2.1 目的

Logits Divergence Loss 用于最大化**干净图像和加噪图像的模型输出（logits）之间的差异**。

**动机**：对抗噪声的目标是让模型产生错误的预测。直接优化logits差异可以更有效地引导噪声生成。

### 2.2 支持的模式

| 模式 | 公式 | 描述 |
|------|------|------|
| `l1` | `mean(\|logits_noisy - logits_clean\|)` | 空间域L1范数 |
| `l2` | `sqrt(mean((logits_noisy - logits_clean)²))` | 空间域L2范数 |
| `fft_l1` | `mean(\|FFT(logits_noisy - logits_clean)\|)` | 频域L1范数 |
| `fft_l2` | `sqrt(mean(\|FFT(logits_noisy - logits_clean)\|²))` | 频域L2范数 |
| `kl_div` | `KL(softmax(noisy) \|\| softmax(clean))` | KL散度 |

### 2.3 计算流程

#### 2.3.1 L1/L2 模式

```
输入: logits_clean [B, C, D, H, W]
      logits_noisy [B, C, D, H, W]
                │
                ▼
┌───────────────────────────────────┐
│  Step 1: 计算差异                  │
│  diff = logits_noisy - logits_clean│
│  形状: [B, C, D, H, W]             │
└───────────────────────────────────┘
                │
                ▼
┌───────────────────────────────────┐
│  Step 2: 计算范数                  │
│  L1: divergence = mean(|diff|)    │
│  L2: divergence = sqrt(mean(diff²))│
└───────────────────────────────────┘
                │
                ▼
输出: -weight * divergence (取负以最大化)
```

#### 2.3.2 FFT_L1/FFT_L2 模式

```
输入: logits_clean [B, C, D, H, W]
      logits_noisy [B, C, D, H, W]
                │
                ▼
┌───────────────────────────────────┐
│  Step 1: 计算差异                  │
│  diff = logits_noisy - logits_clean│
└───────────────────────────────────┘
                │
                ▼
┌───────────────────────────────────┐
│  Step 2: 3D FFT                   │
│  diff_fft = FFT3D(diff)           │
│  形状: [B, C, D, H, W] (complex)   │
└───────────────────────────────────┘
                │
                ▼
┌───────────────────────────────────┐
│  Step 3: 计算范数                  │
│  FFT_L1: divergence = mean(|diff_fft|) │
│  FFT_L2: divergence = sqrt(mean(|diff_fft|²)) │
└───────────────────────────────────┘
                │
                ▼
输出: -weight * divergence
```

#### 2.3.3 KL_DIV 模式

```
输入: logits_clean [B, C, D, H, W]
      logits_noisy [B, C, D, H, W]
                │
                ▼
┌───────────────────────────────────┐
│  Step 1: Temperature Scaling      │
│  logits_clean /= temperature      │
│  logits_noisy /= temperature      │
└───────────────────────────────────┘
                │
                ▼
┌───────────────────────────────────┐
│  Step 2: Softmax                  │
│  prob_clean = softmax(logits_clean)│
│  log_prob_noisy = log_softmax(    │
│                    logits_noisy)   │
└───────────────────────────────────┘
                │
                ▼
┌───────────────────────────────────┐
│  Step 3: KL Divergence            │
│  divergence = KL(log_prob_noisy,  │
│                  prob_clean)       │
└───────────────────────────────────┘
                │
                ▼
输出: -weight * divergence
```

### 2.4 代码实现

```python
class LogitsDivergenceLoss(nn.Module):
    def __init__(
        self,
        mode: str = 'fft_l1',
        weight: float = 1.0,
        temperature: float = 1.0,
        fft_dims: Tuple[int, ...] = (-3, -2, -1),
    ):
        super().__init__()
        self.mode = mode.lower()
        self.weight = weight
        self.temperature = temperature
        self.fft_dims = fft_dims

    def forward(
        self,
        logits_clean: torch.Tensor,  # [B, C, D, H, W]
        logits_noisy: torch.Tensor,  # [B, C, D, H, W]
    ) -> torch.Tensor:
        diff = logits_noisy - logits_clean

        if self.mode == 'l1':
            divergence = diff.abs().mean()
        elif self.mode == 'l2':
            divergence = (diff ** 2).mean().sqrt()
        elif self.mode == 'fft_l1':
            diff_fft = torch.fft.fftn(diff, dim=self.fft_dims)
            divergence = diff_fft.abs().mean()
        elif self.mode == 'fft_l2':
            diff_fft = torch.fft.fftn(diff, dim=self.fft_dims)
            divergence = (diff_fft.abs() ** 2).mean().sqrt()
        elif self.mode == 'kl_div':
            logits_clean_scaled = logits_clean / self.temperature
            logits_noisy_scaled = logits_noisy / self.temperature
            prob_clean = F.softmax(logits_clean_scaled, dim=1)
            log_prob_noisy = F.log_softmax(logits_noisy_scaled, dim=1)
            divergence = F.kl_div(log_prob_noisy, prob_clean,
                                  reduction='batchmean', log_target=False)

        # 取负以最大化divergence
        return -self.weight * divergence
```

### 2.5 配置示例

```yaml
ue:
  algorithm:
    params:
      logits_div:
        enabled: true
        mode: fft_l1      # 使用FFT域L1距离
        weight: 0.1       # 损失权重
        temperature: 1.0  # KL模式的温度参数
```

---

## 3. 训练流程中的Loss使用

### 3.1 总体损失计算

```
                        ┌─────────────┐
                        │  Clean Image │
                        │  x [B,C,D,H,W]│
                        └──────┬──────┘
                               │
              ┌────────────────┼────────────────┐
              │                │                │
              ▼                ▼                ▼
     ┌────────────────┐ ┌────────────────┐ ┌────────────────┐
     │   NoiseUNet    │ │ Surrogate Model│ │ Surrogate Model│
     │  生成噪声 δ    │ │  (clean input) │ │  (noisy input) │
     └───────┬────────┘ └───────┬────────┘ └───────┬────────┘
             │                  │                  │
             │          logits_clean        logits_noisy
             │                  │                  │
             ▼                  └────────┬─────────┘
     ┌────────────────┐                  │
     │ Freq Constraint│                  ▼
     │ (可选)         │         ┌────────────────┐
     └───────┬────────┘         │Logits Div Loss │
             │                  │ (最大化差异)    │
             │                  └───────┬────────┘
             │                          │
             ▼                          ▼
     ┌────────────────┐         loss_div = -w * divergence
     │  Z-Diversity   │
     │  Loss          │
     └───────┬────────┘
             │
             ▼
     loss_z = -w * z_diversity
             │
             │
             ▼
     ┌────────────────────────────────────────┐
     │  Total Loss = DiceCE + loss_z + loss_div│
     └────────────────────────────────────────┘
```

### 3.2 梯度流向

```
Total Loss
    │
    ├──→ DiceCE Loss ──→ 通过 logits_noisy 反传到 NoiseUNet
    │
    ├──→ Z-Diversity Loss ──→ 直接从 delta (噪声) 反传到 NoiseUNet
    │
    └──→ Logits Div Loss ──→ 通过 logits_noisy 反传到 NoiseUNet
                             (logits_clean 是 detached 的)
```

---

## 4. 两个Loss的对比

| 特性 | Z-Diversity Loss | Logits Divergence Loss |
|------|------------------|------------------------|
| **输入** | 噪声 δ | logits_clean, logits_noisy |
| **作用域** | 噪声本身 | 模型输出 |
| **优化目标** | 切片间差异最大化 | 预测差异最大化 |
| **计算方式** | 2D FFT → 相邻切片L2 | 多种模式可选 |
| **直接性** | 直接约束噪声结构 | 通过模型输出间接约束 |
| **适用场景** | 提高Z轴多样性 | 提高攻击效果 |

---

## 5. 使用建议

1. **Z-Diversity Loss**:
   - 当噪声在Z轴方向过于平滑时启用
   - 权重建议: 0.01 ~ 0.1
   - 配合频域约束使用效果更好

2. **Logits Divergence Loss**:
   - 推荐使用 `fft_l1` 模式（默认）
   - 权重建议: 0.05 ~ 0.2
   - 如果优化不稳定，可以降低权重或使用 `l1` 模式
