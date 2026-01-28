# Noise Slice Frequence (with Logits Divergence Loss)

## 概述

`noise_slice_frequence_logits` 是一个频域约束噪声生成算法，现已集成 **Logits Divergence Loss（预测散度损失）**，用于最大化加噪声图像与干净图像之间的预测差异。

## 核心设计

### 1. 基础架构
- **UNet噪声生成器**: 使用小型 U-Net 网络生成基础噪声
- **频域约束**:
  - Z轴高通滤波：最大化层间多样性
  - XY平面低通滤波：确保层内平滑
- **软边缘ROI掩码**: 二值化 → 膨胀 → 高斯模糊
- **Logits Divergence Loss**: 最大化干净与加噪声图像的预测差异

### 2. Logits Divergence Loss

#### 设计目标
最大化加噪声数据集与干净数据集在代理模型上的预测差异，使噪声能更有效地影响模型预测。

#### 数学公式

总损失函数：
```
L_total = L_seg(f(x + δ), y) + λ * L_div(f(x), f(x + δ))
```

其中：
- `L_seg`: 分割损失（DiceCE Loss）
- `L_div`: 散度损失（取负以最大化差异）
- `λ`: 散度损失权重 (`logits_div_weight`)
- `f(x)`: 干净图像的模型预测
- `f(x + δ)`: 加噪声图像的模型预测

#### 支持的散度计算模式

| 模式 | 描述 | 公式 |
|------|------|------|
| `l1` | 直接L1范数 | `mean(\|logits_noisy - logits_clean\|)` |
| `l2` | 直接L2范数 | `sqrt(mean((logits_noisy - logits_clean)^2))` |
| `fft_l1` | FFT后L1范数（推荐） | `mean(\|FFT(logits_noisy - logits_clean)\|)` |
| `fft_l2` | FFT后L2范数 | `sqrt(mean(\|FFT(logits_noisy - logits_clean)\|^2))` |
| `kl_div` | KL散度 | `KL(softmax(logits_clean) \|\| softmax(logits_noisy))` |

**推荐模式**: `fft_l1`
- 在频域计算差异可以捕获更多的结构性差异
- L1范数对异常值更鲁棒

## 配置参数

### YAML 配置文件

```yaml
ue:
  algorithm:
    name: noise_slice_frequence_logits
    params:
      # === Logits Divergence Loss ===
      logits_div_enabled: true    # 启用/禁用散度损失
      logits_div_mode: fft_l1     # 散度计算模式
      logits_div_weight: 1.0      # 散度损失权重
      logits_div_temperature: 1.0 # KL散度温度参数（仅kl_div模式）

      # === 其他参数 ===
      epsilon: 0.0313725          # L_inf 扰动界限 (8/255)
      noise_step: 1               # UNet训练迭代次数
      surrogate_step: 10          # 代理模型训练步数

      # 频域约束
      z_cutoff_low: 0.1
      z_sigma: 0.05
      xy_cutoff_high: 0.3
      xy_sigma: 0.1

      # ROI掩码
      roi_aware: true
      soft_edge: true
      dilate_iterations: 2
      dilate_kernel_size: 3
      gaussian_sigma: 2.0
```

### 参数说明

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| `logits_div_enabled` | bool | `true` | 是否启用散度损失 |
| `logits_div_mode` | str | `fft_l1` | 散度计算模式 |
| `logits_div_weight` | float | `1.0` | 散度损失权重 |
| `logits_div_temperature` | float | `1.0` | KL散度温度（仅kl_div模式） |

## 使用方法

### 命令行

```bash
python ue_generate.py \
    dataset=brats19 \
    task=brats19_ue \
    method=noise_slice_frequence_logits \
    task.run_name=freq_slice_logits \
    training.epochs=100
```

### 调整散度模式

```bash
# 使用直接L1范数
python ue_generate.py \
    method=noise_slice_frequence_logits \
    ue.algorithm.params.logits_div_mode=l1

# 使用KL散度
python ue_generate.py \
    method=noise_slice_frequence_logits \
    ue.algorithm.params.logits_div_mode=kl_div \
    ue.algorithm.params.logits_div_temperature=2.0
```

### 调整损失权重

```bash
# 增加散度损失权重
python ue_generate.py \
    method=noise_slice_frequence_logits \
    ue.algorithm.params.logits_div_weight=2.0
```

### 禁用散度损失

```bash
# 禁用logits divergence loss（回退到原始版本行为）
python ue_generate.py \
    method=noise_slice_frequence_logits \
    ue.algorithm.params.logits_div_enabled=false
```

## 输出指标

训练过程中会记录以下指标：

| 指标 | 描述 |
|------|------|
| `noise_loss` | 总损失 (seg_loss + div_loss) |
| `seg_loss` | 分割损失 |
| `div_loss` | 散度损失（负值表示正在最大化差异） |
| `logits_diff_l1` | 最终噪声的logits L1差异 |
| `delta_linf` | 噪声的L_inf范数 |
| `z_high_freq_energy` | Z轴高频能量比例 |
| `xy_low_freq_energy` | XY平面低频能量比例 |

## 实现细节

### LogitsDivergenceLoss 类

```python
class LogitsDivergenceLoss(nn.Module):
    def __init__(
        self,
        mode: str = 'fft_l1',
        weight: float = 1.0,
        temperature: float = 1.0,
        fft_dims: Tuple[int, ...] = (-3, -2, -1),
    ):
        ...

    def forward(
        self,
        logits_clean: torch.Tensor,
        logits_noisy: torch.Tensor,
    ) -> torch.Tensor:
        # 返回负的散度，以便通过最小化loss来最大化差异
        return -self.weight * divergence
```

### 训练流程

1. **生成噪声**: UNet 生成基础噪声
2. **频域约束**: 应用 Z 轴高通 + XY 平面低通滤波
3. **ROI掩码**: 将噪声限制在ROI区域
4. **获取干净预测**: 通过代理模型获取干净图像的 logits
5. **获取加噪声预测**: 通过代理模型获取加噪声图像的 logits
6. **计算损失**: `L_total = L_seg + L_div`
7. **反向传播**: 更新噪声 UNet 参数

## 与原始版本的对比

| 特性 | 原始版本 (logits_div_enabled=false) | 新版本 (logits_div_enabled=true) |
|------|--------------------------------------|----------------------------------|
| 频域约束 | ✓ | ✓ |
| ROI掩码 | ✓ | ✓ |
| 分割损失 | ✓ | ✓ |
| Logits散度损失 | ✗ | ✓ |
| 输出指标 | 4个 | 7个 |

## 文件结构

```
src/core/ue_algos/
└── noise_slice_frequence_logits.py              # 包含logits散度损失

configs/method/
└── noise_slice_frequence_logits.yaml            # 配置文件

docs/
└── noise_slice_frequence_logits_z_up_logits.md  # 本文档
```

## 注意事项

1. **计算开销**: 添加 logits 散度损失会增加一次干净图像的前向传播，但由于使用 `torch.no_grad()`，不会增加显存占用
2. **权重调节**: 根据实验效果调整 `logits_div_weight`，建议从 1.0 开始
3. **模式选择**: `fft_l1` 是推荐的默认模式，如果需要更强的类别级差异，可以尝试 `kl_div`
