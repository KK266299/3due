# 全局可学习频域截止频率 (Global Learnable Cutoff)

## 概述

`noise_slice_frequence_learnable` 方法在 `noise_slice_frequence_h_l_pass` 基础上，将频域滤波器的两个截止频率从**静态常量**改为**全局可学习参数**。

**核心区别**：截止频率是数据集级别的全局参数（2个标量），不是 per-sample 的。整个数据集共享同一组 `(z_cutoff, xy_cutoff)`，通过梯度下降自动寻找最优值。

## 架构

```
Image x ──→ NoiseUNet ──→ noise δ
                              │
global (z_cutoff, xy_cutoff) ─┤  ← 2个 nn.Parameter 标量
                              ▼
             FreqConstraint(δ, z_c, xy_c) ──→ filtered δ
                                                   │
                                              DiceCE Loss
                                              ╱         ╲
                                      opt_unet.step()  opt_cutoff.step()
```

## 实现细节

### GlobalLearnableCutoff

```python
class GlobalLearnableCutoff(nn.Module):
    z_logit = nn.Parameter(...)   # 1个标量
    xy_logit = nn.Parameter(...)  # 1个标量

    def forward(self):
        z  = sigmoid(z_logit)  * (0.45 - 0.01) + 0.01   # → [0.01, 0.45]
        xy = sigmoid(xy_logit) * (0.45 - 0.05) + 0.05   # → [0.05, 0.45]
        return z, xy
```

- 内部存储为 logit（无约束实数），通过 `sigmoid → 线性映射` 转为有效范围内的截止频率
- 初始化时 logit 值设为 `logit((init - range_min) / (range_max - range_min))`，确保初始输出 = YAML 中的默认值

### 为什么不需要 per-sample 循环？

`h_l_pass` 版本中用 CutoffPredictor 预测 per-sample cutoff（每个样本不同的截止频率），导致 batch 内每个样本的频域 mask 不同，FFT 计算图巨大，必须逐样本处理避免 OOM。

全局版本中截止频率是**标量**，对整个 batch 的 mask 完全相同：
- `M_z` 形状 `[1, 1, D, 1, 1]` — 所有样本共享
- `M_xy` 形状 `[1, 1, 1, H, W]` — 所有样本共享
- 可以直接 batch forward + 一次 backward，无需 per-sample 循环
- 内存消耗与静态 cutoff 版本相同

### 训练流程

每个 batch：
1. `GlobalLearnableCutoff()` → 读取当前 `(z_c, xy_c)`（带梯度）
2. `NoiseUNet(x)` → `δ_raw`（带梯度）
3. `FreqConstraint(δ_raw, z_c, xy_c)` → `δ_filtered`（sigmoid mask，可微）
4. ROI mask → clamp → surrogate → DiceCE Loss
5. `loss.backward()` → 同时更新 UNet 和 cutoff 参数
6. `opt_unet.step()` + `opt_cutoff.step()`

每个 epoch 结束时，日志输出当前的截止频率值：
```
[FreqLearnable] Epoch 10: z_cutoff = 0.123456, xy_cutoff = 0.287654
```

### DC 校正

频域 DC 分量（频率=0）代表全局均值。sigmoid mask 在 DC 位置的值由 cutoff 决定，
但我们希望 DC 保持固定衰减（0.1），因此用 `torch.where` 在 DC 位置应用校正因子：

```python
is_dc_z = (abs_freq_z < 1e-7).view(1, 1, D, 1, 1)
corr = 0.1 / (dc_mz * dc_mxy)
M_z = torch.where(is_dc_z, M_z * corr, M_z)  # 无 inplace 操作
```

## 文件

| 文件 | 说明 |
|------|------|
| `src/core/ue_algos/noise_slice_frequence_learnable.py` | 主实现 |
| `configs/method/noise_slice_frequence_learnable.yaml` | 配置文件 |
| `docs/learnable_cutoff_global.md` | 本文档 |

## 训练命令

```bash
python ue_generate.py \
    dataset=flare21 \
    task=flare21_ue \
    task.run_name=freq_learnable \
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
    ue.algorithm.params.z_cutoff_low=0.1 \
    ue.algorithm.params.xy_cutoff_high=0.3 \
    ue.algorithm.params.cutoff_lr_scale=1.0 \
    ue.io.save_from_epoch=50 \
    ue.io.save_every=10 \
    ue.surrogates.s_seg.in_channels=1 \
    ue.surrogates.s_seg.num_classes=5
```

## 与 h_l_pass 版本的对比

| | `h_l_pass` (learnable_cutoff=true) | `learnable` (本版本) |
|---|---|---|
| 截止频率 | Per-sample（每个样本不同） | 全局（整个数据集共享） |
| 预测方式 | CutoffPredictor 网络 (~4K params) | 2个 nn.Parameter 标量 |
| 内存 | 需要 per-sample 循环避免 OOM | 正常 batch forward |
| 优化器 | opt_unet + opt_cutoff (for network) | opt_unet + opt_cutoff (for 2 scalars) |
| 适用场景 | 不同样本需要不同滤波策略 | 同一数据集统一最优滤波参数 |
