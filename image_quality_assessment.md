# 图像质量评估 (Image Quality Assessment)

## 概述

本工具用于评估 **加噪后图像** 与 **原始图像** 之间的图像质量差异，主要计算以下指标：

| 指标 | 全称 | 含义 | 越高越好？ |
|------|------|------|-----------|
| **PSNR** | Peak Signal-to-Noise Ratio | 峰值信噪比，衡量图像像素级失真程度 | 是 |
| **SSIM** | Structural Similarity Index | 结构相似性，衡量结构、亮度、对比度的综合相似度 | 是 |
| **Noise L2** | L2 Norm (RMS) | 噪声的均方根幅度 | - |
| **Noise Linf** | L-infinity Norm | 噪声的最大绝对值 | - |

## 方法说明

### 1. 噪声加载

噪声通过 `UEShardsAccessor` 从 `manifest.json` 文件加载。噪声以 `int8` 量化格式存储，加载时自动反量化为 `float32`：

```
x_hat = q * scale,   其中 scale = eps / 127.0
```

### 2. 加噪过程

与 `visualize_unet_noise.py` 保持一致，加噪方式为：

```python
noisy_image = clamp(original_image + noise, 0, 1)
```

其中 `original_image` 和 `noise` 均为 `[C, D, H, W]` 张量（4通道3D医学图像），值域 `[0, 1]`。

### 3. PSNR 计算

PSNR 在整个 volume 上计算：

```
MSE = mean((original - noisy)^2)
PSNR = 10 * log10(data_range^2 / MSE)
```

其中 `data_range = 1.0`。实现位于 `src/utils/eval_metrics.py:compute_psnr`。

### 4. SSIM 计算

SSIM 采用 **逐 slice 平均** 策略，沿 depth 维度将 3D volume 切分为 2D slices，逐片计算 SSIM 后取平均值。

SSIM 使用高斯窗口 (默认 `win_size=11`, `sigma=1.5`)，实现基于 [pytorch-msssim](https://github.com/VainF/pytorch-msssim)，代码位于 `src/utils/ssim.py`。

### 5. pyiqa 扩展指标（可选）

通过 `--use_pyiqa` 选项可启用 [IQA-PyTorch](https://github.com/chaofengc/IQA-PyTorch) 库的指标，支持 PSNR、SSIM、LPIPS、DISTS 等。对于多通道 3D 数据，同样采用逐 slice 平均策略，并自动将通道数调整为 3（LPIPS/DISTS 要求）。

## 使用方法

### 基本用法

```bash
python image_qua.py \
    --noise_manifest /path/to/noise/epoch_0099/manifest.json \
    --output_dir outputs/image_quality \
    --dataset_config configs/dataset/brats19.yaml
```

### 完整参数

```bash
python image_qua.py \
    --noise_manifest /path/to/manifest.json \  # 噪声 manifest 路径（必需）
    --output_dir outputs/image_quality \        # 输出目录
    --config configs/config.yaml \              # 完整配置文件（可选）
    --dataset_config configs/dataset/brats19.yaml \  # 数据集配置
    --split train \                             # 数据集分割 (train/val/test)
    --metrics psnr ssim \                       # pyiqa 指标列表
    --use_pyiqa \                               # 启用 pyiqa 指标
    --per_slice \                               # 输出逐 slice 详细结果
    --max_samples 50 \                          # 最多评估样本数 (-1 全部)
    --device cpu                                # 计算设备
```

### 批处理

使用 `run_image_qua.sh` 可批量评估多组实验：

```bash
# 编辑脚本中的 EXPERIMENTS 数组添加实验路径
bash run_image_qua.sh

# 自定义参数
DEVICE=cuda:0 MAX_SAMPLES=50 USE_PYIQA=true bash run_image_qua.sh
```

## 输出文件

| 文件 | 格式 | 说明 |
|------|------|------|
| `results.csv` | CSV | 每个样本的指标值，适合 Excel/Pandas 分析 |
| `results.json` | JSON | 包含汇总统计 (mean/std/min/max) 和逐样本结果 |
| `results_per_slice.csv` | CSV | 逐 slice 的 PSNR/SSIM（需 `--per_slice`） |

### results.csv 示例

```csv
Case ID,Num Slices,PSNR,SSIM,Noise_L2,Noise_Linf
BraTS19_001,155,38.5432,0.9821,0.0089,0.0314
BraTS19_002,155,37.9876,0.9798,0.0095,0.0314
...
```

### results.json 示例

```json
{
  "summary": {
    "PSNR": {"mean": 38.12, "std": 0.85, "min": 36.50, "max": 40.20},
    "SSIM": {"mean": 0.9810, "std": 0.0045, "min": 0.9720, "max": 0.9890},
    "num_samples": 200,
    "noise_manifest": "/path/to/manifest.json"
  },
  "per_sample": [...]
}
```

## 依赖

- Python >= 3.8
- PyTorch >= 1.9.0
- NumPy
- OmegaConf
- pyiqa（可选，`pip install pyiqa`）

## 相关文件

- `image_qua.py` — 主评估脚本
- `run_image_qua.sh` — 批处理脚本
- `visualize_unet_noise.py` — 噪声可视化参考
- `src/utils/eval_metrics.py` — PSNR/SSIM 计算实现
- `src/utils/ssim.py` — SSIM 底层实现
- `src/core/ue_artifacts.py` — 噪声数据加载 (UEShardsAccessor)
