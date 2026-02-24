# cal_metrics.py 使用说明

## 功能

在干净的测试集上加载训练好的 UNet 模型进行推理，计算 **HD95**（Hausdorff Distance 95th percentile）和 **Dice** 指标。

基于 [MONAI](https://monai.io/) 的 `HausdorffDistanceMetric` 和 `DiceMetric` 实现。

## 支持的数据集

| 数据集 | 输入通道 | 类别数 | 评估区域 |
|--------|----------|--------|----------|
| `brats19_seg` | 4 (T1/T1ce/T2/FLAIR) | 4 | ET, TC, WT |
| `flare21_seg` | 1 (CT) | 5 | Liver, Kidney, Spleen, Pancreas |
| `kits19_seg` | 1 (CT) | 3 | Kidney, Tumor |

### 区域定义

**BraTS19:**
- **ET** (Enhancing Tumour): label == 3
- **TC** (Tumour Core): label ∈ {1, 3}
- **WT** (Whole Tumour): label > 0

**FLARE21:**
- 每个器官独立评估：Liver(1), Kidney(2), Spleen(3), Pancreas(4)

**KiTS19:**
- **Kidney**: label ∈ {1, 2}（kidney 包含 tumor 区域）
- **Tumor**: label == 2

## 命令行参数

| 参数 | 类型 | 必需 | 默认值 | 说明 |
|------|------|------|--------|------|
| `--dataset` | str | 是 | - | 数据集名称：`brats19_seg` / `flare21_seg` / `kits19_seg` |
| `--model_path` | str | 是 | - | 模型 checkpoint 路径 (`best_model.pth`) |
| `--output_csv` | str | 否 | 自动 | 输出 CSV 文件路径，默认保存在 checkpoint 同目录 |
| `--device` | str | 否 | `cuda:0` | 计算设备 |
| `--split` | str | 否 | `test` | 数据集 split（一般用 `test`） |

## 使用示例

### 单个模型评估

```bash
python cal_metrics.py \
    --dataset brats19_seg \
    --model_path /home/dengzhipeng/data/project/3d_ue/outputs/brats19_seg/clean/20251220_144653/checkpoints/checkpoints/best_model.pth \
    --device cuda:0
```

### 指定输出路径

```bash
python cal_metrics.py \
    --dataset flare21_seg \
    --model_path /path/to/flare21/best_model.pth \
    --output_csv results/flare21_metrics.csv \
    --device cuda:1
```

### 批量运行

使用 `run_cal_metrics.sh` 脚本批量评估多个模型：

```bash
bash run_cal_metrics.sh
```

## 输出

### 终端输出

```
============================================================
  Results: brats19_seg (test split)
============================================================

Region               Dice      HD95
-------------------------------------
ET                 0.7856    3.1623
TC                 0.8234    2.4495
WT                 0.9012    1.7321
-------------------------------------
Average            0.8367    2.4480
```

### CSV 文件

1. **Per-sample CSV** (`metrics_<dataset>_<split>.csv`)：每个样本每个区域的 Dice 和 HD95 值
2. **Summary CSV** (`metrics_<dataset>_<split>_summary.csv`)：所有区域的平均 Dice 和 HD95

## 指标说明

### HD95 (Hausdorff Distance 95th Percentile)

HD95 衡量预测分割边界与真实分割边界之间的距离，取第 95 百分位数以减少异常值的影响。

- 单位：体素（voxel）
- 值越小越好
- 当 pred 或 gt 为空时返回 `inf`

### Dice Coefficient

Dice 系数衡量预测分割与真实分割之间的重叠程度。

- 范围：[0, 1]
- 值越大越好
- 当 pred 和 gt 都为空时为 NaN（在汇总时被排除）

## 依赖

- PyTorch
- MONAI (`monai.metrics.HausdorffDistanceMetric`, `monai.metrics.DiceMetric`)
- OmegaConf
- tqdm
- pandas, h5py（数据集加载）
