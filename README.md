# Reid UE 项目架构文档

## 概述

Reid UE 是一个基于深度学习的医学影像处理框架，主要针对 MIMIC-CXR 胸部X光数据集进行多标签分类（Classification）和重识别（ReID）任务。项目采用模块化设计，使用 Hydra 配置管理系统，支持灵活的组件配置和实验管理。

## 核心特性

- **统一框架**: 支持分类和重识别两种任务模式
- **模块化设计**: 组件可插拔，易于扩展和维护  
- **配置驱动**: 基于 Hydra 的声明式配置管理
- **组件注册**: 动态组件注册机制，支持运行时组件选择
- **实验管理**: 完整的实验生命周期管理和结果追踪

## 项目结构

```
reid_ue/
├── main.py                    # 程序入口点
├── cls_tasks.py              # 分类任务专用入口
├── train.sh                  # 训练脚本
├── requirements.txt          # 依赖管理
├── configs/                  # 配置文件目录
│   ├── config.yaml           # 主配置文件
│   ├── _global_patches/      # 全局配置补丁
│   ├── dataset/              # 数据集配置
│   ├── experiment/           # 实验配置
│   ├── method/               # 方法配置
│   ├── model/                # 模型配置
│   ├── task/                 # 任务配置
│   └── training/             # 训练配置
├── src/                      # 源代码目录
│   ├── core/                 # 核心组件
│   ├── datasets/             # 数据集实现
│   ├── evaluation/           # 评估策略
│   ├── models/               # 模型实现
│   └── utils/                # 工具函数
├── scripts/                  # 脚本目录
└── tests/                    # 测试代码
```

## 核心架构组件

### 1. 组件注册系统 (`src/registry.py`)

采用注册表模式管理所有组件，支持动态组件发现和实例化：

```python
# 全局注册器
MODELS = Registry("models")
DATASETS = Registry("datasets") 
DATASET_BUILDERS = Registry("dataset_builders")
EVALUATION_STRATEGIES = Registry("evaluation_strategies")
```

**支持的组件类型**:
- 模型 (Models)
- 数据集 (Datasets)
- 数据集构建器 (Dataset Builders)
- 评估策略 (Evaluation Strategies)
- 优化器 (Optimizers)
- 学习率调度器 (Schedulers)
- 损失函数 (Criteria)

### 2. 实验管理器 (`src/core/experiment_manager.py`)

负责协调实验的各个组件，管理完整的实验生命周期：

**核心职责**:
- 组件初始化和配置
- 模型、数据、损失函数、优化器设置
- 训练器配置和执行
- 分布式训练支持

**组件设置流程**:
```python
manager.setup_model()      # 模型初始化
manager.setup_data()       # 数据加载器设置
manager.setup_criterion()  # 损失函数配置
manager.setup_optimizer()  # 优化器配置
manager.setup_scheduler()  # 学习率调度器配置
manager.setup_trainer()    # 训练器配置
manager.train(epochs)      # 执行训练
```

### 3. 训练器架构 (`src/core/trainers/`)

采用继承和组合模式的训练器架构：

- **TrainerBase**: 抽象基类，定义训练器接口和通用逻辑
- **CLSTrainer**: 分类任务专用训练器
- **Hook系统**: 支持训练过程中的扩展点

**训练步骤**:
1. **预处理阶段**: 梯度清零，数据转移到设备
2. **任务特定阶段**: 根据任务类型执行专用逻辑
3. **后处理阶段**: 反向传播，参数更新

### 4. 数据管理 (`src/datasets/`)

**MIMIC-CXR 数据集支持**:
- 多标签分类任务（13个疾病类别）
- 重识别任务
- 灵活的数据增强管道
- 自动数据预处理和加载

**疾病标签**:
```python
DISEASE_LABELS = [
    'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity',
    'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis',
    'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture', 
    'Support Devices'
]
```

### 5. 模型架构 (`src/models/`)

支持多种深度学习模型：

- **ResNet**: ResNet18/34/50/101/152 系列
- **DenseNet**: DenseNet121/169/201 系列  
- **Vision Transformer**: ViT 系列

**模型特性**:
- 预训练权重支持
- 灵活的分类头配置
- Dropout 正则化
- 特征提取和分类输出

### 6. 配置管理系统

基于 Hydra 的层次化配置管理：

**配置层次结构**:
- **数据集配置**: 数据路径、预处理参数
- **模型配置**: 模型架构、预训练权重
- **训练配置**: 学习率、批大小、轮数
- **任务配置**: 任务特定参数
- **实验配置**: 实验名称、保存路径

**配置组合示例**:
```yaml
defaults:
  - dataset: mimic_cxr      # 使用 MIMIC-CXR 数据集
  - model: resnet           # 使用 ResNet 模型
  - method: base            # 基础方法
  - task: mimic_cxr_cls     # 分类任务
  - training: default       # 默认训练配置
  - experiment: default     # 默认实验配置
```

## 数据流程

### 1. 训练数据流
```
原始DICOM图像 → CSV标注文件 → 数据预处理 → 数据增强 → 模型训练 → 损失计算 → 参数更新
```

### 2. 评估数据流  
```
测试图像 → 模型推理 → 预测结果 → 评估指标计算 → 性能报告生成
```

## 扩展指南

### 添加新模型
1. 在 `src/models/` 下实现模型类
2. 使用 `@register_model()` 装饰器注册
3. 在 `configs/model/` 下添加配置文件

### 添加新数据集
1. 在 `src/datasets/` 下实现数据集类
2. 实现对应的数据集构建器
3. 注册数据集和构建器
4. 添加相应配置文件

### 添加新任务
1. 在训练器中实现任务特定逻辑
2. 创建评估策略
3. 配置任务参数
4. 注册相关组件

## 依赖环境

**核心依赖**:
- PyTorch >= 1.9.0
- Torchvision >= 0.10.0
- Hydra-core >= 1.3.2
- Albumentations >= 1.3.0
- NumPy >= 1.21.0

**开发环境**:
- Python 3.8+
- CUDA 支持（推荐）
- 足够的 GPU 内存用于批处理

## 使用示例

### 基础训练
```bash
python main.py
```

### 配置覆盖
```bash
python main.py model=densenet training.epochs=50 experiment.seed=42
```

### 分类任务
```bash
python cls_tasks.py dataset=mimic_cxr model=resnet task=mimic_cxr_cls
```

## 项目优势

1. **高度模块化**: 组件间解耦，易于维护和扩展
2. **配置驱动**: 声明式配置，实验重现性好
3. **类型安全**: 使用 DictConfig 确保配置类型安全
4. **可扩展性**: 注册机制支持动态组件添加
5. **实验友好**: 完整的实验管理和结果追踪

这个架构为医学影像分析提供了一个灵活、可扩展的深度学习框架，支持多种任务和模型的快速实验和部署。