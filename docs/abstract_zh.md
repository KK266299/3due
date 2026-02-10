# 论文摘要（中文版）

## 摘要

公开的 3D 医学图像分割（MIS）数据集的广泛共享极大地推动了临床研究的发展，但也引发了关于未经授权训练 AI 模型以及患者隐私泄露的严重隐忧。不可学习样本（Unlearnable Examples, UE）作为一种数据层面的主动防御手段，可通过添加不可感知的扰动阻止深度神经网络从受保护数据中有效学习。然而，现有 UE 方法主要面向 2D 自然图像分类任务，未能针对 3D 医学图像分割的独特结构进行设计——3D 医学图像具有显著的各向异性特征，且相邻层间存在强烈的空间连续性（层间一致性），这正是 3D 分割模型赖以学习的核心先验。为此，我们提出了一种面向 3D 医学图像分割的不可学习样本生成方法 UMed。UMed 利用基于 UNet 的噪声生成器产生自适应扰动，并通过分割标签的 ROI 掩码将噪声约束在目标区域及其邻域内，从而以最小的扰动预算实现高效保护。在损失设计上，UMed 通过频域引导的对抗损失框架实现两个核心攻击目标：（1）通过 Z-diversity Loss 在噪声层面显式最大化相邻层的频谱差异，从源头上破坏 3D 模型所依赖的层间一致性；（2）通过 Logits Divergence Loss 最大化干净图像与扰动图像之间的分割预测差异，迫使模型在受保护数据上产生完全偏离的输出。实验表明，UMed 仅需极小的扰动幅度（ε = 4/255），即可使 3D 分割模型的性能从 DSC 87% 大幅下降至 7%，同时保持高度的视觉不可感知性。

**关键词**：不可学习样本 · 3D 医学图像分割 · 层间一致性破坏 · 频域对抗损失 · 数据隐私保护

---

# Abstract (English)

## Abstract

The widespread sharing of public 3D medical image segmentation (MIS) datasets has greatly advanced clinical research, yet it also raises serious concerns about unauthorized AI model training and patient privacy. Unlearnable Examples (UEs) offer a proactive data-level defense by adding imperceptible perturbations that prevent deep neural networks from learning effectively on protected data. However, existing UE methods are primarily designed for 2D natural image classification and fail to account for the unique structure of 3D medical images — notably their inherent anisotropy and the strong spatial continuity between adjacent slices (inter-slice consistency), which serves as a critical prior exploited by 3D segmentation models. To address this gap, we propose UMed, an unlearnable example generation method tailored for 3D medical image segmentation. UMed employs a UNet-based noise generator to produce adaptive perturbations, which are constrained to the target region and its vicinity through ROI masks derived from segmentation labels, thereby achieving effective protection with minimal perturbation budget. For the loss design, UMed introduces a frequency-guided adversarial loss framework targeting two complementary attack objectives: (1) a Z-diversity loss that explicitly maximizes the spectral dissimilarity between adjacent slices at the noise level, directly disrupting the inter-slice consistency that 3D models rely on; and (2) a logits divergence loss that maximizes the segmentation prediction discrepancy between clean and perturbed images, forcing the model to produce entirely deviated outputs on protected data. Experiments demonstrate that UMed, with only a minimal perturbation magnitude (ε = 4/255), degrades the performance of 3D segmentation models from 87% DSC to 7% while maintaining high visual imperceptibility.

**Keywords**: Unlearnable Examples · 3D Medical Image Segmentation · Inter-slice Consistency Disruption · Frequency-guided Adversarial Loss · Data Privacy Protection
