# UMed3D 论文摘要

## 摘要（中文）

公开 3D 医学图像分割（MIS）数据集的广泛共享在推动临床研究的同时，也带来了未经授权训练 AI 模型和患者隐私泄露的风险。不可学习样本（Unlearnable Examples, UE）通过添加不可感知的扰动阻止模型从受保护数据中有效学习，但现有方法主要面向 2D 自然图像分类，忽略了 3D 医学图像的各向异性及层间一致性——这一 3D 分割模型的核心学习先验。为此，我们提出 UMed3D，一种面向 3D 医学图像分割的不可学习样本生成方法。UMed3D 采用 UNet 噪声生成器产生 ROI 约束的自适应扰动，并设计频域引导的对抗损失：Z-diversity Loss 最大化相邻层噪声的频谱差异以破坏层间一致性，Logits Divergence Loss 最大化干净与扰动图像间的分割预测差异。实验表明，UMed3D 以极小扰动（ε = 4/255）将 3D 分割模型性能从 DSC 87% 降至 7%，同时保持视觉不可感知性。

**关键词**：不可学习样本 · 3D 医学图像分割 · 层间一致性破坏 · 频域对抗损失 · 数据隐私保护

---

## Abstract (English)

The widespread sharing of public 3D medical image segmentation (MIS) datasets has advanced clinical research but also raised concerns about unauthorized AI training and patient privacy. Unlearnable Examples (UEs) prevent models from learning on protected data by adding imperceptible perturbations, yet existing methods target 2D natural image classification and overlook the anisotropy and inter-slice consistency inherent to 3D medical images — the critical prior exploited by 3D segmentation models. We propose UMed3D, an unlearnable example generation method for 3D medical image segmentation. UMed3D employs a UNet-based noise generator to produce ROI-constrained adaptive perturbations and introduces a frequency-guided adversarial loss framework: a Z-diversity loss that maximizes spectral dissimilarity between adjacent slices to disrupt inter-slice consistency, and a logits divergence loss that maximizes segmentation prediction discrepancy between clean and perturbed images. Experiments show that UMed3D, with minimal perturbation (ε = 4/255), degrades 3D segmentation performance from 87% to 7% DSC while maintaining visual imperceptibility.

**Keywords**: Unlearnable Examples · 3D Medical Image Segmentation · Inter-slice Consistency Disruption · Frequency-guided Adversarial Loss · Data Privacy Protection
