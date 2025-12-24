#!/bin/bash
# ============================================================================
# TAP (Adversarial Examples Make Strong Poisons) Noise Generator
# ============================================================================
# TAP 方法通过 PGD 梯度上升（最大化 loss）生成对抗性噪声
# 与 min-min 方法相反（min-min 使用梯度下降最小化 loss）
#
# 支持两种模式：
#   1. 预训练模式 (有 weights_path): 加载预训练权重，冻结模型，不训练
#   2. 在线训练模式 (无 weights_path): 边训练 surrogate 边生成噪声
#
# 输出目录:
#   噪声: outputs/brats19_ue/<run_name>/<TIMESTAMP>/noise/
# ============================================================================

# TAP 噪声生成 - 使用预训练权重模式
python ue_generate.py \
  dataset=brats19 \
  task.run_name=tap_pretrained_step_5e-3 \
  method=tap \
  task=brats19_ue \
  training.epochs=100 \
  ue.key.type=samplewise \
  ue.key.from=field \
  ue.key.field=case_id \
  ue.algorithm.params.step_size=5e-3 \
  training.batch_size=8 \
  training.gpu_ids=[1] \
  ue.algorithm.params.surrogate_step=2 \
  ue.io.save_from_epoch=0 \
  ue.io.save_every=10 \
  ue.surrogates.s_seg.weights_path=/home/dengzhipeng/data/project/3d_ue/outputs/brats19_seg/clean/20251213_211122/checkpoints/checkpoints/best_model.pth