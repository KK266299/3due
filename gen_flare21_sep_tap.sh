#!/bin/bash
# ============================================================================
# FLARE21 LSP/SEP/TAP 噪声生成脚本 (4/255 epsilon)
# ============================================================================
CKPT_DIR=/home/dengzhipeng/data/project/3d_ue/outputs/flare21_seg/clean/20260224_143824/checkpoints/checkpoints
# ============================================================================
# LSP (Linear Separable Perturbation) - Training-Free
# ============================================================================
# python ue_generate.py \
#   dataset=flare21 \
#   task.run_name=lsp_noise_4_255 \
#   method=lsp \
#   task=flare21_ue \
#   ue.key.type=samplewise \
#   ue.key.from=field \
#   ue.key.field=case_id \
#   ue.algorithm.params.epsilon=0.0156863 \
#   ue.algorithm.params.noise_frame_size=32 \
#   ue.algorithm.params.seed=0 \
#   ue.algorithm.params.class_sep=10.0 \
#   ue.algorithm.params.roi_mode=binary \
#   ue.algorithm.params.num_labels=5
# ============================================================================
# SEP (Shortcut-based Error-minimizing Perturbation) - GPU 4
# ============================================================================
python ue_generate.py \
  dataset=flare21 \
  task.run_name=sep_noise_4_255 \
  method=sep \
  task=flare21_ue \
  training.epochs=1 \
  training.batch_size=4 \
  training.gpu_ids=[4] \
  ue.key.type=samplewise \
  ue.key.from=field \
  ue.key.field=case_id \
  ue.algorithm.params.epsilon=0.0156863 \
  ue.algorithm.params.step_size=0.00196078 \
  ue.algorithm.params.surrogate_step=10 \
  ue.algorithm.params.noise_step=1 \
  ue.algorithm.params.svrg_M=5 \
  ue.io.save_from_epoch=0 \
  ue.io.save_every=1 \
  ue.surrogates.s_seg.in_channels=1 \
  ue.surrogates.s_seg.num_classes=5 \
  "ue.sep.ckpt_paths=[${CKPT_DIR}/checkpoint_epoch_9.pth,${CKPT_DIR}/checkpoint_epoch_19.pth,${CKPT_DIR}/checkpoint_epoch_29.pth,${CKPT_DIR}/checkpoint_epoch_39.pth,${CKPT_DIR}/checkpoint_epoch_49.pth,${CKPT_DIR}/checkpoint_epoch_59.pth,${CKPT_DIR}/checkpoint_epoch_69.pth,${CKPT_DIR}/checkpoint_epoch_79.pth,${CKPT_DIR}/checkpoint_epoch_89.pth,${CKPT_DIR}/checkpoint_epoch_99.pth]" &

# ============================================================================
# TAP (Adversarial Examples Make Strong Poisons) - GPU 4
# ============================================================================
python ue_generate.py \
  dataset=flare21 \
  task.run_name=tap_noise_4_255 \
  method=tap \
  task=flare21_ue \
  training.epochs=100 \
  ue.key.type=samplewise \
  ue.key.from=field \
  ue.key.field=case_id \
  ue.algorithm.params.epsilon=0.0156863 \
  ue.algorithm.params.step_size=2.5e-3 \
  training.batch_size=8 \
  training.gpu_ids=[4] \
  ue.algorithm.params.surrogate_step=10 \
  ue.io.save_from_epoch=0 \
  ue.io.save_every=10 \
  ue.surrogates.s_seg.in_channels=1 \
  ue.surrogates.s_seg.num_classes=5 \
  ue.surrogates.s_seg.weights_path=${CKPT_DIR}/best_model.pth &

wait
bash train_flare21_lsp_sep_tap.sh
