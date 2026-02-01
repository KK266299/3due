#!/usr/bin/env bash
# Sweep z_diversity_weight and cutoff_lr_scale on 4 GPUs (3 tasks each)
# Usage: bash scripts/sweep_learnable_cutoff.sh

set -euo pipefail

EPOCHS=100
BATCH_SIZE=8

# 12 组参数: z_diversity_weight × cutoff_lr_scale
# GPU 0: zdiv=0.0  × lrs={0.1, 1.0, 10.0}
# GPU 1: zdiv=0.05 × lrs={0.1, 1.0, 10.0}
# GPU 2: zdiv=0.1  × lrs={0.1, 1.0, 10.0}
# GPU 3: zdiv=0.2  × lrs={0.1, 1.0, 10.0}

run_task() {
  local GPU=$1 Z_DIV=$2 LR_S=$3
  local RUN_NAME="learnable_zdiv${Z_DIV}_lrs${LR_S}"
  echo "[GPU ${GPU}] ===== ${RUN_NAME} ====="

  CUDA_VISIBLE_DEVICES=${GPU} python ue_generate.py \
    dataset=flare21 \
    task=flare21_ue \
    task.run_name="${RUN_NAME}" \
    method=noise_slice_frequence_learnable \
    training.epochs=${EPOCHS} \
    training.batch_size=${BATCH_SIZE} \
    training.gpu_ids="[0]" \
    ue.key.type=samplewise \
    ue.key.from=field \
    ue.key.field=case_id \
    ue.algorithm.params.epsilon=0.0156863 \
    ue.algorithm.params.noise_step=1 \
    ue.algorithm.params.surrogate_step=10 \
    ue.algorithm.params.roi_aware=true \
    ue.algorithm.params.soft_edge=false \
    ue.algorithm.params.cutoff_lr_scale="${LR_S}" \
    ue.algorithm.params.z_diversity_weight="${Z_DIV}" \
    ue.io.save_from_epoch=50 \
    ue.io.save_every=10 \
    ue.surrogates.s_seg.in_channels=1 \
    ue.surrogates.s_seg.num_classes=5

  echo "[GPU ${GPU}] ===== DONE: ${RUN_NAME} ====="
}

run_gpu() {
  local GPU=$1 Z_DIV=$2
  for LR_S in 0.1 1.0 10.0; do
    run_task "${GPU}" "${Z_DIV}" "${LR_S}"
  done
}

# 4 GPUs 并行，每个 GPU 串行跑 3 个任务
run_gpu 0 0.0  &
run_gpu 1 0.05 &
run_gpu 2 0.1  &
run_gpu 3 0.2  &

wait
echo "All sweeps finished."
