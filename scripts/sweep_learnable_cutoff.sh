#!/usr/bin/env bash
# Sweep learnable cutoff parameters
# Usage: bash scripts/sweep_learnable_cutoff.sh [GPU_ID]

set -euo pipefail

GPU=${1:-0}

# ── 参数网格 ──
Z_CUTOFFS=(0.0 0.05 0.1 0.15 0.2)
XY_CUTOFFS=(0.0 0.1 0.2 0.3 0.4)
Z_DIV_WEIGHTS=(0.0 0.01 0.1)
CUTOFF_LR_SCALES=(0.1 1.0 10.0)

EPOCHS=100
BATCH_SIZE=8

for z_cut in "${Z_CUTOFFS[@]}"; do
  for xy_cut in "${XY_CUTOFFS[@]}"; do
    for z_div in "${Z_DIV_WEIGHTS[@]}"; do
      for lr_s in "${CUTOFF_LR_SCALES[@]}"; do

        RUN_NAME="learnable_z${z_cut}_xy${xy_cut}_zdiv${z_div}_lrs${lr_s}"
        echo "===== ${RUN_NAME} ====="

        python ue_generate.py \
          dataset=flare21 \
          task=flare21_ue \
          task.run_name="${RUN_NAME}" \
          method=noise_slice_frequence_learnable \
          training.epochs=${EPOCHS} \
          training.batch_size=${BATCH_SIZE} \
          training.gpu_ids="[${GPU}]" \
          ue.key.type=samplewise \
          ue.key.from=field \
          ue.key.field=case_id \
          ue.algorithm.params.epsilon=0.0156863 \
          ue.algorithm.params.noise_step=1 \
          ue.algorithm.params.surrogate_step=10 \
          ue.algorithm.params.roi_aware=true \
          ue.algorithm.params.soft_edge=false \
          ue.algorithm.params.z_cutoff_low="${z_cut}" \
          ue.algorithm.params.xy_cutoff_high="${xy_cut}" \
          ue.algorithm.params.cutoff_lr_scale="${lr_s}" \
          ue.algorithm.params.z_diversity_weight="${z_div}" \
          ue.io.save_from_epoch=50 \
          ue.io.save_every=10 \
          ue.surrogates.s_seg.in_channels=1 \
          ue.surrogates.s_seg.num_classes=5

        echo "===== DONE: ${RUN_NAME} ====="
        echo ""

      done
    done
  done
done

echo "All sweeps finished."
