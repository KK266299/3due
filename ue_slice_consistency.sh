#!/bin/bash
# Inter-Slice Consistency Attack Training Script
# This script generates unlearnable examples using inter-slice consistency constraints
#
# Usage:
#   ./ue_slice_consistency.sh [GPU_ID] [EPOCHS] [MODE]
#
# Arguments:
#   GPU_ID  - GPU device ID (default: 0)
#   EPOCHS  - Number of training epochs (default: 100)
#   MODE    - Consistency mode: smooth, disrupt, periodic (default: disrupt)
#
# Examples:
#   ./ue_slice_consistency.sh 0 100 disrupt   # Recommended: disrupt mode
#   ./ue_slice_consistency.sh 0 100 smooth    # Smooth mode
#   ./ue_slice_consistency.sh 0 100 periodic  # Periodic mode

GPU_ID=${1:-0}
EPOCHS=${2:-100}
MODE=${3:-disrupt}

echo "=========================================="
echo "Inter-Slice Consistency Attack"
echo "GPU: ${GPU_ID}, Epochs: ${EPOCHS}"
echo "Mode: ${MODE}"
echo "=========================================="

python ue_generate.py \
    dataset=brats19 \
    task.run_name=unet_noise_slice_${MODE} \
    method=unet_noise_slice \
    task=brats19_ue \
    training.epochs=${EPOCHS} \
    ue.key.type=samplewise \
    ue.key.from=field \
    ue.key.field=case_id \
    training.batch_size=4 \
    training.gpu_ids=[${GPU_ID}] \
    ue.io.save_from_epoch=50 \
    ue.io.save_every=10 \
    ue.algorithm.params.lambda_consistency=0.5 \
    ue.algorithm.params.consistency_mode=${MODE} \
    ue.algorithm.params.consistency_type=l2 \
    ue.algorithm.params.roi_aware=true

echo "=========================================="
echo "Training completed!"
echo "=========================================="