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
python ue_generate.py \
    dataset=brats19 \
    task.run_name=unet_noise_slice_disrupt_no_roi_4_255  \
    method=unet_noise_slice \
    task=brats19_ue \
    training.epochs=100 \
    ue.key.type=samplewise \
    ue.key.from=field \
    ue.key.field=case_id \
    training.batch_size=8 \
    training.eval_batch_size=8 \
    training.gpu_ids=[2] \
    ue.algorithm.params.epsilon=0.0156863 \
    ue.io.save_from_epoch=50 \
    ue.io.save_every=10 \
    ue.algorithm.params.lambda_consistency=0.5 \
    ue.algorithm.params.consistency_mode=disrupt  \
    ue.algorithm.params.consistency_type=l2 \
    ue.algorithm.params.surrogate_step=10 \
    ue.algorithm.params.roi_aware=false

# python ue_generate.py \
#     dataset=brats19 \
#     task.run_name=unet_boundary_noise_0_with_2 \
#     method=unet_boundary_noise \
#     task=brats19_ue \
#     training.epochs=100 \
#     training.batch_size=8 \
#     training.eval_batch_size=8 \
#     training.gpu_ids=[3] \
#     ue.key.type=samplewise \
#     ue.key.from=field \
#     ue.key.field=case_id \
#     ue.algorithm.params.boundary_width=1 \
#     ue.algorithm.params.interior_weight=0 \
#     ue.algorithm.params.boundary_weight=1.0 \
#     ue.algorithm.params.surrogate_step=10 \
#     ue.io.save_from_epoch=50 \
#     ue.io.save_every=10