#!/bin/bash
# Slice-by-Slice UNet Noise Generation Script
# This script generates unlearnable examples using a 2D UNet per-slice noise generator
#
# Usage:
#   ./ue_noise_slice.sh
#
# The 2D UNet processes each depth slice independently,
# generating per-slice noise that is assembled into a 3D volume.

python ue_generate.py \
    dataset=brats19 \
    task.run_name=unet_noise_slice \
    method=unet_noise_slice \
    task=brats19_ue \
    training.epochs=100 \
    ue.key.type=samplewise \
    ue.key.from=field \
    ue.key.field=case_id \
    training.batch_size=8 \
    training.eval_batch_size=8 \
    training.gpu_ids=[1] \
    ue.algorithm.params.surrogate_step=10 \
    ue.io.save_from_epoch=50 \
    ue.io.save_every=10
