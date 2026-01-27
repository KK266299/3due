#!/bin/bash
# ============================================================================
# PUE (Provably Unlearnable Examples) Noise Generator Training Script
# ============================================================================
#   Method: Provably Unlearnable Data Examples (NDSS 2025)
#   Authors: Derui Wang, Minhui Xue, Bo Li, Seyit Camtepe, Liming Zhu
#   Paper: https://www.ndss-symposium.org/wp-content/uploads/2025-886-paper.pdf
#   Code: https://github.com/NeuralSec/certified-data-learnability
#
#   噪声输出: ${task.save_dir}/${task.run_name}/<TIMESTAMP>/noise/
#   manifest: epoch_XXXX/manifest.json
# ============================================================================

python ue_generate.py \
    dataset=brats19 \
    task.run_name=pue_noise_rwp \
    method=pue \
    task=brats19_ue \
    training.epochs=100 \
    ue.key.type=samplewise \
    ue.key.from=field \
    ue.key.field=case_id \
    ue.algorithm.params.step_size=5e-3 \
    training.batch_size=8 \
    training.gpu_ids=[0] \
    ue.algorithm.params.surrogate_step=10 \
    ue.rwp.enabled=true \
    ue.rwp.sigma=0.05 \
    ue.rwp.U_train=4 \
    ue.rwp.U_noise=8 \
    ue.io.save_from_epoch=0 \
    ue.io.save_every=10
