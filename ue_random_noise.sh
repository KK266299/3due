#!/bin/bash
# ============================================================================
# Random Noise 3D - Training-free UE Generator
# ============================================================================
# This script generates random noise for 3D medical imaging (BraTS19).
# Training-free: No surrogate model training required.
#
# Supported noise modes:
#   - gaussian:    Gaussian noise, L∞ normalized to [-eps, eps]
#   - uniform:     Uniform noise U(-eps, eps)
#   - rademacher:  Sign noise, each pixel is +eps or -eps
#   - saltpepper:  Sparse pulse noise with probability p
#   - sparse:      Sparse additive noise with sparsity q
#
# Output:
#   - Noise files: ${task.save_dir}/${task.run_name}/<TIMESTAMP>/noise/
#   - Manifest:    ${task.save_dir}/${task.run_name}/<TIMESTAMP>/noise/manifest.json
# ============================================================================

# ============ Gaussian Noise ============
# manifest: outputs/brats19_ue/random_noise_gaussian/<TIMESTAMP>/noise/manifest.json
python ue_generate.py \
    dataset=brats19 \
    task.run_name=random_noise_gaussian \
    method=random_noise_3d \
    task=brats19_ue \
    ue.key.type=samplewise \
    ue.key.from=field \
    ue.key.field=case_id \
    ue.algorithm.params.mode=gaussian \
    ue.algorithm.params.epsilon=0.0313725 \
    ue.algorithm.params.seed=42

# ============ Uniform Noise ============
# manifest: outputs/brats19_ue/random_noise_uniform/<TIMESTAMP>/noise/manifest.json
python ue_generate.py \
    dataset=brats19 \
    task.run_name=random_noise_uniform \
    method=random_noise_3d \
    task=brats19_ue \
    ue.key.type=samplewise \
    ue.key.from=field \
    ue.key.field=case_id \
    ue.algorithm.params.mode=uniform \
    ue.algorithm.params.epsilon=0.0313725 \
    ue.algorithm.params.seed=42
