#!/bin/bash
# ============================================================================
# LSP (Linear Separable Perturbation) Noise Generator Script
# ============================================================================
# Training-Free Implementation based on "Availability Attacks Create Shortcuts"
#
# Output directory structure:
#   outputs/brats19_ue/lsp_noise/<TIMESTAMP>/ue/lsp/
#   ├── manifest.json         # Manifest file with key mappings
#   └── files/                 # Quantized noise files (int8)
#       ├── 00000000.pt
#       ├── 00000001.pt
#       └── ...
#
# The manifest.json format is identical to minmin output, enabling seamless
# integration with existing poisoning pipelines.
# ============================================================================

python ue_generate.py \
  dataset=brats19 \
  task.run_name=lsp_noise \
  method=lsp \
  task=brats19_ue \
  ue.key.type=samplewise \
  ue.key.from=field \
  ue.key.field=case_id \
  ue.algorithm.params.epsilon=0.0313725 \
  ue.algorithm.params.noise_frame_size=32 \
  ue.algorithm.params.seed=0 \
  ue.algorithm.params.class_sep=10.0 \
  ue.algorithm.params.roi_mode=binary \
  ue.algorithm.params.num_labels=4

# ============================================================================
# Alternative: Run victim training with LSP noise
# ============================================================================
# After generating noise, you can train a victim model using the poisoned data:
#
# python main.py \
#   method=poison_files \
#   model.pretrained=false \
#   dataset=brats19 \
#   task.run_name=victim_lsp \
#   model=unet \
#   model.name=unet \
#   task=brats19_seg \
#   training.epochs=100 \
#   training.optimizer=adam \
#   training.optimizers.adam.lr=5e-4 \
#   training.gpu_ids=[0] \
#   training.batch_size=4 \
#   training.data.poison.perturb_type=samplewise \
#   training.data.poison.key.type=samplewise \
#   training.data.poison.key.from=field \
#   training.data.poison.key.field=case_id \
#   training.data.poison.source.type=manifest \
#   training.data.poison.source.manifest_path=<PATH_TO_MANIFEST>/manifest.json
# ============================================================================