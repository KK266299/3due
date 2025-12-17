#!/bin/bash
# ============================================================================
# UNet Noise Generator Training Script
# ============================================================================
#   噪声: /home/dengzhipeng/data/project/3d_ue/outputs/brats19_ue/unet_noise_exp/<TIMESTAMP>/noise/
#   模型: /home/dengzhipeng/data/project/3d_ue/outputs/brats19_seg/victim_unet_noise/
# ============================================================================

python ue_generate.py \
  dataset=brats19 \
  task.run_name=unet_noise_exp \
  method=unet_noise \
  task=brats19_ue \
  training.epochs=100 \
  training.batch_size=8 \
  training.gpu_ids=[0] \
  ue.key.type=samplewise \
  ue.key.from=field \
  ue.key.field=case_id \
  ue.algorithm.params.epsilon=0.0313725 \
  ue.algorithm.params.step_size=0.0039215 \
  ue.algorithm.params.surrogate_step=20 \
  ue.algorithm.params.noise_step=10 \
  ue.noise_generator.optimizer.lr=1e-4 \
  ue.io.save_from_epoch=0 \
  ue.io.save_every=10

# python main.py \
#   method=poison_files \
#   model.pretrained=false \
#   dataset=brats19 \
#   task.run_name=victim_unet_noise \
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
#   training.data.poison.source.manifest_path=/home/dengzhipeng/data/project/3d_ue/outputs/brats19_ue/unet_noise_exp/<TIMESTAMP>/noise/epoch_0099/manifest.json