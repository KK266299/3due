#!/bin/bash
# FLARE21 随机噪声验证脚本
# 使用随机噪声训练模型，验证噪声对模型的影响

# ============ Gaussian Noise ============
python main.py \
  method=poison_files \
  model.pretrained=false \
  dataset=flare21 \
  task.run_name=victim_random_noise_gaussian_4_255 \
  model=unet \
  model.name=unet \
  task=flare21_seg \
  training.epochs=100 \
  training.optimizer=adam \
  training.optimizers.adam.lr=5e-4 \
  training.gpu_ids=[0] \
  training.batch_size=8 \
  training.eval_batch_size=8 \
  training.data.poison.enabled=true \
  training.data.poison.perturb_type=samplewise \
  training.data.poison.key.type=samplewise \
  training.data.poison.key.from=field \
  training.data.poison.key.field=case_id \
  training.data.poison.source.type=manifest \
  training.data.poison.source.manifest_path=/home/dengzhipeng/data/project/3d_ue/outputs/flare21_ue/random_noise_gaussian_4_255/20260112_000239/noise/manifest.json


# ============ Uniform Noise ============
python main.py \
  method=poison_files \
  model.pretrained=false \
  dataset=flare21 \
  task.run_name=victim_random_noise_uniform_4_255 \
  model=unet \
  model.name=unet \
  task=flare21_seg \
  training.epochs=100 \
  training.optimizer=adam \
  training.optimizers.adam.lr=5e-4 \
  training.gpu_ids=[0] \
  training.batch_size=8 \
  training.eval_batch_size=8 \
  training.data.poison.enabled=true \
  training.data.poison.perturb_type=samplewise \
  training.data.poison.key.type=samplewise \
  training.data.poison.key.from=field \
  training.data.poison.key.field=case_id \
  training.data.poison.source.type=manifest \
  training.data.poison.source.manifest_path=/home/dengzhipeng/data/project/3d_ue/outputs/flare21_ue/random_noise_uniform_4_255/20260112_000711/noise/manifest.json
