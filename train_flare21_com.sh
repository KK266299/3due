#!/bin/bash
# 在带噪声数据上训练 UNet++, TransUNet, AttentionUNet 模型 (FLARE21)
MANIFEST_PATH=/home/dengzhipeng/data/project/3d_ue/outputs/flare21_ue/unet_slice_in_out_4_255/20260111_063710/noise/epoch_0099/manifest.json

# 训练 UNet++
python main.py \
  method=poison_files \
  dataset=flare21 \
  task.run_name=victim_noise_unet_grad_4_255_unetpp \
  model=unet_plusplus \
  model.name=unet_plusplus \
  model.pretrained=false \
  task=flare21_seg \
  training.epochs=100 \
  training.optimizer=adam \
  training.optimizers.adam.lr=5e-4 \
  training.gpu_ids=[7] \
  training.batch_size=1 \
  training.eval_batch_size=1 \
  training.data.poison.enabled=true \
  training.data.poison.perturb_type=samplewise \
  training.data.poison.key.type=samplewise \
  training.data.poison.key.from=field \
  training.data.poison.key.field=case_id \
  training.data.poison.source.type=manifest \
  training.data.poison.source.manifest_path=/home/dengzhipeng/data/project/3d_ue/outputs/flare21_ue/unet_grad_noise_change_4_255/20260111_073347/noise/epoch_0099/manifest.json

# 训练 TransUNet
python main.py \
  method=poison_files \
  dataset=flare21 \
  task.run_name=victim_noise_unet_grad_4_255_transunet \
  model=trans_unet \
  model.name=trans_unet \
  model.pretrained=false \
  task=flare21_seg \
  training.epochs=100 \
  training.optimizer=adam \
  training.optimizers.adam.lr=5e-4 \
  training.gpu_ids=[7] \
  training.batch_size=2 \
  training.eval_batch_size=2 \
  training.data.poison.enabled=true \
  training.data.poison.perturb_type=samplewise \
  training.data.poison.key.type=samplewise \
  training.data.poison.key.from=field \
  training.data.poison.key.field=case_id \
  training.data.poison.source.type=manifest \
  training.data.poison.source.manifest_path=/home/dengzhipeng/data/project/3d_ue/outputs/flare21_ue/unet_grad_noise_change_4_255/20260111_073347/noise/epoch_0099/manifest.json

# 训练 AttentionUNet
python main.py \
  method=poison_files \
  dataset=flare21 \
  task.run_name=victim_noise_unet_grad_4_255_attunet \
  model=attention_unet \
  model.name=attention_unet \
  model.pretrained=false \
  task=flare21_seg \
  training.epochs=100 \
  training.optimizer=adam \
  training.optimizers.adam.lr=5e-4 \
  training.gpu_ids=[7] \
  training.batch_size=2 \
  training.eval_batch_size=2 \
  training.data.poison.enabled=true \
  training.data.poison.perturb_type=samplewise \
  training.data.poison.key.type=samplewise \
  training.data.poison.key.from=field \
  training.data.poison.key.field=case_id \
  training.data.poison.source.type=manifest \
  training.data.poison.source.manifest_path=/home/dengzhipeng/data/project/3d_ue/outputs/flare21_ue/unet_grad_noise_change_4_255/20260111_073347/noise/epoch_0099/manifest.json