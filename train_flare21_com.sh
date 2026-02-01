#!/bin/bash
# 在带噪声数据上训练 UNet++, TransUNet, AttentionUNet 模型 (FLARE21)
MANIFEST_PATH=/home/dengzhipeng/data/project/3d_ue/outputs/flare21_ue/nofreq_zdiv02_logits005/20260204_172349/noise/epoch_0099/manifest.json

# 训练 UNet++
python main.py \
  method=poison_files \
  dataset=flare21 \
  task.run_name=victim_nofreq_zdiv02_logits005_unetpp \
  model=unet_plusplus \
  model.name=unet_plusplus \
  model.pretrained=false \
  task=flare21_seg \
  training.epochs=100 \
  training.optimizer=adam \
  training.optimizers.adam.lr=5e-4 \
  training.gpu_ids=[0] \
  training.batch_size=1 \
  training.eval_batch_size=1 \
  training.data.poison.enabled=true \
  training.data.poison.perturb_type=samplewise \
  training.data.poison.key.type=samplewise \
  training.data.poison.key.from=field \
  training.data.poison.key.field=case_id \
  training.data.poison.source.type=manifest \
  training.data.poison.source.manifest_path=/home/dengzhipeng/data/project/3d_ue/outputs/flare21_ue/nofreq_zdiv02_logits005/20260204_172349/noise/epoch_0099/manifest.json

# 训练 TransUNet
python main.py \
  method=poison_files \
  dataset=flare21 \
  task.run_name=victim_nofreq_zdiv02_logits005_transunet \
  model=trans_unet \
  model.name=trans_unet \
  model.pretrained=false \
  task=flare21_seg \
  training.epochs=100 \
  training.optimizer=adam \
  training.optimizers.adam.lr=5e-4 \
  training.gpu_ids=[0] \
  training.batch_size=2 \
  training.eval_batch_size=2 \
  training.data.poison.enabled=true \
  training.data.poison.perturb_type=samplewise \
  training.data.poison.key.type=samplewise \
  training.data.poison.key.from=field \
  training.data.poison.key.field=case_id \
  training.data.poison.source.type=manifest \
  training.data.poison.source.manifest_path=/home/dengzhipeng/data/project/3d_ue/outputs/flare21_ue/nofreq_zdiv02_logits005/20260204_172349/noise/epoch_0099/manifest.json


# 训练 AttentionUNet
python main.py \
  method=poison_files \
  dataset=flare21 \
  task.run_name=victim_nofreq_zdiv02_logits005_attunet \
  model=attention_unet \
  model.name=attention_unet \
  model.pretrained=false \
  task=flare21_seg \
  training.epochs=100 \
  training.optimizer=adam \
  training.optimizers.adam.lr=5e-4 \
  training.gpu_ids=[0] \
  training.batch_size=2 \
  training.eval_batch_size=2 \
  training.data.poison.enabled=true \
  training.data.poison.perturb_type=samplewise \
  training.data.poison.key.type=samplewise \
  training.data.poison.key.from=field \
  training.data.poison.key.field=case_id \
  training.data.poison.source.type=manifest \
  training.data.poison.source.manifest_path=/home/dengzhipeng/data/project/3d_ue/outputs/flare21_ue/nofreq_zdiv02_logits005/20260204_172349/noise/epoch_0099/manifest.json
