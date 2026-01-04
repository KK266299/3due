#!/bin/bash
# 在 Clean 数据上训练 UNet++, TransUNet, AttentionUNet 模型

# 训练 UNet++
python main.py \
  dataset=brats19 \
  task.run_name=clean_unetpp \
  model=unet_plusplus \
  model.name=unet_plusplus \
  model.pretrained=false \
  task=brats19_seg \
  training.epochs=100 \
  training.optimizer=adam \
  training.optimizers.adam.lr=5e-4 \
  training.gpu_ids=[3] \
  training.batch_size=1 \
  training.eval_batch_size=1

# 训练 TransUNet
python main.py \
  dataset=brats19 \
  task.run_name=clean_transunet \
  model=trans_unet \
  model.name=trans_unet \
  model.pretrained=false \
  task=brats19_seg \
  training.epochs=100 \
  training.optimizer=adam \
  training.optimizers.adam.lr=5e-4 \
  training.gpu_ids=[3] \
  training.batch_size=2 \
  training.eval_batch_size=2

# 训练 AttentionUNet
python main.py \
  dataset=brats19 \
  task.run_name=clean_attunet \
  model=attention_unet \
  model.name=attention_unet \
  model.pretrained=false \
  task=brats19_seg \
  training.epochs=100 \
  training.optimizer=adam \
  training.optimizers.adam.lr=5e-4 \
  training.gpu_ids=[3] \
  training.batch_size=2 \
  training.eval_batch_size=2