#!/bin/bash
# 在 Clean KiTS19 数据集上训练 UNet 模型，使用更高的肿瘤类别权重 (10.0)

python main.py \
  dataset=kits19 \
  task=kits19_seg \
  task.run_name=clean_w10 \
  model=unet \
  model.name=unet \
  model.pretrained=false \
  training.epochs=150 \
  training.optimizer=adam \
  training.optimizers.adam.lr=5e-4 \
  training.gpu_ids=[0] \
  training.batch_size=8 \
  training.model_save_start=0 \
  training.model_save_freq=10 \
  training.criterion.ce_weight=[0.1,1.0,10.0]
