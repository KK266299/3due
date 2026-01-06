# 训练 Clean UNet 模型，保存多个检查点供 SEP 使用
python main.py \
  dataset=kits19 \
  task.run_name=clean \
  model=unet \
  model.name=unet \
  model.pretrained=false \
  task=kits19_seg \
  training.epochs=100 \
  training.optimizer=adam \
  training.optimizers.adam.lr=5e-4 \
  training.gpu_ids=[6] \
  training.batch_size=8 \
  training.model_save_start=0 \
  training.model_save_freq=10
