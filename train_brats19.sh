python main.py \
  method=poison_files \
  model.pretrained=false \
  dataset=brats19 \
  task.run_name=victim_minmin_20251217 \
  model=unet \
  model.name=unet \
  task=brats19_seg \
  training.epochs=100 \
  training.optimizer=adam \
  training.optimizers.adam.lr=5e-4 \
  training.gpu_ids=[0] \
  training.batch_size=1 \
  training.data.poison.perturb_type=samplewise \
  training.data.poison.key.type=samplewise \
  training.data.poison.key.from=field \
  training.data.poison.key.field=case_id \
  training.data.poison.source.type=manifest \
  training.data.poison.source.manifest_path=/home/dengzhipeng/data/project/3d_ue/outputs/brats19_ue/minmin_noise_step_5e-3/20251216_195958/noise/epoch_0099/manifest.json



# python main.py \
#   model.pretrained=true \
#   dataset=brats19 \
#   task.run_name=clean \
#   model=unet \
#   model.name=unet \
#   task=brats19_seg \
#   training.epochs=100 \
#   training.optimizer=adam \
#   training.optimizers.adam.lr=5e-4 \
#   training.gpu_ids=[0] \
#   training.batch_size=16
# python main.py \
#   model.pretrained=false \
#   model.embed_dim=64 \
#   dataset=grape \
#   task.run_name=classwise_generator_poisoned \
#   model=efficientnet \
#   model.name=efficientnet_b2 \
#   method=poison_files \
#   task=grape_reid \
#   training.data.poison.enabled=true \
#   training.data.poison.source.manifest_path=/home/dengzhipeng/data/project/reid_ue/outputs/grape_ue/generator_classwise_lr_1e-5/20250910_045728/noise/epoch_0049/manifest.json \
#   training.data.poison.perturb_type=samplewise \
#   training.epochs=100 \
#   training.optimizer=adam \
#   training.optimizers.adam.lr=1e-4 \
#   training.gpu_ids=[0] \
#   training.batch_size=64