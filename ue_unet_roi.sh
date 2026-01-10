#!/bin/bash
# ============================================================================
# UNet ROI Noise Generator Training Script
# ============================================================================
# ROI 噪声特点:
#   - 噪声仅在 label > 0 的区域 (肿瘤/病灶)
#   - 背景区域噪声为 0
#
#   噪声: /home/dengzhipeng/data/project/3d_ue/outputs/brats19_ue/unet_roi_noise_exp/<TIMESTAMP>/noise/
#   模型: /home/dengzhipeng/data/project/3d_ue/outputs/brats19_seg/victim_unet_roi_noise/
# ============================================================================

python ue_generate.py \
    dataset=kits19 \
    task.run_name=unet_roi_noise_4_255 \
    method=unet_roi_noise \
    task=kits19_ue \
    training.epochs=100 \
    ue.key.type=samplewise \
    ue.key.from=field \
    ue.key.field=case_id \
    training.batch_size=8 \
    training.gpu_ids=[0] \
    ue.algorithm.params.epsilon=0.0156863 \
    ue.algorithm.params.surrogate_step=10 \
    ue.io.save_from_epoch=50 \
    ue.io.save_every=10

# python main.py \
#   method=poison_files \
#   model.pretrained=false \
#   dataset=brats19 \
#   task.run_name=victim_unet_roi_noise \
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
#   training.data.poison.source.manifest_path=/home/dengzhipeng/data/project/3d_ue/outputs/brats19_ue/unet_roi_noise_exp/<TIMESTAMP>/noise/epoch_0099/manifest.json