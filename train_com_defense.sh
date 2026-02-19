#!/bin/bash
# FLARE21 Defense Augmentation 消融实验
# 4 种防御方法分布在 GPU 4/5/6 上并行运行
#   GPU 4: gaussian_blur, gamma          (串行)
#   GPU 5: low_resolution                (单个)
#   GPU 6: random_affine                 (单个)

MANIFEST_PATH=/home/dengzhipeng/data/project/3d_ue/outputs/flare21_ue/nofreq_zdiv02_logits005/20260204_172349/noise/epoch_0099/manifest.json

mkdir -p logs

# ==================== GPU 4 (串行: gaussian_blur + gamma) ====================
(
    echo "[GPU 4] 开始实验 1/2: defense_gaussian_blur"
    python main.py \
        method=poison_files \
        model.pretrained=false \
        dataset=flare21 \
        task.run_name=victim_defense_gaussian_blur \
        model=unet \
        model.name=unet \
        task=flare21_seg \
        training.epochs=100 \
        training.optimizer=adam \
        training.optimizers.adam.lr=5e-4 \
        training.gpu_ids=[4] \
        training.batch_size=2 \
        training.eval_batch_size=2 \
        training.data.poison.enabled=true \
        training.data.poison.perturb_type=samplewise \
        training.data.poison.key.type=samplewise \
        training.data.poison.key.from=field \
        training.data.poison.key.field=case_id \
        training.data.poison.source.type=manifest \
        training.data.poison.source.manifest_path=${MANIFEST_PATH} \
        training.data.poison.defense.type=gaussian_blur \
        training.data.poison.defense.sigma=1.0 \
        2>&1 | tee logs/victim_defense_gaussian_blur.log

    echo "[GPU 4] 开始实验 2/2: defense_gamma"
    python main.py \
        method=poison_files \
        model.pretrained=false \
        dataset=flare21 \
        task.run_name=victim_defense_gamma \
        model=unet \
        model.name=unet \
        task=flare21_seg \
        training.epochs=100 \
        training.optimizer=adam \
        training.optimizers.adam.lr=5e-4 \
        training.gpu_ids=[4] \
        training.batch_size=2 \
        training.eval_batch_size=2 \
        training.data.poison.enabled=true \
        training.data.poison.perturb_type=samplewise \
        training.data.poison.key.type=samplewise \
        training.data.poison.key.from=field \
        training.data.poison.key.field=case_id \
        training.data.poison.source.type=manifest \
        training.data.poison.source.manifest_path=${MANIFEST_PATH} \
        'training.data.poison.defense.type=gamma' \
        'training.data.poison.defense.gamma_range=[0.7,1.5]' \
        2>&1 | tee logs/victim_defense_gamma.log

    echo "[GPU 4] 所有实验完成"
) &

# ==================== GPU 5 (low_resolution) ====================
(
    echo "[GPU 5] 开始实验: defense_low_resolution"
    python main.py \
        method=poison_files \
        model.pretrained=false \
        dataset=flare21 \
        task.run_name=victim_defense_low_resolution \
        model=unet \
        model.name=unet \
        task=flare21_seg \
        training.epochs=100 \
        training.optimizer=adam \
        training.optimizers.adam.lr=5e-4 \
        training.gpu_ids=[5] \
        training.batch_size=2 \
        training.eval_batch_size=2 \
        training.data.poison.enabled=true \
        training.data.poison.perturb_type=samplewise \
        training.data.poison.key.type=samplewise \
        training.data.poison.key.from=field \
        training.data.poison.key.field=case_id \
        training.data.poison.source.type=manifest \
        training.data.poison.source.manifest_path=${MANIFEST_PATH} \
        training.data.poison.defense.type=low_resolution \
        training.data.poison.defense.scale=0.5 \
        2>&1 | tee logs/victim_defense_low_resolution.log

    echo "[GPU 5] 所有实验完成"
) &

# ==================== GPU 6 (random_affine) ====================
(
    echo "[GPU 6] 开始实验: defense_random_affine"
    python main.py \
        method=poison_files \
        model.pretrained=false \
        dataset=flare21 \
        task.run_name=victim_defense_random_affine \
        model=unet \
        model.name=unet \
        task=flare21_seg \
        training.epochs=100 \
        training.optimizer=adam \
        training.optimizers.adam.lr=5e-4 \
        training.gpu_ids=[6] \
        training.batch_size=2 \
        training.eval_batch_size=2 \
        training.data.poison.enabled=true \
        training.data.poison.perturb_type=samplewise \
        training.data.poison.key.type=samplewise \
        training.data.poison.key.from=field \
        training.data.poison.key.field=case_id \
        training.data.poison.source.type=manifest \
        training.data.poison.source.manifest_path=${MANIFEST_PATH} \
        training.data.poison.defense.type=random_affine \
        training.data.poison.defense.rotation_range=15.0 \
        'training.data.poison.defense.scale_range=[0.85,1.25]' \
        training.data.poison.defense.prob=0.5 \
        2>&1 | tee logs/victim_defense_random_affine.log

    echo "[GPU 6] 所有实验完成"
) &

echo "已启动 3 个 GPU (4,5,6) 的 FLARE21 Defense 消融实验"
echo ""
echo "实验对应关系 (4个实验):"
echo "  GPU 4: gaussian_blur (sigma=1.0) + gamma (range=[0.7,1.5])"
echo "  GPU 5: low_resolution (scale=0.5)"
echo "  GPU 6: random_affine (rot=15°, scale=[0.85,1.25], prob=0.5)"
echo ""
echo "查看日志: tail -f logs/victim_defense_*.log"
echo "查看进程: ps aux | grep main.py"
echo ""
echo "等待所有实验完成..."
wait
echo "所有 FLARE21 Defense 消融实验已完成！"
