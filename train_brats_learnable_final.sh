#!/bin/bash
# BraTS19 Victim 训练消融实验脚本 (nofreq_learnable)
# 使用 nofreq_learnable ablation 生成的噪声训练 victim 模型
# 在 4 个 GPU (4,5,6,7) 上并行运行，每个 GPU 内部串行执行

mkdir -p logs

# ==================== GPU 4 (串行) ====================
(
    echo "[GPU 4] 开始实验 1/3: victim_nofreq_learnable_zdiv005_logits0"
    python main.py \
        method=poison_files \
        model.pretrained=false \
        dataset=brats19 \
        task.run_name=victim_nofreq_learnable_zdiv005_logits0 \
        model=unet \
        model.name=unet \
        task=brats19_seg \
        training.epochs=100 \
        training.optimizer=adam \
        training.optimizers.adam.lr=5e-4 \
        training.gpu_ids=[4] \
        training.batch_size=8 \
        training.eval_batch_size=8 \
        training.data.poison.perturb_type=samplewise \
        training.data.poison.key.type=samplewise \
        training.data.poison.key.from=field \
        training.data.poison.key.field=case_id \
        training.data.poison.source.type=manifest \
        training.data.poison.source.manifest_path=/home/dengzhipeng/data/project/3d_ue/outputs/brats19_ue/nofreq_learnable_zdiv005_logits0/20260205_154323/noise/epoch_0099/manifest.json \
        2>&1 | tee logs/victim_nofreq_learnable_zdiv005_logits0.log

    echo "[GPU 4] 开始实验 2/3: victim_nofreq_learnable_zdiv01_logits0"
    python main.py \
        method=poison_files \
        model.pretrained=false \
        dataset=brats19 \
        task.run_name=victim_nofreq_learnable_zdiv01_logits0 \
        model=unet \
        model.name=unet \
        task=brats19_seg \
        training.epochs=100 \
        training.optimizer=adam \
        training.optimizers.adam.lr=5e-4 \
        training.gpu_ids=[4] \
        training.batch_size=8 \
        training.eval_batch_size=8 \
        training.data.poison.perturb_type=samplewise \
        training.data.poison.key.type=samplewise \
        training.data.poison.key.from=field \
        training.data.poison.key.field=case_id \
        training.data.poison.source.type=manifest \
        training.data.poison.source.manifest_path=/home/dengzhipeng/data/project/3d_ue/outputs/brats19_ue/nofreq_learnable_zdiv01_logits0/20260205_211847/noise/epoch_0099/manifest.json \
        2>&1 | tee logs/victim_nofreq_learnable_zdiv01_logits0.log

    echo "[GPU 4] 开始实验 3/3: victim_nofreq_learnable_zdiv02_logits0"
    python main.py \
        method=poison_files \
        model.pretrained=false \
        dataset=brats19 \
        task.run_name=victim_nofreq_learnable_zdiv02_logits0 \
        model=unet \
        model.name=unet \
        task=brats19_seg \
        training.epochs=100 \
        training.optimizer=adam \
        training.optimizers.adam.lr=5e-4 \
        training.gpu_ids=[4] \
        training.batch_size=8 \
        training.eval_batch_size=8 \
        training.data.poison.perturb_type=samplewise \
        training.data.poison.key.type=samplewise \
        training.data.poison.key.from=field \
        training.data.poison.key.field=case_id \
        training.data.poison.source.type=manifest \
        training.data.poison.source.manifest_path=/home/dengzhipeng/data/project/3d_ue/outputs/brats19_ue/nofreq_learnable_zdiv02_logits0/20260206_023536/noise/epoch_0099/manifest.json \
        2>&1 | tee logs/victim_nofreq_learnable_zdiv02_logits0.log

    echo "[GPU 4] 所有实验完成"
) &

# ==================== GPU 5 (串行) ====================
(
    echo "[GPU 5] 开始实验 1/3: victim_nofreq_learnable_zdiv0_logits001"
    python main.py \
        method=poison_files \
        model.pretrained=false \
        dataset=brats19 \
        task.run_name=victim_nofreq_learnable_zdiv0_logits001 \
        model=unet \
        model.name=unet \
        task=brats19_seg \
        training.epochs=100 \
        training.optimizer=adam \
        training.optimizers.adam.lr=5e-4 \
        training.gpu_ids=[5] \
        training.batch_size=8 \
        training.eval_batch_size=8 \
        training.data.poison.perturb_type=samplewise \
        training.data.poison.key.type=samplewise \
        training.data.poison.key.from=field \
        training.data.poison.key.field=case_id \
        training.data.poison.source.type=manifest \
        training.data.poison.source.manifest_path=/home/dengzhipeng/data/project/3d_ue/outputs/brats19_ue/nofreq_learnable_zdiv0_logits001/20260205_154323/noise/epoch_0099/manifest.json \
        2>&1 | tee logs/victim_nofreq_learnable_zdiv0_logits001.log

    echo "[GPU 5] 开始实验 2/3: victim_nofreq_learnable_zdiv0_logits005"
    python main.py \
        method=poison_files \
        model.pretrained=false \
        dataset=brats19 \
        task.run_name=victim_nofreq_learnable_zdiv0_logits005 \
        model=unet \
        model.name=unet \
        task=brats19_seg \
        training.epochs=100 \
        training.optimizer=adam \
        training.optimizers.adam.lr=5e-4 \
        training.gpu_ids=[5] \
        training.batch_size=8 \
        training.eval_batch_size=8 \
        training.data.poison.perturb_type=samplewise \
        training.data.poison.key.type=samplewise \
        training.data.poison.key.from=field \
        training.data.poison.key.field=case_id \
        training.data.poison.source.type=manifest \
        training.data.poison.source.manifest_path=/home/dengzhipeng/data/project/3d_ue/outputs/brats19_ue/nofreq_learnable_zdiv0_logits005/20260205_222803/noise/epoch_0099/manifest.json \
        2>&1 | tee logs/victim_nofreq_learnable_zdiv0_logits005.log

    echo "[GPU 5] 开始实验 3/3: victim_nofreq_learnable_zdiv0_logits01"
    python main.py \
        method=poison_files \
        model.pretrained=false \
        dataset=brats19 \
        task.run_name=victim_nofreq_learnable_zdiv0_logits01 \
        model=unet \
        model.name=unet \
        task=brats19_seg \
        training.epochs=100 \
        training.optimizer=adam \
        training.optimizers.adam.lr=5e-4 \
        training.gpu_ids=[5] \
        training.batch_size=8 \
        training.eval_batch_size=8 \
        training.data.poison.perturb_type=samplewise \
        training.data.poison.key.type=samplewise \
        training.data.poison.key.from=field \
        training.data.poison.key.field=case_id \
        training.data.poison.source.type=manifest \
        training.data.poison.source.manifest_path=/home/dengzhipeng/data/project/3d_ue/outputs/brats19_ue/nofreq_learnable_zdiv0_logits01/20260206_050343/noise/epoch_0099/manifest.json \
        2>&1 | tee logs/victim_nofreq_learnable_zdiv0_logits01.log

    echo "[GPU 5] 所有实验完成"
) &

# ==================== GPU 6 (串行) ====================
(
    echo "[GPU 6] 开始实验 1/3: victim_nofreq_learnable_zdiv01_logits001"
    python main.py \
        method=poison_files \
        model.pretrained=false \
        dataset=brats19 \
        task.run_name=victim_nofreq_learnable_zdiv01_logits001 \
        model=unet \
        model.name=unet \
        task=brats19_seg \
        training.epochs=100 \
        training.optimizer=adam \
        training.optimizers.adam.lr=5e-4 \
        training.gpu_ids=[6] \
        training.batch_size=8 \
        training.eval_batch_size=8 \
        training.data.poison.perturb_type=samplewise \
        training.data.poison.key.type=samplewise \
        training.data.poison.key.from=field \
        training.data.poison.key.field=case_id \
        training.data.poison.source.type=manifest \
        training.data.poison.source.manifest_path=/home/dengzhipeng/data/project/3d_ue/outputs/brats19_ue/nofreq_learnable_zdiv01_logits001/20260205_154323/noise/epoch_0099/manifest.json \
        2>&1 | tee logs/victim_nofreq_learnable_zdiv01_logits001.log

    echo "[GPU 6] 开始实验 2/3: victim_nofreq_learnable_zdiv01_logits005"
    python main.py \
        method=poison_files \
        model.pretrained=false \
        dataset=brats19 \
        task.run_name=victim_nofreq_learnable_zdiv01_logits005 \
        model=unet \
        model.name=unet \
        task=brats19_seg \
        training.epochs=100 \
        training.optimizer=adam \
        training.optimizers.adam.lr=5e-4 \
        training.gpu_ids=[6] \
        training.batch_size=8 \
        training.eval_batch_size=8 \
        training.data.poison.perturb_type=samplewise \
        training.data.poison.key.type=samplewise \
        training.data.poison.key.from=field \
        training.data.poison.key.field=case_id \
        training.data.poison.source.type=manifest \
        training.data.poison.source.manifest_path=/home/dengzhipeng/data/project/3d_ue/outputs/brats19_ue/nofreq_learnable_zdiv01_logits005/20260205_222405/noise/epoch_0099/manifest.json \
        2>&1 | tee logs/victim_nofreq_learnable_zdiv01_logits005.log

    echo "[GPU 6] 开始实验 3/3: victim_nofreq_learnable_zdiv005_logits005"
    python main.py \
        method=poison_files \
        model.pretrained=false \
        dataset=brats19 \
        task.run_name=victim_nofreq_learnable_zdiv005_logits005 \
        model=unet \
        model.name=unet \
        task=brats19_seg \
        training.epochs=100 \
        training.optimizer=adam \
        training.optimizers.adam.lr=5e-4 \
        training.gpu_ids=[6] \
        training.batch_size=8 \
        training.eval_batch_size=8 \
        training.data.poison.perturb_type=samplewise \
        training.data.poison.key.type=samplewise \
        training.data.poison.key.from=field \
        training.data.poison.key.field=case_id \
        training.data.poison.source.type=manifest \
        training.data.poison.source.manifest_path=/home/dengzhipeng/data/project/3d_ue/outputs/brats19_ue/nofreq_learnable_zdiv005_logits005/20260206_045654/noise/epoch_0099/manifest.json \
        2>&1 | tee logs/victim_nofreq_learnable_zdiv005_logits005.log

    echo "[GPU 6] 所有实验完成"
) &

# ==================== GPU 7 (串行) ====================
(
    echo "[GPU 7] 开始实验 1/3: victim_nofreq_learnable_zdiv005_logits001"
    python main.py \
        method=poison_files \
        model.pretrained=false \
        dataset=brats19 \
        task.run_name=victim_nofreq_learnable_zdiv005_logits001 \
        model=unet \
        model.name=unet \
        task=brats19_seg \
        training.epochs=100 \
        training.optimizer=adam \
        training.optimizers.adam.lr=5e-4 \
        training.gpu_ids=[7] \
        training.batch_size=8 \
        training.eval_batch_size=8 \
        training.data.poison.perturb_type=samplewise \
        training.data.poison.key.type=samplewise \
        training.data.poison.key.from=field \
        training.data.poison.key.field=case_id \
        training.data.poison.source.type=manifest \
        training.data.poison.source.manifest_path=/home/dengzhipeng/data/project/3d_ue/outputs/brats19_ue/nofreq_learnable_zdiv005_logits001/20260205_154323/noise/epoch_0099/manifest.json \
        2>&1 | tee logs/victim_nofreq_learnable_zdiv005_logits001.log

    echo "[GPU 7] 开始实验 2/3: victim_nofreq_learnable_zdiv02_logits001"
    python main.py \
        method=poison_files \
        model.pretrained=false \
        dataset=brats19 \
        task.run_name=victim_nofreq_learnable_zdiv02_logits001 \
        model=unet \
        model.name=unet \
        task=brats19_seg \
        training.epochs=100 \
        training.optimizer=adam \
        training.optimizers.adam.lr=5e-4 \
        training.gpu_ids=[7] \
        training.batch_size=8 \
        training.eval_batch_size=8 \
        training.data.poison.perturb_type=samplewise \
        training.data.poison.key.type=samplewise \
        training.data.poison.key.from=field \
        training.data.poison.key.field=case_id \
        training.data.poison.source.type=manifest \
        training.data.poison.source.manifest_path=/home/dengzhipeng/data/project/3d_ue/outputs/brats19_ue/nofreq_learnable_zdiv02_logits001/20260205_222410/noise/epoch_0099/manifest.json \
        2>&1 | tee logs/victim_nofreq_learnable_zdiv02_logits001.log

    echo "[GPU 7] 开始实验 3/3: victim_nofreq_learnable_zdiv02_logits005"
    python main.py \
        method=poison_files \
        model.pretrained=false \
        dataset=brats19 \
        task.run_name=victim_nofreq_learnable_zdiv02_logits005 \
        model=unet \
        model.name=unet \
        task=brats19_seg \
        training.epochs=100 \
        training.optimizer=adam \
        training.optimizers.adam.lr=5e-4 \
        training.gpu_ids=[7] \
        training.batch_size=8 \
        training.eval_batch_size=8 \
        training.data.poison.perturb_type=samplewise \
        training.data.poison.key.type=samplewise \
        training.data.poison.key.from=field \
        training.data.poison.key.field=case_id \
        training.data.poison.source.type=manifest \
        training.data.poison.source.manifest_path=/home/dengzhipeng/data/project/3d_ue/outputs/brats19_ue/nofreq_learnable_zdiv02_logits005/20260206_045622/noise/epoch_0099/manifest.json \
        2>&1 | tee logs/victim_nofreq_learnable_zdiv02_logits005.log

    echo "[GPU 7] 所有实验完成"
) &

echo "已启动 4 个 GPU (4,5,6,7) 的 BraTS19 victim 训练任务"
echo ""
echo "实验对应关系 (12个实验, nofreq_learnable):"
echo "  GPU 4: zdiv=0.05, zdiv=0.1, zdiv=0.2 (logits=0)"
echo "  GPU 5: logits=0.01, logits=0.05, logits=0.1 (zdiv=0)"
echo "  GPU 6: zdiv=0.1+logits=0.01, zdiv=0.1+logits=0.05, zdiv=0.05+logits=0.05"
echo "  GPU 7: zdiv=0.05+logits=0.01, zdiv=0.2+logits=0.01, zdiv=0.2+logits=0.05"
echo ""
echo "查看日志: tail -f logs/victim_nofreq_*.log"
echo "查看进程: ps aux | grep main.py"
echo ""
echo "等待所有实验完成..."
wait
echo "所有 BraTS19 victim 训练已完成！"
