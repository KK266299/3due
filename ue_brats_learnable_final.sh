#!/bin/bash
# 关闭频域约束消融实验脚本 - BraTS19
# 在 4 个 GPU 上并行运行，每个 GPU 内部串行执行

mkdir -p logs

# 基础参数（关闭频域约束）
BASE_CMD="python ue_generate.py \
    dataset=brats19 \
    task=brats19_ue \
    method=noise_slice_frequence_learnable \
    training.epochs=100 \
    training.batch_size=8 \
    ue.key.type=samplewise \
    ue.key.from=field \
    ue.key.field=case_id \
    ue.algorithm.params.epsilon=0.0156863 \
    ue.algorithm.params.noise_step=1 \
    ue.algorithm.params.surrogate_step=10 \
    ue.algorithm.params.roi_aware=true \
    ue.algorithm.params.soft_edge=false \
    ue.algorithm.params.freq_constraint_enabled=false \
    ue.io.save_from_epoch=50 \
    ue.io.save_every=10 \
    ue.surrogates.s_seg.in_channels=4 \
    ue.surrogates.s_seg.num_classes=4"

# ==================== GPU 4 (串行) ====================
(
    echo "[GPU 4] 开始实验 1/3: zdiv=0.05"
    $BASE_CMD \
        training.gpu_ids=[4] \
        task.run_name=nofreq_learnable_zdiv005_logits0 \
        ue.algorithm.params.z_diversity_weight=0.05 \
        ue.algorithm.params.logits_div_enabled=false \
        2>&1 | tee logs/brats19_nofreq_learnable_zdiv005_logits0.log

    echo "[GPU 4] 开始实验 2/3: zdiv=0.1"
    $BASE_CMD \
        training.gpu_ids=[4] \
        task.run_name=nofreq_learnable_zdiv01_logits0 \
        ue.algorithm.params.z_diversity_weight=0.1 \
        ue.algorithm.params.logits_div_enabled=false \
        2>&1 | tee logs/brats19_nofreq_learnable_zdiv01_logits0.log

    echo "[GPU 4] 开始实验 3/3: zdiv=0.2"
    $BASE_CMD \
        training.gpu_ids=[4] \
        task.run_name=nofreq_learnable_zdiv02_logits0 \
        ue.algorithm.params.z_diversity_weight=0.2 \
        ue.algorithm.params.logits_div_enabled=false \
        2>&1 | tee logs/brats19_nofreq_learnable_zdiv02_logits0.log

    echo "[GPU 4] 所有实验完成"
) &

# ==================== GPU 5 (串行) ====================
(
    echo "[GPU 5] 开始实验 1/3: logits=0.01"
    $BASE_CMD \
        training.gpu_ids=[5] \
        task.run_name=nofreq_learnable_zdiv0_logits001 \
        ue.algorithm.params.z_diversity_weight=0.0 \
        ue.algorithm.params.logits_div_enabled=true \
        ue.algorithm.params.logits_div_mode=fft_l1 \
        ue.algorithm.params.logits_div_weight=0.01 \
        2>&1 | tee logs/brats19_nofreq_learnable_zdiv0_logits001.log

    echo "[GPU 5] 开始实验 2/3: logits=0.05"
    $BASE_CMD \
        training.gpu_ids=[5] \
        task.run_name=nofreq_learnable_zdiv0_logits005 \
        ue.algorithm.params.z_diversity_weight=0.0 \
        ue.algorithm.params.logits_div_enabled=true \
        ue.algorithm.params.logits_div_mode=fft_l1 \
        ue.algorithm.params.logits_div_weight=0.05 \
        2>&1 | tee logs/brats19_nofreq_learnable_zdiv0_logits005.log

    echo "[GPU 5] 开始实验 3/3: logits=0.1"
    $BASE_CMD \
        training.gpu_ids=[5] \
        task.run_name=nofreq_learnable_zdiv0_logits01 \
        ue.algorithm.params.z_diversity_weight=0.0 \
        ue.algorithm.params.logits_div_enabled=true \
        ue.algorithm.params.logits_div_mode=fft_l1 \
        ue.algorithm.params.logits_div_weight=0.1 \
        2>&1 | tee logs/brats19_nofreq_learnable_zdiv0_logits01.log

    echo "[GPU 5] 所有实验完成"
) &

# ==================== GPU 6 (串行) ====================
(
    echo "[GPU 6] 开始实验 1/3: zdiv=0.1+logits=0.01"
    $BASE_CMD \
        training.gpu_ids=[6] \
        task.run_name=nofreq_learnable_zdiv01_logits001 \
        ue.algorithm.params.z_diversity_weight=0.1 \
        ue.algorithm.params.logits_div_enabled=true \
        ue.algorithm.params.logits_div_mode=fft_l1 \
        ue.algorithm.params.logits_div_weight=0.01 \
        2>&1 | tee logs/brats19_nofreq_learnable_zdiv01_logits001.log

    echo "[GPU 6] 开始实验 2/3: zdiv=0.1+logits=0.05"
    $BASE_CMD \
        training.gpu_ids=[6] \
        task.run_name=nofreq_learnable_zdiv01_logits005 \
        ue.algorithm.params.z_diversity_weight=0.1 \
        ue.algorithm.params.logits_div_enabled=true \
        ue.algorithm.params.logits_div_mode=fft_l1 \
        ue.algorithm.params.logits_div_weight=0.05 \
        2>&1 | tee logs/brats19_nofreq_learnable_zdiv01_logits005.log

    echo "[GPU 6] 开始实验 3/3: zdiv=0.05+logits=0.05"
    $BASE_CMD \
        training.gpu_ids=[6] \
        task.run_name=nofreq_learnable_zdiv005_logits005 \
        ue.algorithm.params.z_diversity_weight=0.05 \
        ue.algorithm.params.logits_div_enabled=true \
        ue.algorithm.params.logits_div_mode=fft_l1 \
        ue.algorithm.params.logits_div_weight=0.05 \
        2>&1 | tee logs/brats19_nofreq_learnable_zdiv005_logits005.log

    echo "[GPU 6] 所有实验完成"
) &

# ==================== GPU 7 (串行) ====================
(
    echo "[GPU 7] 开始实验 1/3: zdiv=0.05+logits=0.01"
    $BASE_CMD \
        training.gpu_ids=[7] \
        task.run_name=nofreq_learnable_zdiv005_logits001 \
        ue.algorithm.params.z_diversity_weight=0.05 \
        ue.algorithm.params.logits_div_enabled=true \
        ue.algorithm.params.logits_div_mode=fft_l1 \
        ue.algorithm.params.logits_div_weight=0.01 \
        2>&1 | tee logs/brats19_nofreq_learnable_zdiv005_logits001.log

    echo "[GPU 7] 开始实验 2/3: zdiv=0.2+logits=0.01"
    $BASE_CMD \
        training.gpu_ids=[7] \
        task.run_name=nofreq_learnable_zdiv02_logits001 \
        ue.algorithm.params.z_diversity_weight=0.2 \
        ue.algorithm.params.logits_div_enabled=true \
        ue.algorithm.params.logits_div_mode=fft_l1 \
        ue.algorithm.params.logits_div_weight=0.01 \
        2>&1 | tee logs/brats19_nofreq_learnable_zdiv02_logits001.log

    echo "[GPU 7] 开始实验 3/3: zdiv=0.2+logits=0.05"
    $BASE_CMD \
        training.gpu_ids=[7] \
        task.run_name=nofreq_learnable_zdiv02_logits005 \
        ue.algorithm.params.z_diversity_weight=0.2 \
        ue.algorithm.params.logits_div_enabled=true \
        ue.algorithm.params.logits_div_mode=fft_l1 \
        ue.algorithm.params.logits_div_weight=0.05 \
        2>&1 | tee logs/brats19_nofreq_learnable_zdiv02_logits005.log

    echo "[GPU 7] 所有实验完成"
) &

echo "已启动 4 个 GPU 的实验任务（每个 GPU 串行执行 3 个实验）"
echo ""
echo "freq_constraint_enabled=false（关闭频域约束）"
echo ""
echo "参数范围:"
echo "  z_diversity_weight: 0, 0.05, 0.1, 0.2"
echo "  logits_div_weight:  0, 0.01, 0.05, 0.1"
echo ""
echo "GPU 分配:"
echo "  GPU 4: zdiv=0.05, zdiv=0.1, zdiv=0.2"
echo "  GPU 5: logits=0.01, logits=0.05, logits=0.1"
echo "  GPU 6: zdiv=0.1+logits=0.01, zdiv=0.1+logits=0.05, zdiv=0.05+logits=0.05"
echo "  GPU 7: zdiv=0.05+logits=0.01, zdiv=0.2+logits=0.01, zdiv=0.2+logits=0.05"
echo ""
echo "查看日志: tail -f logs/brats19_nofreq_learnable_*.log"
echo "查看进程: ps aux | grep ue_generate"
echo ""
echo "等待所有实验完成..."
wait
echo "所有实验已完成！"
