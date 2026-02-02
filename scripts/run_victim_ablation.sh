#!/bin/bash
# Victim 训练消融实验脚本
# 使用 learnable ablation 生成的噪声训练 victim 模型
# 在 4 个 GPU 上并行运行，每个 GPU 内部串行执行

mkdir -p logs

# UE 输出基础目录
UE_BASE="/home/dengzhipeng/data/project/3d_ue/outputs/flare21_ue"

# 获取最新 manifest 路径的辅助函数
get_manifest() {
    local run_name=$1
    local latest=$(ls -1 "$UE_BASE/$run_name" 2>/dev/null | sort | tail -1)
    echo "$UE_BASE/$run_name/$latest/noise/manifest.json"
}

# 基础参数
BASE_CMD="python main.py \
    dataset=flare21 \
    task=flare21_seg \
    model=unet \
    training.epochs=200 \
    training.batch_size=2 \
    training.data.poison.enabled=true \
    training.data.poison.perturb_type=samplewise \
    training.data.poison.apply_stage=before_normalize \
    training.data.poison.key.type=samplewise \
    training.data.poison.key.from=field \
    training.data.poison.key.field=case_id \
    training.data.poison.source.type=shards \
    training.data.transforms.forbid_geom_aug=true \
    training.data.transforms.normalize=false"

# ==================== GPU 0 (串行) ====================
(
    echo "[GPU 0] 开始实验 1/3: victim_learnable_zdiv0_logits0"
    $BASE_CMD \
        training.gpu_ids=[0] \
        task.run_name=victim_learnable_zdiv0_logits0 \
        training.data.poison.source.manifest_path=$(get_manifest learnable_zdiv0_logits0) \
        2>&1 | tee logs/victim_learnable_zdiv0_logits0.log

    echo "[GPU 0] 开始实验 2/3: victim_learnable_zdiv01_logits0"
    $BASE_CMD \
        training.gpu_ids=[0] \
        task.run_name=victim_learnable_zdiv01_logits0 \
        training.data.poison.source.manifest_path=$(get_manifest learnable_zdiv01_logits0) \
        2>&1 | tee logs/victim_learnable_zdiv01_logits0.log

    echo "[GPU 0] 开始实验 3/3: victim_learnable_zdiv02_logits0"
    $BASE_CMD \
        training.gpu_ids=[0] \
        task.run_name=victim_learnable_zdiv02_logits0 \
        training.data.poison.source.manifest_path=$(get_manifest learnable_zdiv02_logits0) \
        2>&1 | tee logs/victim_learnable_zdiv02_logits0.log

    echo "[GPU 0] 所有实验完成"
) &

# ==================== GPU 1 (串行) ====================
(
    echo "[GPU 1] 开始实验 1/3: victim_learnable_zdiv0_logits001"
    $BASE_CMD \
        training.gpu_ids=[1] \
        task.run_name=victim_learnable_zdiv0_logits001 \
        training.data.poison.source.manifest_path=$(get_manifest learnable_zdiv0_logits001) \
        2>&1 | tee logs/victim_learnable_zdiv0_logits001.log

    echo "[GPU 1] 开始实验 2/3: victim_learnable_zdiv0_logits005"
    $BASE_CMD \
        training.gpu_ids=[1] \
        task.run_name=victim_learnable_zdiv0_logits005 \
        training.data.poison.source.manifest_path=$(get_manifest learnable_zdiv0_logits005) \
        2>&1 | tee logs/victim_learnable_zdiv0_logits005.log

    echo "[GPU 1] 开始实验 3/3: victim_learnable_zdiv0_logits01"
    $BASE_CMD \
        training.gpu_ids=[1] \
        task.run_name=victim_learnable_zdiv0_logits01 \
        training.data.poison.source.manifest_path=$(get_manifest learnable_zdiv0_logits01) \
        2>&1 | tee logs/victim_learnable_zdiv0_logits01.log

    echo "[GPU 1] 所有实验完成"
) &

# ==================== GPU 2 (串行) ====================
(
    echo "[GPU 2] 开始实验 1/3: victim_learnable_zdiv01_logits001"
    $BASE_CMD \
        training.gpu_ids=[2] \
        task.run_name=victim_learnable_zdiv01_logits001 \
        training.data.poison.source.manifest_path=$(get_manifest learnable_zdiv01_logits001) \
        2>&1 | tee logs/victim_learnable_zdiv01_logits001.log

    echo "[GPU 2] 开始实验 2/3: victim_learnable_zdiv01_logits005"
    $BASE_CMD \
        training.gpu_ids=[2] \
        task.run_name=victim_learnable_zdiv01_logits005 \
        training.data.poison.source.manifest_path=$(get_manifest learnable_zdiv01_logits005) \
        2>&1 | tee logs/victim_learnable_zdiv01_logits005.log

    echo "[GPU 2] 开始实验 3/3: victim_learnable_zdiv005_logits005"
    $BASE_CMD \
        training.gpu_ids=[2] \
        task.run_name=victim_learnable_zdiv005_logits005 \
        training.data.poison.source.manifest_path=$(get_manifest learnable_zdiv005_logits005) \
        2>&1 | tee logs/victim_learnable_zdiv005_logits005.log

    echo "[GPU 2] 所有实验完成"
) &

# ==================== GPU 3 (串行) ====================
(
    echo "[GPU 3] 开始实验 1/3: victim_learnable_zdiv005_logits001"
    $BASE_CMD \
        training.gpu_ids=[3] \
        task.run_name=victim_learnable_zdiv005_logits001 \
        training.data.poison.source.manifest_path=$(get_manifest learnable_zdiv005_logits001) \
        2>&1 | tee logs/victim_learnable_zdiv005_logits001.log

    echo "[GPU 3] 开始实验 2/3: victim_learnable_zdiv02_logits001"
    $BASE_CMD \
        training.gpu_ids=[3] \
        task.run_name=victim_learnable_zdiv02_logits001 \
        training.data.poison.source.manifest_path=$(get_manifest learnable_zdiv02_logits001) \
        2>&1 | tee logs/victim_learnable_zdiv02_logits001.log

    echo "[GPU 3] 开始实验 3/3: victim_learnable_zdiv02_logits005"
    $BASE_CMD \
        training.gpu_ids=[3] \
        task.run_name=victim_learnable_zdiv02_logits005 \
        training.data.poison.source.manifest_path=$(get_manifest learnable_zdiv02_logits005) \
        2>&1 | tee logs/victim_learnable_zdiv02_logits005.log

    echo "[GPU 3] 所有实验完成"
) &

echo "已启动 4 个 GPU 的 victim 训练任务（每个 GPU 串行执行 3 个实验）"
echo ""
echo "实验对应关系:"
echo "  GPU 0: baseline, zdiv=0.1, zdiv=0.2"
echo "  GPU 1: logits=0.01, logits=0.05, logits=0.1"
echo "  GPU 2: zdiv=0.1+logits=0.01, zdiv=0.1+logits=0.05, zdiv=0.05+logits=0.05"
echo "  GPU 3: zdiv=0.05+logits=0.01, zdiv=0.2+logits=0.01, zdiv=0.2+logits=0.05"
echo ""
echo "查看日志: tail -f logs/victim_*.log"
echo "查看进程: ps aux | grep main.py"
echo ""
echo "等待所有实验完成..."
wait
echo "所有 victim 训练已完成！"
