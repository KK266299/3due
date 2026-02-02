#!/bin/bash
# Victim 训练消融实验脚本
# 使用 learnable ablation 生成的噪声训练 victim 模型
# 在 4 个 GPU 上并行运行，每个 GPU 内部串行执行

mkdir -p logs

# UE 输出目录
UE_BASE_DIR="/home/dengzhipeng/data/project/3d_ue/outputs/flare21_ue"

# 获取最新的 manifest.json 路径
get_latest_manifest() {
    local run_name=$1
    local run_dir="${UE_BASE_DIR}/${run_name}"

    if [ ! -d "$run_dir" ]; then
        echo "ERROR: 目录不存在: $run_dir" >&2
        return 1
    fi

    # 找到最新的时间戳文件夹（按名称排序，最后一个是最新的）
    local latest_timestamp=$(ls -1 "$run_dir" 2>/dev/null | sort | tail -1)

    if [ -z "$latest_timestamp" ]; then
        echo "ERROR: 没有找到时间戳文件夹: $run_dir" >&2
        return 1
    fi

    local manifest_path="${run_dir}/${latest_timestamp}/noise/manifest.json"

    if [ ! -f "$manifest_path" ]; then
        echo "ERROR: manifest.json 不存在: $manifest_path" >&2
        return 1
    fi

    echo "$manifest_path"
}

# 运行单个 victim 训练
run_victim_training() {
    local gpu_id=$1
    local ue_run_name=$2
    local victim_run_name="victim_${ue_run_name}"

    echo "[GPU $gpu_id] 查找 $ue_run_name 的 manifest..."
    local manifest_path=$(get_latest_manifest "$ue_run_name")

    if [ $? -ne 0 ]; then
        echo "[GPU $gpu_id] 跳过 $ue_run_name: 无法找到 manifest"
        return 1
    fi

    echo "[GPU $gpu_id] 开始训练: $victim_run_name"
    echo "[GPU $gpu_id] 使用 manifest: $manifest_path"

    python main.py \
        dataset=flare21 \
        task=flare21_seg \
        model=unet \
        training.gpu_ids=[$gpu_id] \
        training.epochs=200 \
        training.batch_size=2 \
        task.run_name="$victim_run_name" \
        training.data.poison.enabled=true \
        training.data.poison.perturb_type=samplewise \
        training.data.poison.apply_stage=before_normalize \
        training.data.poison.key.type=samplewise \
        training.data.poison.key.from=field \
        training.data.poison.key.field=case_id \
        training.data.poison.source.type=shards \
        training.data.poison.source.manifest_path="$manifest_path" \
        training.data.transforms.forbid_geom_aug=true \
        training.data.transforms.normalize=false \
        2>&1 | tee "logs/${victim_run_name}.log"

    echo "[GPU $gpu_id] 完成: $victim_run_name"
}

# 检查所有实验的 manifest 是否存在
echo "检查所有实验的 manifest..."
echo ""

ALL_EXPERIMENTS=(
    "learnable_zdiv0_logits0"
    "learnable_zdiv01_logits0"
    "learnable_zdiv02_logits0"
    "learnable_zdiv0_logits001"
    "learnable_zdiv0_logits005"
    "learnable_zdiv0_logits01"
    "learnable_zdiv01_logits001"
    "learnable_zdiv01_logits005"
    "learnable_zdiv005_logits005"
    "learnable_zdiv005_logits001"
    "learnable_zdiv02_logits001"
    "learnable_zdiv02_logits005"
)

missing_count=0
for exp in "${ALL_EXPERIMENTS[@]}"; do
    manifest=$(get_latest_manifest "$exp" 2>/dev/null)
    if [ $? -eq 0 ]; then
        echo "  [OK] $exp -> $manifest"
    else
        echo "  [MISSING] $exp"
        ((missing_count++))
    fi
done

echo ""
if [ $missing_count -gt 0 ]; then
    echo "警告: 有 $missing_count 个实验的 manifest 缺失"
    echo "按 Ctrl+C 取消，或等待 5 秒继续..."
    sleep 5
fi

echo ""
echo "开始 victim 训练..."
echo ""

# ==================== GPU 0 (串行) ====================
(
    run_victim_training 0 "learnable_zdiv0_logits0"
    run_victim_training 0 "learnable_zdiv01_logits0"
    run_victim_training 0 "learnable_zdiv02_logits0"
    echo "[GPU 0] 所有实验完成"
) &

# ==================== GPU 1 (串行) ====================
(
    run_victim_training 1 "learnable_zdiv0_logits001"
    run_victim_training 1 "learnable_zdiv0_logits005"
    run_victim_training 1 "learnable_zdiv0_logits01"
    echo "[GPU 1] 所有实验完成"
) &

# ==================== GPU 2 (串行) ====================
(
    run_victim_training 2 "learnable_zdiv01_logits001"
    run_victim_training 2 "learnable_zdiv01_logits005"
    run_victim_training 2 "learnable_zdiv005_logits005"
    echo "[GPU 2] 所有实验完成"
) &

# ==================== GPU 3 (串行) ====================
(
    run_victim_training 3 "learnable_zdiv005_logits001"
    run_victim_training 3 "learnable_zdiv02_logits001"
    run_victim_training 3 "learnable_zdiv02_logits005"
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
