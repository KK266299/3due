#!/bin/bash
# 批量运行 cal_metrics.py —— 4 张 GPU 并行，每张 GPU 内的任务也并行
#
# 使用方法：
#   bash eval.sh
#
# GPU 分配：cuda:4 / cuda:5 / cuda:6 / cuda:7
# 每张 GPU 上的所有任务同时以后台进程启动，4 张卡之间也并行。
# 日志写入 logs/eval/ 目录。

set -e

GPUS=(4 5 6 7)
LOG_DIR="logs/eval"
mkdir -p "${LOG_DIR}"

# ── 全部任务 ──────────────────────────────────────────────────────
# 格式: "数据集名称|模型路径"
ALL_JOBS=(
    # ── brats19 (8) ──
    "brats19_seg|/home/dengzhipeng/data/project/3d_ue/outputs/brats19_seg/clean/20251220_144653/checkpoints/checkpoints/best_model.pth"
    "brats19_seg|/home/dengzhipeng/data/project/3d_ue/outputs/brats19_seg/victim_random_noise_gaussian/20251228_082846/checkpoints/checkpoints/best_model.pth"
    "brats19_seg|/home/dengzhipeng/data/project/3d_ue/outputs/brats19_seg/victim_minmin_4_255/20251220_061401/checkpoints/checkpoints/best_model.pth"
    "brats19_seg|/home/dengzhipeng/data/project/3d_ue/outputs/brats19_seg/victim_lsp_4_255/20260225_075837/checkpoints/checkpoints/best_model.pth"
    "brats19_seg|/home/dengzhipeng/data/project/3d_ue/outputs/brats19_seg/victim_sep/20251222_124616/checkpoints/checkpoints/best_model.pth"
    "brats19_seg|/home/dengzhipeng/data/project/3d_ue/outputs/brats19_seg/victim_tap_noise_4_255/20260225_225630/checkpoints/checkpoints/best_model.pth"
    "brats19_seg|/home/dengzhipeng/data/project/3d_ue/outputs/brats19_seg/victim_pue_4_255/20260225_021757/checkpoints/checkpoints/best_model.pth"
    "brats19_seg|/home/dengzhipeng/data/project/3d_ue/outputs/brats19_seg/victim_umed_brats19/20260225_183433/checkpoints/checkpoints/best_model.pth"
    # ── flare21 (9) ──
    "flare21_seg|/home/dengzhipeng/data/project/3d_ue/outputs/flare21_seg/clean/20260224_143824/checkpoints/checkpoints/best_model.pth"
    "flare21_seg|/home/dengzhipeng/data/project/3d_ue/outputs/flare21_seg/victim_random_noise_gaussian_4_255/20260225_081338/checkpoints/checkpoints/best_model.pth"
    "flare21_seg|/home/dengzhipeng/data/project/3d_ue/outputs/flare21_seg/victim_minmin_noise_step_5e-3_4_255/20260112_061355/checkpoints/checkpoints/best_model.pth"
    "flare21_seg|/home/dengzhipeng/data/project/3d_ue/outputs/flare21_seg/victim_sep/20260224_165559/checkpoints/checkpoints/best_model.pth"
    "flare21_seg|/home/dengzhipeng/data/project/3d_ue/outputs/flare21_seg/victim_tap_4_255/20260225_033246/checkpoints/checkpoints/best_model.pth"
    "flare21_seg|/home/dengzhipeng/data/project/3d_ue/outputs/flare21_seg/victim_umed_flare21/20260225_165841/checkpoints/checkpoints/best_model.pth"
    "flare21_seg|/home/dengzhipeng/data/project/3d_ue/outputs/flare21_seg/victim_nofreq_zdiv02_logits005/20260205_061624/checkpoints/checkpoints/best_model.pth"
    "flare21_seg|/home/dengzhipeng/data/project/3d_ue/outputs/flare21_seg/victim_nofreq_zdiv0_logits01/20260205_061751/checkpoints/checkpoints/best_model.pth"
    "flare21_seg|/home/dengzhipeng/data/project/3d_ue/outputs/flare21_seg/victim_nofreq_zdiv0_logits005/20260205_044934/checkpoints/checkpoints/best_model.pth"
)

NUM_JOBS=${#ALL_JOBS[@]}
NUM_GPUS=${#GPUS[@]}

echo "============================================================"
echo "  Total jobs : ${NUM_JOBS}"
echo "  GPUs       : ${GPUS[*]}"
echo "  Log dir    : ${LOG_DIR}"
echo "============================================================"
echo ""

# ── 用来收集所有后台 PID，最后统一 wait ──
ALL_PIDS=()

# ── 轮询分配 & 启动 ──────────────────────────────────────────────
for (( i=0; i<NUM_JOBS; i++ )); do
    gpu_idx=$(( i % NUM_GPUS ))
    DEVICE="cuda:${GPUS[$gpu_idx]}"

    IFS='|' read -r DATASET MODEL_PATH <<< "${ALL_JOBS[$i]}"

    # 从模型路径提取简短名字用于日志
    # 格式: .../<dataset>/<method>/<timestamp>/checkpoints/...
    METHOD=$(echo "${MODEL_PATH}" | awk -F'/' '{for(j=1;j<=NF;j++) if($j ~ /_seg$/) print $(j+1)}')
    LOG_FILE="${LOG_DIR}/${DATASET}_${METHOD}.log"

    if [ ! -f "${MODEL_PATH}" ]; then
        echo "[WARN] Model not found: ${MODEL_PATH}, skipping..."
        continue
    fi

    echo "[Launch] GPU=${DEVICE}  ${DATASET}/${METHOD}  -> ${LOG_FILE}"

    python cal_metrics.py \
        --dataset "${DATASET}" \
        --model_path "${MODEL_PATH}" \
        --device "${DEVICE}" \
        > "${LOG_FILE}" 2>&1 &

    ALL_PIDS+=($!)
done

echo ""
echo "All ${#ALL_PIDS[@]} jobs launched. Waiting..."
echo ""

# ── 等待全部完成 ──────────────────────────────────────────────────
FAIL=0
for pid in "${ALL_PIDS[@]}"; do
    if ! wait "$pid"; then
        FAIL=$((FAIL + 1))
    fi
done

echo "============================================================"
if [ $FAIL -eq 0 ]; then
    echo "  All jobs finished successfully."
else
    echo "  ${FAIL} job(s) FAILED. Check logs in ${LOG_DIR}/"
fi
echo "============================================================"