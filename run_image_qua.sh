#!/bin/bash
# =========================================================================
# 图像质量评估批处理脚本 (Image Quality Assessment Batch Script)
#
# 用法:
#   bash run_image_qua.sh
#
# 说明:
#   评估不同实验中加噪图像与原始图像的 PSNR / SSIM 指标。
#   修改下方 EXPERIMENTS 数组添加更多实验路径。
# =========================================================================

set -e

# ---- 全局配置 ----
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python}"
DATASET_CONFIG="${DATASET_CONFIG:-configs/dataset/brats19.yaml}"
SPLIT="${SPLIT:-train}"
OUTPUT_BASE="${OUTPUT_BASE:-outputs/image_quality}"
DEVICE="${DEVICE:-cpu}"
MAX_SAMPLES="${MAX_SAMPLES:--1}"   # -1 表示评估全部样本
USE_PYIQA="${USE_PYIQA:-false}"     # 设为 true 启用 pyiqa 指标

# ---- 实验列表 ----
# 格式: "实验名称|noise manifest 路径"
# 根据需要添加或修改下方条目
EXPERIMENTS=(
    "nofreq_learnable_zdiv02_logits005_epoch99|/home/dengzhipeng/data/project/3d_ue/outputs/brats19_ue/nofreq_learnable_zdiv02_logits005/20260206_045622/noise/epoch_0099/manifest.json"
    # --- 在此添加更多实验 ---
    # "experiment_name|/path/to/noise/manifest.json"
)

# ---- 主循环 ----
echo "=========================================="
echo " Image Quality Assessment Batch Runner"
echo "=========================================="
echo "Dataset config: ${DATASET_CONFIG}"
echo "Split: ${SPLIT}"
echo "Device: ${DEVICE}"
echo "Max samples: ${MAX_SAMPLES}"
echo "Use pyiqa: ${USE_PYIQA}"
echo ""

for entry in "${EXPERIMENTS[@]}"; do
    # 解析实验名称和路径
    IFS='|' read -r EXP_NAME MANIFEST_PATH <<< "${entry}"

    echo "------------------------------------------"
    echo "Experiment: ${EXP_NAME}"
    echo "Manifest:   ${MANIFEST_PATH}"
    echo "------------------------------------------"

    # 检查 manifest 是否存在
    if [ ! -f "${MANIFEST_PATH}" ]; then
        echo "[SKIP] Manifest not found: ${MANIFEST_PATH}"
        echo ""
        continue
    fi

    OUTPUT_DIR="${OUTPUT_BASE}/${EXP_NAME}"

    # 构建命令
    CMD="${PYTHON} ${SCRIPT_DIR}/image_qua.py \
        --noise_manifest ${MANIFEST_PATH} \
        --output_dir ${OUTPUT_DIR} \
        --dataset_config ${DATASET_CONFIG} \
        --split ${SPLIT} \
        --device ${DEVICE}"

    # 添加可选参数
    if [ "${MAX_SAMPLES}" != "-1" ]; then
        CMD="${CMD} --max_samples ${MAX_SAMPLES}"
    fi

    if [ "${USE_PYIQA}" = "true" ]; then
        CMD="${CMD} --use_pyiqa --metrics psnr ssim"
    fi

    # 执行
    echo "Running: ${CMD}"
    eval ${CMD}

    echo ""
    echo "[Done] Results saved to: ${OUTPUT_DIR}"
    echo ""
done

# ---- 汇总 ----
echo "=========================================="
echo " All experiments completed."
echo " Results directory: ${OUTPUT_BASE}"
echo "=========================================="

# 打印所有实验的汇总 CSV (如果存在)
echo ""
echo "Per-experiment results:"
for entry in "${EXPERIMENTS[@]}"; do
    IFS='|' read -r EXP_NAME MANIFEST_PATH <<< "${entry}"
    CSV_FILE="${OUTPUT_BASE}/${EXP_NAME}/results.csv"
    if [ -f "${CSV_FILE}" ]; then
        echo ""
        echo "--- ${EXP_NAME} ---"
        head -1 "${CSV_FILE}"
        tail -n +2 "${CSV_FILE}" | head -5
        TOTAL=$(wc -l < "${CSV_FILE}")
        if [ "${TOTAL}" -gt 6 ]; then
            echo "... (${TOTAL} total rows, showing first 5)"
        fi
    fi
done
