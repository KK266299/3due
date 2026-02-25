#!/bin/bash
# 批量运行 cal_metrics.py 计算 Dice / HD95 指标
#
# 使用方法：
#   bash run_cal_metrics.sh              # 默认用 cuda:0 顺序跑
#   bash run_cal_metrics.sh --gpu 5      # 指定单个 GPU
#   bash run_cal_metrics.sh --parallel   # 两个数据集分别用 cuda:0 和 cuda:1 并行
#
# 跑完后用 collect_results.py 汇总：
#   python collect_results.py --dataset brats19_seg
#   python collect_results.py --dataset flare21_seg

set -e

DEVICE="cuda:0"
PARALLEL=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --gpu) DEVICE="cuda:$2"; shift 2;;
        --parallel) PARALLEL=true; shift;;
        *) echo "Unknown option: $1"; exit 1;;
    esac
done

# ── BraTS19 模型列表 ──────────────────────────────────────────────
BRATS19_MODELS=(
    "/home/dengzhipeng/data/project/3d_ue/outputs/brats19_seg/clean/20251220_144653/checkpoints/checkpoints/best_model.pth"
    "/home/dengzhipeng/data/project/3d_ue/outputs/brats19_seg/victim_random_noise_gaussian/20251228_082846/checkpoints/checkpoints/best_model.pth"
    "/home/dengzhipeng/data/project/3d_ue/outputs/brats19_seg/victim_minmin_4_255/20251220_061401/checkpoints/checkpoints/best_model.pth"
    "/home/dengzhipeng/data/project/3d_ue/outputs/brats19_seg/victim_lsp_4_255/20260225_075837/checkpoints/checkpoints/best_model.pth"
    "/home/dengzhipeng/data/project/3d_ue/outputs/brats19_seg/victim_sep/20251222_124616/checkpoints/checkpoints/best_model.pth"
    "/home/dengzhipeng/data/project/3d_ue/outputs/brats19_seg/victim_pue_4_255/20260225_021757/checkpoints/checkpoints/best_model.pth"
)

# ── FLARE21 模型列表 ─────────────────────────────────────────────
FLARE21_MODELS=(
    "/home/dengzhipeng/data/project/3d_ue/outputs/flare21_seg/clean/20260224_143824/checkpoints/checkpoints/best_model.pth"
    "/home/dengzhipeng/data/project/3d_ue/outputs/flare21_seg/victim_random_noise_gaussian_4_255/20260225_081338/checkpoints/checkpoints/best_model.pth"
    "/home/dengzhipeng/data/project/3d_ue/outputs/flare21_seg/victim_minmin_noise_step_5e-3_4_255/20260112_061355/checkpoints/checkpoints/best_model.pth"
    "/home/dengzhipeng/data/project/3d_ue/outputs/flare21_seg/victim_sep/20260224_165559/checkpoints/checkpoints/best_model.pth"
    "/home/dengzhipeng/data/project/3d_ue/outputs/flare21_seg/victim_umed_flare21/20260225_165841/checkpoints/checkpoints/best_model.pth"
)

# ── 运行函数 ──────────────────────────────────────────────────────
run_dataset() {
    local DATASET=$1
    local DEV=$2
    shift 2
    local MODELS=("$@")

    echo ""
    echo "########################################"
    echo "  Dataset: ${DATASET}  Device: ${DEV}"
    echo "########################################"

    for MODEL_PATH in "${MODELS[@]}"; do
        echo "============================================================"
        echo "  Model: ${MODEL_PATH}"
        echo "============================================================"

        if [ ! -f "${MODEL_PATH}" ]; then
            echo "[WARN] Model not found, skipping..."
            continue
        fi

        python cal_metrics.py \
            --dataset "${DATASET}" \
            --model_path "${MODEL_PATH}" \
            --device "${DEV}"

        echo ""
    done

    # 汇总
    echo "---- Collecting results for ${DATASET} ----"
    python collect_results.py --dataset "${DATASET}"
    echo ""
}

# ── 执行 ──────────────────────────────────────────────────────────
if $PARALLEL; then
    echo "[Mode] Parallel: brats19 on cuda:0, flare21 on cuda:1"
    run_dataset "brats19_seg" "cuda:0" "${BRATS19_MODELS[@]}" &
    run_dataset "flare21_seg" "cuda:1" "${FLARE21_MODELS[@]}" &
    wait
else
    echo "[Mode] Sequential on ${DEVICE}"
    run_dataset "brats19_seg" "${DEVICE}" "${BRATS19_MODELS[@]}"
    run_dataset "flare21_seg" "${DEVICE}" "${FLARE21_MODELS[@]}"
fi

echo ""
echo "========================================"
echo "  [All Done]"
echo "========================================"
