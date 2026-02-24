#!/bin/bash
# 批量运行 cal_metrics.py 计算 HD95 和 Dice 指标
#
# 使用方法：
#   bash run_cal_metrics.sh
#
# 可以修改下方的 MODEL_LIST 数组来添加更多模型

set -e

DEVICE="cuda:0"

# ── 模型列表 ──────────────────────────────────────────────────────
# 格式: "数据集名称|模型路径"
MODEL_LIST=(
    # ── brats19 ──
    "brats19_seg|/home/dengzhipeng/data/project/3d_ue/outputs/brats19_seg/clean/20251220_144653/checkpoints/checkpoints/best_model.pth"
    "brats19_seg|/home/dengzhipeng/data/project/3d_ue/outputs/brats19_seg/victim_random_noise_gaussian/20251228_082846/checkpoints/checkpoints/best_model.pth"
    "brats19_seg|/home/dengzhipeng/data/project/3d_ue/outputs/brats19_seg/victim_minmin_4_255/20251220_061401/checkpoints/checkpoints/best_model.pth"
    "brats19_seg|/home/dengzhipeng/data/project/3d_ue/outputs/brats19_seg/victim_sep/20251222_124616/checkpoints/checkpoints/best_model.pth"
    "brats19_seg|/home/dengzhipeng/data/project/3d_ue/outputs/brats19_seg/victim_nofreq_learnable_zdiv02_logits005/20260207_135201/checkpoints/checkpoints/best_model.pth"
    # ── flare21 ──
    # "flare21_seg|/path/to/flare21/best_model.pth"
    # ── kits19 ──
    # "kits19_seg|/path/to/kits19/best_model.pth"
)

# ── 运行 ──────────────────────────────────────────────────────────
for entry in "${MODEL_LIST[@]}"; do
    IFS='|' read -r DATASET MODEL_PATH <<< "$entry"

    echo "============================================================"
    echo "  Dataset:    ${DATASET}"
    echo "  Model:      ${MODEL_PATH}"
    echo "  Device:     ${DEVICE}"
    echo "============================================================"

    if [ ! -f "${MODEL_PATH}" ]; then
        echo "[WARN] Model not found: ${MODEL_PATH}, skipping..."
        continue
    fi

    python cal_metrics.py \
        --dataset "${DATASET}" \
        --model_path "${MODEL_PATH}" \
        --device "${DEVICE}"

    echo ""
done

echo "[All Done]"
