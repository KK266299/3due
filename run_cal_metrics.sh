#!/bin/bash
# 批量运行 cal_metrics.py 计算 Dice / HD95 指标
# 4 卡并行: GPU 0/2/5/7
#
# 使用方法：
#   bash run_cal_metrics.sh
#
# 跑完后汇总：
#   python collect_results.py --dataset brats19_seg --percent
#   python collect_results.py --dataset flare21_seg --percent

set -e

run_one() {
    local DATASET=$1
    local MODEL_PATH=$2
    local DEV=$3
    echo "[${DEV}] ${DATASET} | $(basename $(dirname $(dirname $(dirname ${MODEL_PATH}))))"
    if [ ! -f "${MODEL_PATH}" ]; then
        echo "[WARN] Model not found: ${MODEL_PATH}, skipping..."
        return
    fi
    python cal_metrics.py --dataset "${DATASET}" --model_path "${MODEL_PATH}" --device "${DEV}"
}

# ── GPU 0: brats19 x3 ────────────────────────────────────────────
(
run_one brats19_seg /home/dengzhipeng/data/project/3d_ue/outputs/brats19_seg/clean/20251220_144653/checkpoints/checkpoints/best_model.pth cuda:0
run_one brats19_seg /home/dengzhipeng/data/project/3d_ue/outputs/brats19_seg/victim_random_noise_gaussian/20251228_082846/checkpoints/checkpoints/best_model.pth cuda:0
run_one brats19_seg /home/dengzhipeng/data/project/3d_ue/outputs/brats19_seg/victim_minmin_4_255/20251220_061401/checkpoints/checkpoints/best_model.pth cuda:0
echo "[GPU 0] Done."
) &

# ── GPU 2: brats19 x3 ────────────────────────────────────────────
(
run_one brats19_seg /home/dengzhipeng/data/project/3d_ue/outputs/brats19_seg/victim_lsp_4_255/20260225_075837/checkpoints/checkpoints/best_model.pth cuda:2
run_one brats19_seg /home/dengzhipeng/data/project/3d_ue/outputs/brats19_seg/victim_sep/20251222_124616/checkpoints/checkpoints/best_model.pth cuda:2
run_one brats19_seg /home/dengzhipeng/data/project/3d_ue/outputs/brats19_seg/victim_pue_4_255/20260225_021757/checkpoints/checkpoints/best_model.pth cuda:2
echo "[GPU 2] Done."
) &

# ── GPU 5: flare21 x3 ────────────────────────────────────────────
(
run_one flare21_seg /home/dengzhipeng/data/project/3d_ue/outputs/flare21_seg/clean/20260224_143824/checkpoints/checkpoints/best_model.pth cuda:5
run_one flare21_seg /home/dengzhipeng/data/project/3d_ue/outputs/flare21_seg/victim_random_noise_gaussian_4_255/20260225_081338/checkpoints/checkpoints/best_model.pth cuda:5
run_one flare21_seg /home/dengzhipeng/data/project/3d_ue/outputs/flare21_seg/victim_minmin_noise_step_5e-3_4_255/20260112_061355/checkpoints/checkpoints/best_model.pth cuda:5
echo "[GPU 5] Done."
) &

# ── GPU 7: flare21 x2 ────────────────────────────────────────────
(
run_one flare21_seg /home/dengzhipeng/data/project/3d_ue/outputs/flare21_seg/victim_sep/20260224_165559/checkpoints/checkpoints/best_model.pth cuda:7
run_one flare21_seg /home/dengzhipeng/data/project/3d_ue/outputs/flare21_seg/victim_umed_flare21/20260225_165841/checkpoints/checkpoints/best_model.pth cuda:7
echo "[GPU 7] Done."
) &

wait

echo ""
echo "========================================"
echo "  All metrics computed. Collecting..."
echo "========================================"
python collect_results.py --dataset brats19_seg --percent
python collect_results.py --dataset flare21_seg --percent
echo "[All Done]"
