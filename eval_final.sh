# 1) brats19 nofreq_learnable
python cal_metrics.py \
    --dataset brats19_seg \
    --model_path /home/dengzhipeng/data/project/3d_ue/outputs/brats19_seg/victim_nofreq_learnable_zdiv02_logits005/20260207_135201/checkpoints/checkpoints/best_model.pth \
    --device cuda:5 &

# 2) flare21 lsp
python cal_metrics.py \
    --dataset flare21_seg \
    --model_path /home/dengzhipeng/data/project/3d_ue/outputs/flare21_seg/victim_lsp/20260224_143735/checkpoints/checkpoints/best_model.pth \
    --device cuda:5 &

wait 
echo "All tasks done! Generating summary..."