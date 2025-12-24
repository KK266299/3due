python visualize_unet_noise.py \
    --model_path /home/dengzhipeng/data/project/3d_ue/outputs/brats19_seg/victim_sep/20251222_124616/checkpoints/checkpoints/best_model.pth \
    --noise_dir /home/dengzhipeng/data/project/3d_ue/outputs/brats19_ue/sep_noise/20251221_082830/noise/epoch_0000/ \
    --output_dir /home/dengzhipeng/data/project/3d_ue/outputs/visualize \
    --dataset_config configs/dataset/brats19.yaml \
    --split train \
    --sample_idx 0 \
    --slice_idx -1 \
    --num_samples 5
