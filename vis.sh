python visualize_unet_noise.py \
    --model_path /home/dengzhipeng/data/project/3d_ue/outputs/brats19_seg/victim_unet_boundary_3/20251226_024420/checkpoints/checkpoints/best_model.pth \
    --noise_dir /home/dengzhipeng/data/project/3d_ue/outputs/brats19_ue/unet_noise_boundary_0_3/20251225_142314/noise/epoch_0099 \
    --output_dir /home/dengzhipeng/data/project/3d_ue/outputs/visualize \
    --dataset_config configs/dataset/brats19.yaml \
    --split train \
    --sample_idx 0 \
    --slice_idx -1 \
    --num_samples 5
