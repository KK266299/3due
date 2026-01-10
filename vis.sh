python visualize_unet_noise.py \
    --model_path /home/dengzhipeng/data/project/3d_ue/outputs/kits19_seg/victim_unet_roi_4_255/20260105_130824/checkpoints/checkpoints/best_model.pth \
    --noise_dir /home/dengzhipeng/data/project/3d_ue/outputs/kits19_ue/unet_roi_noise_4_255/20260105_075932/noise/epoch_0099 \
    --output_dir /home/dengzhipeng/data/project/3d_ue/outputs/visualize \
    --dataset_config configs/dataset/kits19.yaml \
    --split train \
    --sample_idx 0 \
    --slice_idx 305 \
    --num_samples 1
