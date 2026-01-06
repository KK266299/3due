python visualize_unet_noise.py \
    --model_path /home/dengzhipeng/data/project/3d_ue/outputs/brats19_seg/victim_unet_slice_in_out_4_255/20260104_015305/checkpoints/checkpoints/best_model.pth \
    --noise_dir /home/dengzhipeng/data/project/3d_ue/outputs/brats19_ue/unet_slice_in_out_4_255/20260103_093048/noise/epoch_0099 \
    --output_dir /home/dengzhipeng/data/project/3d_ue/outputs/visualize \
    --dataset_config configs/dataset/brats19.yaml \
    --split train \
    --sample_idx 0 \
    --slice_idx -1 \
    --num_samples 5
