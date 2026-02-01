# GPU 0: brats19 random, minmin, lsp
(
CUDA_VISIBLE_DEVICES=0
python image_qua.py --noise_manifest /home/dengzhipeng/data/project/3d_ue/outputs/brats19_ue/random_noise_gaussian/20251228_073254/noise/manifest.json --output_dir outputs/image_quality/brats19/random_noise_gaussian --dataset_config configs/dataset/brats19.yaml --device cuda:0 &
python image_qua.py --noise_manifest /home/dengzhipeng/data/project/3d_ue/outputs/brats19_ue/minmin_noise_step_5e-3_4_255/20251219_115411/noise/epoch_0099/manifest.json --output_dir outputs/image_quality/brats19/minmin_noise_step_5e-3_4_255 --dataset_config configs/dataset/brats19.yaml --device cuda:0 &
python image_qua.py --noise_manifest /home/dengzhipeng/data/project/3d_ue/outputs/brats19_ue/lsp_noise/20251225_110708/ue/lsp/manifest.json --output_dir outputs/image_quality/brats19/lsp_noise --dataset_config configs/dataset/brats19.yaml --device cuda:0 &
wait
) &

# GPU 2: brats19 sep, pue, umed
(
CUDA_VISIBLE_DEVICES=2
python image_qua.py --noise_manifest /home/dengzhipeng/data/project/3d_ue/outputs/brats19_ue/sep_noise/20251221_082830/noise/epoch_0000/manifest.json --output_dir outputs/image_quality/brats19/sep_noise --dataset_config configs/dataset/brats19.yaml --device cuda:0 &
python image_qua.py --noise_manifest /home/dengzhipeng/data/project/3d_ue/outputs/brats19_ue/pue_4_255/20260224_171456/noise/epoch_0099/manifest.json --output_dir outputs/image_quality/brats19/pue_4_255 --dataset_config configs/dataset/brats19.yaml --device cuda:0 &
python image_qua.py --noise_manifest /home/dengzhipeng/data/project/3d_ue/outputs/brats19_ue/umed_brats19/20260225_072310/noise/epoch_0099/manifest.json --output_dir outputs/image_quality/brats19/umed_brats19 --dataset_config configs/dataset/brats19.yaml --device cuda:0 &
wait
) &

# GPU 4: brats19 nofreq, flare21 random, flare21 minmin
(
CUDA_VISIBLE_DEVICES=4
python image_qua.py --noise_manifest /home/dengzhipeng/data/project/3d_ue/outputs/brats19_ue/nofreq_learnable_zdiv02_logits005/20260206_045622/noise/epoch_0099/manifest.json --output_dir outputs/image_quality/brats19/nofreq_learnable_zdiv02_logits005 --dataset_config configs/dataset/brats19.yaml --device cuda:0 &
python image_qua.py --noise_manifest /home/dengzhipeng/data/project/3d_ue/outputs/flare21_ue/random_noise_gaussian/20260225_081059/noise/manifest.json --output_dir outputs/image_quality/flare21/random_noise_gaussian --dataset_config configs/dataset/flare21.yaml --device cuda:0 &
python image_qua.py --noise_manifest /home/dengzhipeng/data/project/3d_ue/outputs/flare21_ue/minmin_noise_step_5e-3_4_255/20260111_154607/noise/epoch_0099/manifest.json --output_dir outputs/image_quality/flare21/minmin_noise_step_5e-3_4_255 --dataset_config configs/dataset/flare21.yaml --device cuda:0 &
wait
) &

# GPU 5: flare21 lsp, sep
(
CUDA_VISIBLE_DEVICES=5
python image_qua.py --noise_manifest /home/dengzhipeng/data/project/3d_ue/outputs/flare21_ue/lsp_noise_4_255/20260224_143422/ue/lsp/manifest.json --output_dir outputs/image_quality/flare21/lsp_noise_4_255 --dataset_config configs/dataset/flare21.yaml --device cuda:0 &
python image_qua.py --noise_manifest /home/dengzhipeng/data/project/3d_ue/outputs/flare21_ue/sep_noise_4_255/20260224_163024/noise/epoch_0000/manifest.json --output_dir outputs/image_quality/flare21/sep_noise_4_255 --dataset_config configs/dataset/flare21.yaml --device cuda:0 &
wait
) &

# GPU 7: flare21 umed, nofreq
(
CUDA_VISIBLE_DEVICES=7
python image_qua.py --noise_manifest /home/dengzhipeng/data/project/3d_ue/outputs/flare21_ue/umed_flare21/20260225_072339/noise/epoch_0099/manifest.json --output_dir outputs/image_quality/flare21/umed_flare21 --dataset_config configs/dataset/flare21.yaml --device cuda:0 &
python image_qua.py --noise_manifest /home/dengzhipeng/data/project/3d_ue/outputs/flare21_ue/nofreq_zdiv02_logits005/20260204_172349/noise/epoch_0099/manifest.json --output_dir outputs/image_quality/flare21/nofreq_zdiv02_logits005 --dataset_config configs/dataset/flare21.yaml --device cuda:0 &
wait
) &

wait
echo "All 13 tasks done! Generating summary..."

# ---- 汇总表格 ----
python -c "
import json, os, glob

base = 'outputs/image_quality'
rows = []
for dataset in ['brats19', 'flare21']:
    for jf in sorted(glob.glob(os.path.join(base, dataset, '*/results.json'))):
        method = os.path.basename(os.path.dirname(jf))
        with open(jf) as f:
            data = json.load(f)
        s = data.get('summary', {})
        psnr = s.get('PSNR', {})
        ssim = s.get('SSIM', {})
        l2   = s.get('Noise_L2', {})
        linf = s.get('Noise_Linf', {})
        rows.append((dataset, method, s.get('num_samples', 0),
                      psnr.get('mean', 0), psnr.get('std', 0),
                      ssim.get('mean', 0), ssim.get('std', 0),
                      l2.get('mean', 0), linf.get('mean', 0)))

# print table
hdr = f\"{'Dataset':<10} {'Method':<40} {'N':>4}  {'PSNR':>8} {'±std':>7}  {'SSIM':>8} {'±std':>7}  {'L2':>8} {'Linf':>8}\"
sep = '-' * len(hdr)
print()
print(sep)
print(hdr)
print(sep)
cur_ds = ''
for ds, m, n, pm, ps, sm, ss, l2m, lim in rows:
    if ds != cur_ds:
        if cur_ds: print(sep)
        cur_ds = ds
    print(f'{ds:<10} {m:<40} {n:>4}  {pm:>8.2f} {ps:>7.2f}  {sm:>8.4f} {ss:>7.4f}  {l2m:>8.5f} {lim:>8.5f}')
print(sep)
print()
"
