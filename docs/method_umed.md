# UMed: Unlearnable Medical Image Segmentation

**Paper**: [UMed: Unlearnable Examples for Medical Image Segmentation via Contour- and Texture-Aware Perturbations](https://arxiv.org/abs/2403.14250)

---

## Method Overview

UMed protects medical image segmentation datasets by injecting imperceptible, contour- and texture-aware perturbations that prevent unauthorized deep learning models from learning meaningful segmentation information.

### Core Architecture

UMed uses two trainable perturbation generators:

| Component | Network | Target Region | Clipping Strategy |
|-----------|---------|---------------|-------------------|
| **G_c** (Contour Perturbator) | CDC U-Net (Central-Difference-Aware Conv) | Boundary band (dilate - erode) | Fixed `[-eps, eps]` masked to contour |
| **G_t** (Texture Perturbator) | Standard U-Net | Interior ROI (excl. contour) | Adaptive LBP-scaled `[-eps*lbp, eps*lbp]` |

**Final protected image**:

```
x_p = clip[0, 1](x + delta_c + delta_t)
```

### Bi-Level Optimization (Algorithm 1)

```
for epoch = 1 to E:
    for batch (x, y) in D_clean:
        delta_c = G_c(x) * contour_mask
        delta_t = G_t(x) * interior_mask (LBP adaptive clipping)
        x_p = clip[0,1](x + delta_c + delta_t)

        if iteration % surrogate_step != 0:
            # G-step: Update generators G_c, G_t
            theta_Gc -= lr_g * grad(L_seg(F_s(x_p), y), theta_Gc)
            theta_Gt -= lr_g * grad(L_seg(F_s(x_p), y), theta_Gt)
        else:
            # S-step: Update surrogate F_s
            theta_Fs -= lr_s * grad(L_seg(F_s(x_p), y), theta_Fs)
```

### Key Technical Components

1. **Central-Difference-Aware Convolution (CDC)**: Modified 3D convolution that combines vanilla and central-difference operations to enhance contour feature capture.

2. **LBP Texture Map**: Simplified 3D Local Binary Pattern computes texture intensity at each voxel by comparing with 6-connected neighbours. Higher texture regions permit larger perturbation bounds.

3. **Morphological Boundary Extraction**: `boundary = dilate(fg_mask) - erode(fg_mask)` using max-pool based morphological operations. `interior = fg_mask - boundary`.

---

## Paper Hyperparameters

| Parameter | Paper Value | Config Key |
|-----------|-------------|------------|
| Perturbation bound (epsilon) | 4/255 (~0.0157) | `ue.algorithm.params.epsilon` |
| Learning rate (generators) | 1e-4 | `ue.generator_optimizer.lr` |
| Learning rate (surrogate) | 1e-4 | `ue.surrogates.s_seg.optimizer.lr` |
| Training epochs | 100 | `training.epochs` |
| Surrogate update frequency | Every 5 iterations | `ue.algorithm.params.surrogate_step` |
| Boundary width (kernel radius) | 3 voxels | `ue.algorithm.params.boundary_width` |
| CDC theta (initial) | 0.5 | `ue.contour_unet.cdc_theta` |

---

## File Structure

```
src/core/ue_algos/umed.py          # UMed algorithm implementation
configs/method/umed.yaml            # Hydra configuration
docs/method_umed.md                 # This documentation
```

---

## Run Commands

### Basic Run (Paper Defaults)

```bash
python ue_generate.py \
    dataset=brats19 \
    task.run_name=umed_default \
    method=umed \
    task=brats19_ue \
    training.epochs=100 \
    training.batch_size=8 \
    training.gpu_ids=[0] \
    ue.key.type=samplewise \
    ue.key.from=field \
    ue.key.field=case_id \
    ue.algorithm.params.epsilon=0.0156863 \
    ue.algorithm.params.boundary_width=3 \
    ue.algorithm.params.surrogate_step=5 \
    ue.io.save_from_epoch=50 \
    ue.io.save_every=10
```

### Comparative Experiment (with custom GPU and naming)

```bash
python ue_generate.py \
    dataset=brats19 \
    task.run_name=umed_eps4_bw3 \
    method=umed \
    task=brats19_ue \
    training.epochs=100 \
    training.batch_size=8 \
    training.gpu_ids=[1] \
    ue.key.type=samplewise \
    ue.key.from=field \
    ue.key.field=case_id \
    ue.algorithm.params.epsilon=0.0156863 \
    ue.algorithm.params.boundary_width=3 \
    ue.algorithm.params.surrogate_step=5 \
    ue.algorithm.params.noise_step=1 \
    ue.io.save_from_epoch=50 \
    ue.io.save_every=10
```

### Ablation: Larger Perturbation Budget (8/255)

```bash
python ue_generate.py \
    dataset=brats19 \
    task.run_name=umed_eps8_bw3 \
    method=umed \
    task=brats19_ue \
    training.epochs=100 \
    training.batch_size=8 \
    training.gpu_ids=[1] \
    ue.key.type=samplewise \
    ue.key.from=field \
    ue.key.field=case_id \
    ue.algorithm.params.epsilon=0.0313725 \
    ue.algorithm.params.boundary_width=3 \
    ue.algorithm.params.surrogate_step=5 \
    ue.io.save_from_epoch=50 \
    ue.io.save_every=10
```

### Ablation: Wider Boundary (boundary_width=5)

```bash
python ue_generate.py \
    dataset=brats19 \
    task.run_name=umed_eps4_bw5 \
    method=umed \
    task=brats19_ue \
    training.epochs=100 \
    training.batch_size=8 \
    training.gpu_ids=[1] \
    ue.key.type=samplewise \
    ue.key.from=field \
    ue.key.field=case_id \
    ue.algorithm.params.epsilon=0.0156863 \
    ue.algorithm.params.boundary_width=5 \
    ue.algorithm.params.surrogate_step=5 \
    ue.io.save_from_epoch=50 \
    ue.io.save_every=10
```

---

## Implementation Notes

### Differences from Original Paper (2D -> 3D Adaptation)

1. **3D Convolutions**: All convolutions (including CDC) are 3D for volumetric BraTS data.
2. **LBP Simplification**: Uses 6-connected 3D neighbours instead of the original 2D 8-neighbour LBP. Result is normalised to [0, 1].
3. **Morphological Operations**: Uses 3D max-pool based dilation/erosion instead of 2D operations.
4. **Generator Architecture**: G_c uses custom CDCUNet3d; G_t uses MONAI UNet. Both follow the same encoder-decoder paradigm.

### Storage Consistency

- Noise storage follows the same `noise_backend.commit_batch()` / `batch_noise()` pattern as all other methods.
- IO config uses `strategy: files`, `dtype: int8`, consistent with `unet_roi_noise` and `unet_boundary_noise`.
- Epoch-based saving controlled by `save_from_epoch` and `save_every`.

### Loss Function

Segmentation loss = DiceCE Loss (Dice + Cross-Entropy), consistent with the existing surrogate training pipeline and the paper's specification of `L_ce + L_dice`.
