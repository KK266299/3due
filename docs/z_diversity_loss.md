# Z-axis Diversity Loss for Learnable Cutoff Frequency

## Overview

The `noise_slice_frequence_learnable` algorithm now supports an optional **Z-axis Diversity Loss** that encourages inter-slice diversity in the generated noise. This loss maximizes the L2 difference between adjacent slices in the frequency domain.

## Configuration

Add the following parameter to enable z-axis diversity regularization:

```yaml
ue.algorithm.params.z_diversity_weight: 0.1  # Set to 0 to disable (default)
```

- `z_diversity_weight = 0.0` (default): Disabled, no z-diversity loss
- `z_diversity_weight > 0`: Enabled, the loss encourages inter-slice variation

## How It Works

The z-diversity loss is computed as follows:

1. Apply 2D FFT on each slice (xy-plane) of the noise tensor
2. Compute the magnitude spectrum for each slice
3. Calculate L2 difference between adjacent slices along the z-axis
4. Average over all slice pairs, channels, and batches

Since we want to **maximize** diversity, we add the **negative** of this value to the total loss:

```
total_loss = seg_loss + z_diversity_weight * (-z_diversity)
```

## Command Line Examples

### Basic Usage (z_diversity disabled by default)

```bash
python ue_generate.py \
    dataset=flare21 \
    task=flare21_ue \
    task.run_name=noise_slice_freq_learnable \
    method=noise_slice_frequence_learnable \
    training.epochs=100 \
    training.batch_size=8 \
    training.gpu_ids=[0] \
    ue.key.type=samplewise \
    ue.key.from=field \
    ue.key.field=case_id \
    ue.algorithm.params.epsilon=0.0156863 \
    ue.algorithm.params.noise_step=1 \
    ue.algorithm.params.surrogate_step=10 \
    ue.algorithm.params.roi_aware=true \
    ue.algorithm.params.soft_edge=false \
    ue.io.save_from_epoch=50 \
    ue.io.save_every=10 \
    ue.surrogates.s_seg.in_channels=1 \
    ue.surrogates.s_seg.num_classes=5
```

### With Z-Diversity Loss Enabled

```bash
python ue_generate.py \
    dataset=flare21 \
    task=flare21_ue \
    task.run_name=noise_slice_freq_learnable_zdiv \
    method=noise_slice_frequence_learnable \
    training.epochs=100 \
    training.batch_size=8 \
    training.gpu_ids=[0] \
    ue.key.type=samplewise \
    ue.key.from=field \
    ue.key.field=case_id \
    ue.algorithm.params.epsilon=0.0156863 \
    ue.algorithm.params.noise_step=1 \
    ue.algorithm.params.surrogate_step=10 \
    ue.algorithm.params.roi_aware=true \
    ue.algorithm.params.soft_edge=false \
    ue.algorithm.params.z_diversity_weight=0.1 \
    ue.io.save_from_epoch=50 \
    ue.io.save_every=10 \
    ue.surrogates.s_seg.in_channels=1 \
    ue.surrogates.s_seg.num_classes=5
```

## Output Metrics

When z-diversity is enabled, the following additional metrics are logged:

| Metric | Description |
|--------|-------------|
| `z_diversity` | Mean inter-slice L2 difference in frequency domain (always logged) |
| `z_diversity_loss` | The weighted z-diversity loss term (only when `z_diversity_weight > 0`) |

## Parameter Reference

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `z_diversity_weight` | float | 0.0 | Weight for z-axis diversity loss. Set to 0 to disable. |
| `epsilon` | float | 8/255 | L-infinity bound for noise perturbation |
| `noise_step` | int | 1 | Number of noise update steps per batch |
| `surrogate_step` | int | 10 | Number of surrogate update steps per noise step |
| `roi_aware` | bool | true | Apply noise only to ROI regions |
| `soft_edge` | bool | true | Use soft edge for ROI mask |
| `z_cutoff_low` | float | 0.1 | Initial z-axis high-pass cutoff frequency |
| `xy_cutoff_high` | float | 0.3 | Initial xy-plane low-pass cutoff frequency |
| `cutoff_lr_scale` | float | 1.0 | Learning rate scale for cutoff parameters |
