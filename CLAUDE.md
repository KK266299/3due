# Learnable Filter Cutoff — Design Notes

## What was done

File: `src/core/ue_algos/noise_slice_frequence_h_l_pass.py`
Config: `configs/method/noise_slice_frequence_h_l_pass.yaml`

### Problem
The frequency domain filter for 3D noise generation uses two cutoff frequencies:
- `z_cutoff_low`: Z-axis high-pass (inter-slice diversity)
- `xy_cutoff_high`: XY-plane low-pass (intra-slice smoothness)

These were static constants from YAML. We want them to be **learnable, per-sample** values.

### Solution: Separate CutoffPredictor + per-sample gradient accumulation

Architecture (matches the diagram):
```
Image x ──→ NoiseUNet ──→ noise δ
   │
   └──→ CutoffPredictor ──→ (z_cutoff, xy_cutoff)
                │
noise δ ──→ FreqConstraint(cutoffs) ──→ filtered δ
                                           │
                                      DiceCE Loss
                                      ╱         ╲
                              opt_unet.step()  opt_cutoff.step()
```

- **NoiseUNet**: unchanged C-channel output, produces noise `[B, C, D, H, W]`
- **CutoffPredictor**: lightweight standalone network (AdaptiveAvgPool3d(4) → MLP → sigmoid → range map), produces `z_cutoff [B]`, `xy_cutoff [B]`
- **FreqConstraint**: uses sigmoid-based differentiable mask (not torch.where), separable M_z × M_xy
- **Two optimizers** (`opt_unet`, `opt_cutoff`), both updated from the same DiceCE loss
- **Toggle**: `ue.algorithm.params.learnable_cutoff: true/false`

### OOM mitigation: per-sample gradient accumulation

3D medical images are large. Backpropagating through FFT with per-sample masks on a full batch causes OOM.

Solution: **per-sample gradient accumulation loop**
```python
opt_unet.zero_grad()
opt_cutoff.zero_grad()
for i in range(B):
    noise_i = noise_unet(x_i)              # with grad
    z_c_i, xy_c_i = cutoff_predictor(x_i)  # with grad
    filtered_i = freq_constraint(noise_i, z_c_i, xy_c_i)  # FFT with grad
    loss_i = DiceCE(...) / B
    loss_i.backward()   # graph freed, memory reclaimed for this sample
opt_unet.step()         # accumulated gradient from all B samples
opt_cutoff.step()       # accumulated gradient from all B samples
```
- Peak memory = 1 sample (not B)
- Both networks get full-batch averaged gradients
- Static path (`learnable_cutoff=false`) still uses batch-level forward as before

### Key design decisions

1. **Separate CutoffPredictor (not merged into UNet)**
   - Two independent optimizers allow different learning rates
   - CutoffPredictor is tiny (~4K params), UNet is large
   - Clear separation of concerns

2. **Why sigmoid for mask instead of torch.where?**
   - `torch.where` is non-differentiable at the boundary
   - `sigmoid((freq - cutoff) / sigma)` is smooth and differentiable w.r.t. cutoff

3. **Why separable mask application?**
   - M_z depends only on z-freq → shape [B,1,D,1,1]
   - M_xy depends only on xy-freq → shape [B,1,1,H,W]
   - Applied sequentially via broadcasting, never materializing full [B,D,H,W]

4. **Why per-sample loop?**
   - Even with gradient checkpointing, autograd must save the full complex FFT tensor
   - Per-sample loop guarantees peak memory = 1 sample regardless of batch size

5. **Initialization**: CutoffPredictor MLP last layer weight=0, bias=logit(default) so initial output matches YAML defaults (0.1, 0.3)

### Files changed
- `src/core/ue_algos/noise_slice_frequence_h_l_pass.py` — main implementation
- `configs/method/noise_slice_frequence_h_l_pass.yaml` — added `learnable_cutoff` + `cutoff_lr_scale`
- `docs/learnable_cutoff_explanation.md` — detailed explanation (may be outdated, this file is canonical)
