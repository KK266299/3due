# file: src/core/ue_algos/spectral_visualizer.py
"""
Visualization utilities for Coherent Spectral UE.

Provides visualization of:
  - Spectral mask M (frequency domain)
  - Learnable spectral tensor P (magnitude and phase)
  - Generated noise delta (spatial domain)
  - Inter-slice coherence analysis
"""
from __future__ import annotations
from pathlib import Path
from typing import Optional, Tuple
import torch
import numpy as np

# Lazy import matplotlib to avoid issues in headless environments
def _get_plt():
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    return plt


def visualize_spectral_mask(
    M: torch.Tensor,
    save_path: Optional[Path] = None,
    title: str = "Spectral Mask M"
) -> None:
    """
    Visualize the 3D spectral mask M.

    Args:
        M: Spectral mask [D, H, W]
        save_path: Path to save the figure
        title: Figure title
    """
    plt = _get_plt()
    M_np = M.detach().cpu().numpy()
    D, H, W = M_np.shape

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(title, fontsize=14)

    # 1. Center slices in each dimension
    # XY plane (z=0, DC slice)
    ax = axes[0, 0]
    im = ax.imshow(M_np[0, :, :], cmap='viridis', vmin=0, vmax=1)
    ax.set_title(f'XY plane (kz=0, DC)')
    ax.set_xlabel('kx')
    ax.set_ylabel('ky')
    plt.colorbar(im, ax=ax)

    # XZ plane (y=0)
    ax = axes[0, 1]
    im = ax.imshow(M_np[:, 0, :], cmap='viridis', vmin=0, vmax=1, aspect='auto')
    ax.set_title(f'XZ plane (ky=0)')
    ax.set_xlabel('kx')
    ax.set_ylabel('kz')
    plt.colorbar(im, ax=ax)

    # YZ plane (x=0)
    ax = axes[0, 2]
    im = ax.imshow(M_np[:, :, 0], cmap='viridis', vmin=0, vmax=1, aspect='auto')
    ax.set_title(f'YZ plane (kx=0)')
    ax.set_xlabel('ky')
    ax.set_ylabel('kz')
    plt.colorbar(im, ax=ax)

    # 2. 1D profiles
    # Z-axis profile (kx=ky=0)
    ax = axes[1, 0]
    z_profile = M_np[:, 0, 0]
    ax.plot(np.arange(D) - D//2, np.fft.fftshift(z_profile), 'b-', linewidth=2)
    ax.set_title('Z-axis profile (kx=ky=0)')
    ax.set_xlabel('kz (centered)')
    ax.set_ylabel('M value')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.1)

    # Radial profile in XY plane (kz=0)
    ax = axes[1, 1]
    center_y, center_x = H // 2, W // 2
    M_xy_shifted = np.fft.fftshift(M_np[0, :, :])
    radii = []
    values = []
    for r in range(min(center_x, center_y)):
        # Sample at angle 0 (positive x direction)
        if center_x + r < W:
            radii.append(r / W)  # Normalized frequency
            values.append(M_xy_shifted[center_y, center_x + r])
    ax.plot(radii, values, 'r-', linewidth=2)
    ax.set_title('XY radial profile (kz=0)')
    ax.set_xlabel('Radial frequency (normalized)')
    ax.set_ylabel('M value')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.1)

    # 3D summary statistics
    ax = axes[1, 2]
    ax.axis('off')
    stats_text = f"""
    Spectral Mask Statistics:

    Shape: {M_np.shape}
    Min: {M_np.min():.4f}
    Max: {M_np.max():.4f}
    Mean: {M_np.mean():.4f}
    Non-zero ratio: {(M_np > 0.01).sum() / M_np.size:.2%}

    DC value (M[0,0,0]): {M_np[0,0,0]:.4f}
    """
    ax.text(0.1, 0.5, stats_text, fontsize=12, family='monospace',
            verticalalignment='center', transform=ax.transAxes)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()


def visualize_spectral_tensor_P(
    P_real: torch.Tensor,
    P_imag: torch.Tensor,
    M: torch.Tensor,
    save_path: Optional[Path] = None,
    channel: int = 0,
    title: str = "Spectral Tensor P"
) -> None:
    """
    Visualize the learnable spectral tensor P.

    Args:
        P_real: Real part [C, D, H, W]
        P_imag: Imaginary part [C, D, H, W]
        M: Spectral mask [D, H, W]
        save_path: Path to save the figure
        channel: Which channel to visualize
        title: Figure title
    """
    plt = _get_plt()

    P_r = P_real[channel].detach().cpu().numpy()
    P_i = P_imag[channel].detach().cpu().numpy()
    M_np = M.detach().cpu().numpy()

    # Compute magnitude and phase
    P_mag = np.sqrt(P_r**2 + P_i**2)
    P_phase = np.arctan2(P_i, P_r)

    # Masked versions
    P_mag_masked = P_mag * M_np
    P_phase_masked = P_phase * M_np

    D, H, W = P_r.shape

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle(f"{title} (channel {channel})", fontsize=14)

    # Row 1: XY plane (kz=0)
    ax = axes[0, 0]
    im = ax.imshow(np.fft.fftshift(P_mag[0]), cmap='hot')
    ax.set_title('|P| (kz=0)')
    plt.colorbar(im, ax=ax)

    ax = axes[0, 1]
    im = ax.imshow(np.fft.fftshift(P_phase[0]), cmap='twilight', vmin=-np.pi, vmax=np.pi)
    ax.set_title('∠P (kz=0)')
    plt.colorbar(im, ax=ax)

    ax = axes[0, 2]
    im = ax.imshow(np.fft.fftshift(P_mag_masked[0]), cmap='hot')
    ax.set_title('|M·P| (kz=0)')
    plt.colorbar(im, ax=ax)

    ax = axes[0, 3]
    im = ax.imshow(np.fft.fftshift(P_phase_masked[0]), cmap='twilight', vmin=-np.pi, vmax=np.pi)
    ax.set_title('∠(M·P) (kz=0)')
    plt.colorbar(im, ax=ax)

    # Row 2: XZ plane (ky=0)
    ax = axes[1, 0]
    im = ax.imshow(np.fft.fftshift(P_mag[:, 0, :], axes=0), cmap='hot', aspect='auto')
    ax.set_title('|P| (ky=0)')
    plt.colorbar(im, ax=ax)

    ax = axes[1, 1]
    im = ax.imshow(np.fft.fftshift(P_phase[:, 0, :], axes=0), cmap='twilight',
                   vmin=-np.pi, vmax=np.pi, aspect='auto')
    ax.set_title('∠P (ky=0)')
    plt.colorbar(im, ax=ax)

    ax = axes[1, 2]
    im = ax.imshow(np.fft.fftshift(P_mag_masked[:, 0, :], axes=0), cmap='hot', aspect='auto')
    ax.set_title('|M·P| (ky=0)')
    plt.colorbar(im, ax=ax)

    ax = axes[1, 3]
    im = ax.imshow(np.fft.fftshift(P_phase_masked[:, 0, :], axes=0), cmap='twilight',
                   vmin=-np.pi, vmax=np.pi, aspect='auto')
    ax.set_title('∠(M·P) (ky=0)')
    plt.colorbar(im, ax=ax)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()


def visualize_noise_delta(
    delta: torch.Tensor,
    roi_mask: Optional[torch.Tensor] = None,
    save_path: Optional[Path] = None,
    channel: int = 0,
    title: str = "Generated Noise Delta"
) -> None:
    """
    Visualize the generated noise in spatial domain.

    Args:
        delta: Noise tensor [C, D, H, W] or [B, C, D, H, W]
        roi_mask: Optional ROI mask [D, H, W] or [B, 1, D, H, W]
        save_path: Path to save the figure
        channel: Which channel to visualize
        title: Figure title
    """
    plt = _get_plt()

    if delta.dim() == 5:
        delta = delta[0]  # Take first batch
    if roi_mask is not None and roi_mask.dim() == 5:
        roi_mask = roi_mask[0, 0]

    delta_np = delta[channel].detach().cpu().numpy()
    D, H, W = delta_np.shape

    n_slices = min(6, D)
    slice_indices = np.linspace(0, D-1, n_slices, dtype=int)

    fig, axes = plt.subplots(3, n_slices, figsize=(3*n_slices, 9))
    fig.suptitle(f"{title} (channel {channel})", fontsize=14)

    vmax = np.abs(delta_np).max()
    vmin = -vmax

    # Row 1: Delta slices
    for i, z in enumerate(slice_indices):
        ax = axes[0, i]
        im = ax.imshow(delta_np[z], cmap='RdBu_r', vmin=vmin, vmax=vmax)
        ax.set_title(f'z={z}')
        ax.axis('off')
    plt.colorbar(im, ax=axes[0, :].tolist(), shrink=0.8, label='δ value')

    # Row 2: ROI mask slices (if provided)
    if roi_mask is not None:
        roi_np = roi_mask.detach().cpu().numpy()
        for i, z in enumerate(slice_indices):
            ax = axes[1, i]
            im = ax.imshow(roi_np[z], cmap='gray', vmin=0, vmax=1)
            ax.set_title(f'ROI z={z}')
            ax.axis('off')
        plt.colorbar(im, ax=axes[1, :].tolist(), shrink=0.8, label='ROI')
    else:
        for ax in axes[1, :]:
            ax.axis('off')
            ax.text(0.5, 0.5, 'No ROI', ha='center', va='center', transform=ax.transAxes)

    # Row 3: Delta histogram and inter-slice correlation
    ax = axes[2, 0]
    ax.hist(delta_np.flatten(), bins=100, density=True, alpha=0.7)
    ax.set_title('Delta histogram')
    ax.set_xlabel('Value')
    ax.set_ylabel('Density')

    ax = axes[2, 1]
    # Inter-slice correlation
    correlations = []
    for z in range(D - 1):
        s1 = delta_np[z].flatten()
        s2 = delta_np[z + 1].flatten()
        if np.std(s1) > 1e-8 and np.std(s2) > 1e-8:
            corr = np.corrcoef(s1, s2)[0, 1]
            correlations.append(corr)
    ax.plot(correlations, 'b-', linewidth=1.5)
    ax.axhline(y=np.mean(correlations) if correlations else 0, color='r', linestyle='--',
               label=f'mean={np.mean(correlations):.3f}' if correlations else 'N/A')
    ax.set_title('Inter-slice correlation')
    ax.set_xlabel('Slice index')
    ax.set_ylabel('Correlation')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Statistics
    ax = axes[2, 2]
    ax.axis('off')
    stats_text = f"""
    Delta Statistics:

    Shape: {delta_np.shape}
    Min: {delta_np.min():.6f}
    Max: {delta_np.max():.6f}
    Mean: {delta_np.mean():.6f}
    Std: {delta_np.std():.6f}
    L∞: {np.abs(delta_np).max():.6f}

    Mean inter-slice corr: {np.mean(correlations):.4f}
    """
    ax.text(0.1, 0.5, stats_text, fontsize=10, family='monospace',
            verticalalignment='center', transform=ax.transAxes)

    # XZ and YZ cross-sections
    ax = axes[2, 3]
    ax.imshow(delta_np[:, H//2, :], cmap='RdBu_r', vmin=vmin, vmax=vmax, aspect='auto')
    ax.set_title('XZ cross-section (y=H/2)')
    ax.set_xlabel('x')
    ax.set_ylabel('z')

    ax = axes[2, 4]
    ax.imshow(delta_np[:, :, W//2], cmap='RdBu_r', vmin=vmin, vmax=vmax, aspect='auto')
    ax.set_title('YZ cross-section (x=W/2)')
    ax.set_xlabel('y')
    ax.set_ylabel('z')

    # Hide remaining axes
    for i in range(5, n_slices):
        axes[2, i].axis('off')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()


def visualize_frequency_spectrum(
    delta: torch.Tensor,
    save_path: Optional[Path] = None,
    channel: int = 0,
    title: str = "Noise Frequency Spectrum"
) -> None:
    """
    Visualize the frequency spectrum of the generated noise.

    Args:
        delta: Noise tensor [C, D, H, W]
        save_path: Path to save the figure
        channel: Which channel to visualize
        title: Figure title
    """
    plt = _get_plt()

    delta_np = delta[channel].detach().cpu().numpy()
    D, H, W = delta_np.shape

    # Compute 3D FFT
    delta_fft = np.fft.fftn(delta_np)
    delta_mag = np.abs(delta_fft)
    delta_mag_log = np.log1p(delta_mag)  # Log scale for better visualization

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f"{title} (channel {channel})", fontsize=14)

    # Row 1: Magnitude spectrum slices
    ax = axes[0, 0]
    im = ax.imshow(np.fft.fftshift(delta_mag_log[0]), cmap='hot')
    ax.set_title('|FFT(δ)| XY plane (kz=0)')
    plt.colorbar(im, ax=ax)

    ax = axes[0, 1]
    im = ax.imshow(np.fft.fftshift(delta_mag_log[:, 0, :], axes=0), cmap='hot', aspect='auto')
    ax.set_title('|FFT(δ)| XZ plane (ky=0)')
    plt.colorbar(im, ax=ax)

    ax = axes[0, 2]
    im = ax.imshow(np.fft.fftshift(delta_mag_log[:, :, 0], axes=0), cmap='hot', aspect='auto')
    ax.set_title('|FFT(δ)| YZ plane (kx=0)')
    plt.colorbar(im, ax=ax)

    # Row 2: 1D power spectra
    ax = axes[1, 0]
    z_power = delta_mag[:, 0, 0]
    freqs = np.fft.fftfreq(D)
    ax.semilogy(np.fft.fftshift(freqs), np.fft.fftshift(z_power), 'b-', linewidth=1.5)
    ax.set_title('Z-axis power spectrum')
    ax.set_xlabel('Frequency (normalized)')
    ax.set_ylabel('Power (log)')
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    # Radial power spectrum in XY plane
    center = H // 2
    radial_bins = np.zeros(center)
    radial_counts = np.zeros(center)
    mag_xy = np.fft.fftshift(delta_mag[0])
    for y in range(H):
        for x in range(W):
            r = int(np.sqrt((y - center)**2 + (x - W//2)**2))
            if r < center:
                radial_bins[r] += mag_xy[y, x]
                radial_counts[r] += 1
    radial_power = radial_bins / np.maximum(radial_counts, 1)
    ax.semilogy(np.arange(center) / W, radial_power, 'r-', linewidth=1.5)
    ax.set_title('XY radial power spectrum (kz=0)')
    ax.set_xlabel('Radial frequency (normalized)')
    ax.set_ylabel('Power (log)')
    ax.grid(True, alpha=0.3)

    # Statistics
    ax = axes[1, 2]
    ax.axis('off')

    # Compute energy distribution
    total_energy = (delta_mag**2).sum()
    low_freq_mask = np.zeros_like(delta_mag, dtype=bool)
    low_freq_mask[:D//4, :H//4, :W//4] = True
    low_freq_mask[-(D//4):, :H//4, :W//4] = True
    low_freq_mask[:D//4, -(H//4):, :W//4] = True
    low_freq_mask[:D//4, :H//4, -(W//4):] = True
    # ... (simplified, just corners)
    low_freq_energy = (delta_mag[low_freq_mask]**2).sum() if low_freq_mask.any() else 0

    stats_text = f"""
    Frequency Spectrum Statistics:

    Total energy: {total_energy:.2e}
    DC component: {delta_mag[0,0,0]:.4f}
    Max frequency: {delta_mag.max():.4f}

    Energy concentration:
    - Low freq (corners): {low_freq_energy/total_energy:.2%}
    """
    ax.text(0.1, 0.5, stats_text, fontsize=11, family='monospace',
            verticalalignment='center', transform=ax.transAxes)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()


def create_all_visualizations(
    noise_gen,  # SpectralNoiseGenerator
    save_dir: Path,
    epoch: int,
    roi_mask: Optional[torch.Tensor] = None,
) -> None:
    """
    Create all visualizations and save to directory.

    Args:
        noise_gen: SpectralNoiseGenerator instance
        save_dir: Directory to save visualizations
        epoch: Current epoch number
        roi_mask: Optional ROI mask
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        # Generate noise
        delta = noise_gen()  # [C, D, H, W]
        M = noise_gen.spectral_mask
        P_real = noise_gen.P_real
        P_imag = noise_gen.P_imag

        # 1. Spectral mask
        visualize_spectral_mask(
            M,
            save_path=save_dir / f"epoch_{epoch:04d}_spectral_mask.png",
            title=f"Spectral Mask M (Epoch {epoch})"
        )

        # 2. Spectral tensor P
        visualize_spectral_tensor_P(
            P_real, P_imag, M,
            save_path=save_dir / f"epoch_{epoch:04d}_spectral_P.png",
            channel=0,
            title=f"Spectral Tensor P (Epoch {epoch})"
        )

        # 3. Noise delta
        roi_for_vis = roi_mask[0, 0] if roi_mask is not None and roi_mask.dim() == 5 else roi_mask
        visualize_noise_delta(
            delta,
            roi_mask=roi_for_vis,
            save_path=save_dir / f"epoch_{epoch:04d}_noise_delta.png",
            channel=0,
            title=f"Generated Noise δ (Epoch {epoch})"
        )

        # 4. Frequency spectrum of noise
        visualize_frequency_spectrum(
            delta,
            save_path=save_dir / f"epoch_{epoch:04d}_noise_spectrum.png",
            channel=0,
            title=f"Noise Frequency Spectrum (Epoch {epoch})"
        )