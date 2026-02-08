import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

np.random.seed(42)

fig, ax = plt.subplots(1, 1, figsize=(3.2, 3.0), dpi=300)

n_slices = 4
slice_w, slice_h = 1.8, 1.8
dx, dy = 0.32, 0.32

slice_colors = ['#0B3D6B', '#1A5C8E', '#1F78B4', '#3498DB']
bg_color = '#F0F5FA'

for i in range(n_slices):
    x0 = i * dx
    y0 = i * dy
    zo = i + 1

    # Slice background
    rect = patches.FancyBboxPatch(
        (x0, y0), slice_w, slice_h,
        boxstyle="round,pad=0.015",
        facecolor=bg_color, edgecolor='#8EAEC0',
        linewidth=0.9, zorder=zo, alpha=0.93
    )
    ax.add_patch(rect)

    color = slice_colors[i]
    # ROI center
    cx_r = x0 + slice_w * 0.52
    cy_r = y0 + slice_h * 0.50

    def in_slice(px, py):
        return (px > x0 + 0.06) & (px < x0 + slice_w - 0.06) & \
               (py > y0 + 0.06) & (py < y0 + slice_h - 0.06)

    if i == 0:
        # Dense random scatter
        n_pts = 600
        rx = np.random.randn(n_pts) * 0.38 + cx_r
        ry = np.random.randn(n_pts) * 0.38 + cy_r
        sizes = np.random.uniform(1.0, 8.0, n_pts)
        m = in_slice(rx, ry)
        ax.scatter(rx[m], ry[m], s=sizes[m], c=color,
                   alpha=0.7, edgecolors='none', zorder=zo + 0.5)

    elif i == 1:
        # Horizontal stripe noise
        for j in range(30):
            ly = y0 + 0.1 + (slice_h - 0.2) * j / 30
            dist = abs(ly - cy_r)
            if dist < 0.6:
                strength = 1.0 - dist / 0.6
                lx_s = cx_r - 0.55 * strength + np.random.uniform(-0.05, 0.05)
                lx_e = cx_r + 0.55 * strength + np.random.uniform(-0.05, 0.05)
                lx_s = max(lx_s, x0 + 0.06)
                lx_e = min(lx_e, x0 + slice_w - 0.06)
                lw = np.random.uniform(0.8, 2.2) * strength
                ax.plot([lx_s, lx_e], [ly, ly], color=color,
                        linewidth=lw, alpha=0.6 * strength, zorder=zo + 0.5)

    elif i == 2:
        # Diagonal cross-hatch
        for j in range(25):
            offset = -1.0 + j * 0.085
            x1 = cx_r + offset - 0.45
            y1 = cy_r - 0.55
            x2 = cx_r + offset + 0.45
            y2 = cy_r + 0.55
            x1c, x2c = max(x1, x0+0.06), min(x2, x0+slice_w-0.06)
            y1c, y2c = max(y1, y0+0.06), min(y2, y0+slice_h-0.06)
            if x1c < x2c and y1c < y2c:
                ax.plot([x1c, x2c], [y1c, y2c], color=color,
                        linewidth=0.9, alpha=0.55, zorder=zo + 0.5)
                ax.plot([x1c, x2c], [y2c, y1c], color=color,
                        linewidth=0.9, alpha=0.45, zorder=zo + 0.5)

    elif i == 3:
        # Concentric rings
        for r in np.arange(0.05, 0.65, 0.055):
            alpha_r = 0.65 - r * 0.4
            circle = patches.Circle((cx_r, cy_r), r,
                fill=False, edgecolor=color,
                linewidth=1.2, alpha=max(alpha_r, 0.25), zorder=zo + 0.5)
            ax.add_patch(circle)
            circle.set_clip_path(rect)

ax.set_xlim(-0.1, (n_slices-1) * dx + slice_w + 0.1)
ax.set_ylim(-0.1, (n_slices-1) * dy + slice_h + 0.1)
ax.set_aspect('equal')
ax.axis('off')

plt.tight_layout(pad=0.05)
plt.savefig('/home/user/3due/docs/noise_delta.png', dpi=300,
    bbox_inches='tight', transparent=True, pad_inches=0.03)
plt.savefig('/home/user/3due/docs/noise_delta.pdf',
    bbox_inches='tight', transparent=True, pad_inches=0.03)
print("Done")
