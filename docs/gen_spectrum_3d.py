import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(123)

fig, ax = plt.subplots(1, 1, figsize=(3.5, 3.5), dpi=300)

# 3D axes origin
ox, oy = 1.75, 1.4

# Axis lengths
fz_len = 1.5   # vertical (up)
fy_len = 1.6   # horizontal (right)
fx_len = 1.1   # diagonal (left-down, perspective)

# Axis endpoints
fz_end = (ox, oy + fz_len)
fy_end = (ox + fy_len, oy)
fx_end = (ox - fx_len * 0.7, oy - fx_len * 0.5)

# Draw axes
arrow_kw = dict(head_width=0.06, head_length=0.05, fc='#2C3E50', ec='#2C3E50', linewidth=1.5)
ax.annotate('', xy=fz_end, xytext=(ox, oy),
            arrowprops=dict(arrowstyle='->', color='#2C3E50', lw=1.8))
ax.annotate('', xy=fy_end, xytext=(ox, oy),
            arrowprops=dict(arrowstyle='->', color='#2C3E50', lw=1.8))
ax.annotate('', xy=fx_end, xytext=(ox, oy),
            arrowprops=dict(arrowstyle='->', color='#2C3E50', lw=1.8))

# Dashed grid lines on fxy plane
for i in range(3):
    frac = 0.3 + i * 0.25
    # Lines parallel to fy
    px = ox + (fx_end[0] - ox) * frac
    py = oy + (fx_end[1] - oy) * frac
    ax.plot([px, px + fy_len * 0.6], [py, py],
            color='#BDC3C7', linewidth=0.5, linestyle='--', zorder=0)
    # Lines parallel to fx
    px2 = ox + fy_len * frac * 0.6
    ax.plot([px2, px2 + (fx_end[0] - ox) * 0.5],
            [oy, oy + (fx_end[1] - oy) * 0.5],
            color='#BDC3C7', linewidth=0.5, linestyle='--', zorder=0)

# Blue cloud (clean logits spectrum) - compact, near origin, low frequency
n_blue = 400
bx = np.random.randn(n_blue) * 0.18 + ox - 0.3
by = np.random.randn(n_blue) * 0.22 + oy + 0.15
bsizes = np.random.uniform(3, 25, n_blue)
# Gaussian falloff for alpha
bdist = np.sqrt((bx - (ox - 0.3))**2 + (by - (oy + 0.15))**2)
balpha = np.clip(0.5 - bdist * 0.8, 0.05, 0.5)
ax.scatter(bx, by, s=bsizes, c='#2471A3', alpha=balpha, edgecolors='none', zorder=2)
# Dense core
n_core = 150
bcx = np.random.randn(n_core) * 0.08 + ox - 0.3
bcy = np.random.randn(n_core) * 0.10 + oy + 0.15
ax.scatter(bcx, bcy, s=np.random.uniform(5, 20, n_core),
           c='#1A5276', alpha=0.6, edgecolors='none', zorder=3)

# Red cloud (perturbed logits spectrum) - dispersed, spread out
n_red = 600
rx = np.random.randn(n_red) * 0.45 + ox + 0.5
ry = np.random.randn(n_red) * 0.40 + oy + 0.55
rsizes = np.random.uniform(3, 30, n_red)
rdist = np.sqrt((rx - (ox + 0.5))**2 + (ry - (oy + 0.55))**2)
ralpha = np.clip(0.45 - rdist * 0.5, 0.03, 0.45)
ax.scatter(rx, ry, s=rsizes, c='#C0392B', alpha=ralpha, edgecolors='none', zorder=2)
# Some outlier scattered points (high frequency spread)
n_out = 80
rox = np.random.randn(n_out) * 0.7 + ox + 0.5
roy = np.random.randn(n_out) * 0.6 + oy + 0.55
ax.scatter(rox, roy, s=np.random.uniform(2, 12, n_out),
           c='#E74C3C', alpha=0.25, edgecolors='none', zorder=2)

# Green double arrow between clouds (maximize divergence)
arrow_cx = ox + 0.08
arrow_cy = oy + 0.45
ax.annotate('', xy=(arrow_cx + 0.35, arrow_cy + 0.2),
            xytext=(arrow_cx - 0.15, arrow_cy - 0.1),
            arrowprops=dict(arrowstyle='<->', color='#27AE60', lw=2.8,
                           connectionstyle='arc3,rad=0'))

# Axis labels (keep only f_z, f_y, f_x - no Chinese text)
ax.text(fz_end[0] - 0.12, fz_end[1] + 0.05, r'$f_z$',
        fontsize=13, fontweight='bold', color='#2C3E50', fontfamily='serif')
ax.text(fy_end[0] + 0.05, fy_end[1] - 0.02, r'$f_y$',
        fontsize=13, fontweight='bold', color='#2C3E50', fontfamily='serif')
ax.text(fx_end[0] - 0.15, fx_end[1] - 0.08, r'$f_x$',
        fontsize=13, fontweight='bold', color='#2C3E50', fontfamily='serif')

ax.set_xlim(-0.2, 3.6)
ax.set_ylim(-0.1, 3.3)
ax.set_aspect('equal')
ax.axis('off')

plt.tight_layout(pad=0.05)
plt.savefig('/home/user/3due/docs/spectrum_3d.png', dpi=300,
    bbox_inches='tight', transparent=True, pad_inches=0.03)
plt.savefig('/home/user/3due/docs/spectrum_3d.pdf',
    bbox_inches='tight', transparent=True, pad_inches=0.03)
print("Done")
