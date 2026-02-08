import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots(1, 1, figsize=(3.2, 1.8), dpi=300)

# Colors
blue_fill = '#DBEDF9'
blue_border = '#6CB4E0'
blue_dark = '#2980B9'
dark_text = '#2C3E50'
skip_color = '#90B8D4'

cx, cy = 1.6, 0.85
total_w = 2.6
h_max = 1.35
h_min = 0.45
gap = 0.06

# Encoder trapezoid
enc_poly = np.array([
    [cx - total_w/2,  cy + h_max/2],
    [cx - gap/2,      cy + h_min/2],
    [cx - gap/2,      cy - h_min/2],
    [cx - total_w/2,  cy - h_max/2],
])

# Decoder trapezoid
dec_poly = np.array([
    [cx + gap/2,      cy + h_min/2],
    [cx + total_w/2,  cy + h_max/2],
    [cx + total_w/2,  cy - h_max/2],
    [cx + gap/2,      cy - h_min/2],
])

ax.add_patch(plt.Polygon(enc_poly, closed=True,
    facecolor=blue_fill, edgecolor=blue_border, linewidth=2.0, zorder=1))
ax.add_patch(plt.Polygon(dec_poly, closed=True,
    facecolor=blue_fill, edgecolor=blue_border, linewidth=2.0, zorder=1))

# Bottleneck
bh = h_min * 0.6
ax.add_patch(plt.Rectangle((cx - gap/2, cy - bh/2), gap, bh,
    facecolor=blue_dark, edgecolor='#1B6B9A', linewidth=1.5, zorder=3))

# Label
ax.text(cx, cy - h_max/2 - 0.13, '3D Seg.', fontsize=9, fontweight='bold',
    ha='center', va='top', color=dark_text, fontfamily='sans-serif')

ax.set_xlim(cx - total_w/2 - 0.15, cx + total_w/2 + 0.15)
ax.set_ylim(cy - h_max/2 - 0.35, cy + h_max/2 + 0.25)
ax.set_aspect('equal')
ax.axis('off')

plt.tight_layout(pad=0.05)
plt.savefig('/home/user/3due/docs/network_icon.png', dpi=300,
    bbox_inches='tight', transparent=True, pad_inches=0.03)
plt.savefig('/home/user/3due/docs/network_icon.pdf',
    bbox_inches='tight', transparent=True, pad_inches=0.03)
print("Done")
