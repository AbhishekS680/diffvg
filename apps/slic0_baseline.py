# slic0_baseline.py
# SLIC0 superpixel segmentation as a reconstruction baseline

import numpy as np
import skimage.io
from skimage.segmentation import slic, mark_boundaries
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import os
os.makedirs('results/slic0', exist_ok=True)

N_SEGMENTS = 300  # target number of superpixels, not guaranteed

target = skimage.io.imread('imgs/fruit_basket.png') / 255.0
target = target[:, :, :3]  # keep RGB only

# SLIC0 = SLIC with slic_zero=True, removes the compactness parameter
segments = slic(target, n_segments=N_SEGMENTS, slic_zero=True, start_label=0)

# Reconstruct: fill each superpixel with its own average color
reconstructed = np.zeros_like(target)
for seg_id in np.unique(segments):
    mask = segments == seg_id
    reconstructed[mask] = target[mask].mean(axis=0)

# Error metric
mse = ((target - reconstructed) ** 2).mean()
print(f'N_segments requested: {N_SEGMENTS}')
print(f'N_segments actual: {len(np.unique(segments))}')  # SLIC0 doesn't guarantee the exact count
print(f'MSE: {mse:.6f}')

skimage.io.imsave('results/slic0/target.png', (target * 255).astype(np.uint8))
skimage.io.imsave('results/slic0/reconstructed.png', (reconstructed * 255).astype(np.uint8))

# Boundary overlay for visual inspection
boundary_img = mark_boundaries(target, segments, color=(1, 0, 0))
skimage.io.imsave('results/slic0/boundaries.png', (boundary_img * 255).astype(np.uint8))

# Standalone error heatmap
error_map = ((target - reconstructed) ** 2).mean(axis=2)
fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(error_map, cmap='inferno')
ax.set_title('Error heatmap')
ax.axis('off')
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
plt.savefig('results/slic0/error_heatmap.png', bbox_inches='tight', dpi=150)
plt.close(fig)
print('saved error_heatmap.png')

# All comparison image
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
axes[0].imshow(target); axes[0].set_title('Target'); axes[0].axis('off')
axes[1].imshow(reconstructed); axes[1].set_title(f'SLIC0 (N={len(np.unique(segments))})'); axes[1].axis('off')
im = axes[2].imshow(error_map, cmap='inferno'); axes[2].set_title('Error heatmap'); axes[2].axis('off')
fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
plt.savefig('results/slic0/all_comparison.png', bbox_inches='tight', dpi=150)
plt.close(fig)
print('saved all_comparison.png')