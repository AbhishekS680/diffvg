# median_filter_baseline.py
# Median filtering as a reconstruction baseline

import numpy as np
import skimage.io
from scipy.ndimage import median_filter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import os
os.makedirs('results/median_filter', exist_ok=True)

# neighborhood size
# bigger = more smoothing, like a blob
KERNEL_SIZE = 15

target = skimage.io.imread('imgs/fruit_basket.png') / 255.0
target = target[:, :, :3]  # keep RGB only

# Apply median filter per channel
# size=(k, k, 1) means to not blur across channels
reconstructed = median_filter(target, size=(KERNEL_SIZE, KERNEL_SIZE, 1))

# Error metric
mse = ((target - reconstructed) ** 2).mean()
print(f'Kernel size: {KERNEL_SIZE}')
print(f'MSE: {mse:.6f}')

skimage.io.imsave('results/median_filter/target.png', (target * 255).astype(np.uint8))
skimage.io.imsave('results/median_filter/reconstructed.png', (reconstructed * 255).astype(np.uint8))

# Error heatmap
error_map = ((target - reconstructed) ** 2).mean(axis=2)
fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(error_map, cmap='inferno')
ax.set_title('Error heatmap')
ax.axis('off')
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
plt.savefig('results/median_filter/error_heatmap.png', bbox_inches='tight', dpi=150)
plt.close(fig)
print('saved error_heatmap.png')

# All comparison image
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
axes[0].imshow(target); axes[0].set_title('Target'); axes[0].axis('off')
axes[1].imshow(reconstructed); axes[1].set_title(f'Median filter (k={KERNEL_SIZE})'); axes[1].axis('off')
im = axes[2].imshow(error_map, cmap='inferno'); axes[2].set_title('Error heatmap'); axes[2].axis('off')
fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
plt.savefig('results/median_filter/all_comparison.png', bbox_inches='tight', dpi=150)
plt.close(fig)
print('saved all_comparison.png')