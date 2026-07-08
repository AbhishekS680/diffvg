# shepard_segmented.py
# Per-segment Shepard reconstruction using Mean Shift segmentation

import pydiffvg
import torch
import skimage.io
import skimage.color
import numpy as np # Used for array math
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.cluster import MeanShift
import os

# --- Parameters ---
N_per_segment = 10 # control points per segment
q = 3.0
iters = 100
seg_size = 0.2 # controls segment size, a larger value means fewer but bigger segments

# --- Load target image ---
target_np = skimage.io.imread('imgs/fruit_basket.png').astype(np.float32) / 255.0
target_np = target_np[:, :, :3]
canvas_height, canvas_width = target_np.shape[0], target_np.shape[1]
print('target shape:', target_np.shape)

os.makedirs('results/shepard_segmented', exist_ok=True)
pydiffvg.imwrite(torch.from_numpy(target_np), 'results/shepard_segmented/target.png', gamma=1.0)

# --- Running Mean Shift segmentation ---
print('Running mean shift segmentation...')
h, w = target_np.shape[:2] # Getting image size

# Creates a 2D array for every pixel and stores x, y, r, g, b
coords = np.column_stack([
    np.indices((h, w))[1].ravel() / w, # x
    np.indices((h, w))[0].ravel() / h, # y
    target_np.reshape(-1, 3) # RGB values
])

start = time.time()
ms = MeanShift(bandwidth=seg_size, bin_seeding=True)
ms.fit(coords)
labels = ms.labels_ # Assigns a segment number to each pixel
n_segments = len(np.unique(labels))
print(f'Segmentation done in {time.time()-start:.1f}s, {n_segments} segments found')

# Save segmentation visualization
labels_2d = labels.reshape(h, w)
seg_viz = skimage.color.label2rgb(labels_2d, target_np, kind='avg') # Replaces every pixel with its segment's average colour
pydiffvg.imwrite(
    torch.from_numpy(seg_viz.astype(np.float32)),
    'results/shepard_segmented/segments.png',
    gamma=1.0)
print('Saved segments.png')

# Save each segment individually
os.makedirs('results/shepard_segmented/segments', exist_ok=True)
for seg_id in range(n_segments):
    mask = (labels_2d == seg_id)
    seg_img = np.zeros_like(target_np)
    seg_img[mask] = target_np[mask]  # show the original colors only
    pydiffvg.imwrite(
        torch.from_numpy(seg_img.astype(np.float32)),
        f'results/shepard_segmented/segments/seg_{seg_id}.png',
        gamma=1.0)
print('Saved individual segments')

# --- Run Shepard optimization per segment ---
print('Running per-segment Shepard optimization...')
final_image = np.zeros_like(target_np) # Starts off as empty, but the segments will be written on it
target_tensor = torch.from_numpy(target_np)

all_positions = []  # collect final control point positions from each segment
segment_losses = []

for seg_id in range(n_segments):
    mask = (labels_2d == seg_id) # A boolean grid where its true for only pixels in the segment
    
    # Edge case if a boundary has no pixels
    if mask.sum() == 0:
        continue

    # Bounding box of this segment
    ys, xs = np.where(mask)
    x_min, x_max = xs.min() / canvas_width,  xs.max() / canvas_width
    y_min, y_max = ys.min() / canvas_height, ys.max() / canvas_height

    # Initialize control points randomly within the segments
    positions_n = torch.zeros(N_per_segment, 2)
    positions_n[:, 0] = torch.rand(N_per_segment) * (x_max - x_min) + x_min
    positions_n[:, 1] = torch.rand(N_per_segment) * (y_max - y_min) + y_min
    positions_n = positions_n.clone().requires_grad_(True)

    # Initialize colors from segment average + small noise
    # seg_color = target_np[mask].mean(axis=0)
    # colors = torch.tensor(seg_color).unsqueeze(0).repeat(N_per_segment, 1)
    # colors = (colors + torch.rand_like(colors) * 0.1).clamp(0, 1).clone().requires_grad_(True)

    colors = torch.rand(N_per_segment, 3).clamp(0, 1).clone().requires_grad_(True)

    optimizer = torch.optim.Adam([positions_n, colors], lr=1e-2)
    mask_tensor = torch.from_numpy(mask).unsqueeze(-1).expand(-1, -1, 3) # True is 1.0, and false is 0.0

    for t in range(iters):
        optimizer.zero_grad()
        positions = positions_n * torch.tensor([canvas_width, canvas_height])
        img = pydiffvg.ShepardRenderFunction.apply(
            positions, colors, q, canvas_width, canvas_height)

        # Loss only on pixels belonging to this segment
        loss = ((img - target_tensor) * mask_tensor).pow(2).sum()
        loss.backward()
        optimizer.step()

        if t % 10 == 0: # Save every 10 iterations
            with torch.no_grad():
                debug_img = pydiffvg.ShepardRenderFunction.apply(
                    positions_n * torch.tensor([canvas_width, canvas_height]),
                    colors, q, canvas_width, canvas_height)
                debug_frame = final_image.copy()
                debug_frame[mask] = debug_img.clamp(0, 1).numpy()[mask]
                frame = torch.from_numpy(debug_frame.astype(np.float32))
                os.makedirs(f'results/shepard_segmented/seg_{seg_id}', exist_ok=True)
                pydiffvg.imwrite(frame, f'results/shepard_segmented/seg_{seg_id}/iter_{t//10 + 1}.png', gamma=1.0)

        with torch.no_grad():
            positions_n.clamp_(0.0, 1.0)
            colors.clamp_(0.0, 1.0)

    print(f'segment {seg_id+1}/{n_segments} done, loss: {loss.item():.2f}')
    segment_losses.append(loss.item())

    # Write this segment's pixels into the final image
    with torch.no_grad():
        positions = positions_n * torch.tensor([canvas_width, canvas_height])
        img = pydiffvg.ShepardRenderFunction.apply(
            positions, colors, q, canvas_width, canvas_height)
        final_image[mask] = img.clamp(0, 1).numpy()[mask]

        all_positions.append(positions_n.detach() * torch.tensor([canvas_width, canvas_height]))

        # Save frame showing reconstruction progress after each segment
        frame = torch.from_numpy(final_image.astype(np.float32))
        pydiffvg.imwrite(frame, f'results/shepard_segmented/iter_{seg_id}.png', gamma=1.0)

# --- Save results ---
final_tensor = torch.from_numpy(final_image.astype(np.float32))
pydiffvg.imwrite(final_tensor, 'results/shepard_segmented/final.png', gamma=1.0)
print('saved final.png')

# -----------------------------
# Plot final loss per segment
# -----------------------------
fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(range(len(segment_losses)), segment_losses)
ax.set_xlabel('Segment ID')
ax.set_ylabel('Final loss')
ax.set_title(f'Final loss per segment (N={N_per_segment} pts, {iters} iters)')
plt.savefig('results/shepard_segmented/loss_per_segment.png', bbox_inches='tight', dpi=150)
plt.close(fig)
print('saved loss_per_segment.png')

# -----------------------------------------------------------------------
# Visualization: overlay all control point locations on the final render
# -----------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(final_image)

for pos in all_positions:
    pos_np = pos.cpu().numpy()
    ax.scatter(pos_np[:, 0], pos_np[:, 1], c='red', s=10, edgecolors='white', linewidths=0.5)

ax.set_xlim(0, canvas_width)
ax.set_ylim(canvas_height, 0)
fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
ax.axis('off')
plt.savefig('results/shepard_segmented/final_labeled.png', bbox_inches='tight', pad_inches=0, dpi=150)
plt.close(fig)
print('saved final_labeled.png')

# -------------------------
# Per-pixel error heatmap
# -------------------------
fig, ax = plt.subplots(figsize=(8, 6))

# Per-pixel error: mean squared difference across RGB channels
error_map = ((target_np - final_image) ** 2).mean(axis=2)

im = ax.imshow(error_map, cmap='inferno')
ax.axis('off')
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
plt.savefig('results/shepard_segmented/error_heatmap.png', bbox_inches='tight', dpi=150)
plt.close(fig)
print('saved error_heatmap.png')

# ---------------------------------------------------------------
# Comparison: target | segmentation | rendered | error heatmap
# ---------------------------------------------------------------

fig, axes = plt.subplots(1, 4, figsize=(24, 6))

axes[0].imshow(target_np)
axes[0].set_title('Target')
axes[0].axis('off')

axes[1].imshow(seg_viz)
axes[1].set_title(f'Segmentation ({n_segments} regions)')
axes[1].axis('off')

axes[2].imshow(final_image)
axes[2].set_title(f'Segmented Shepard ({N_per_segment} pts/segment)')
axes[2].axis('off')

im = axes[3].imshow(error_map, cmap='inferno')
axes[3].set_title('Error heatmap')
axes[3].axis('off')
fig.colorbar(im, ax=axes[3], fraction=0.046, pad=0.04)

plt.savefig('results/shepard_segmented/all_comparison.png', bbox_inches='tight', dpi=150)
plt.close(fig)
print('saved all_comparison.png')

from subprocess import call
call(["ffmpeg", "-framerate", "3", "-i",
    "results/shepard_segmented/iter_%d.png", "-vb", "20M",
    "results/shepard_segmented/out.mp4"])