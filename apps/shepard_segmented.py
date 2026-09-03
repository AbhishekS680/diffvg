# shepard_segmented.py
#
# Experimental variant of Shepard IDW reconstruction. The target image
# is first split into regions via Mean Shift colour segmentation, then
# a separate small Shepard IDW field is optimized independently within
# each region, with control points initialized inside that region's
# bounding box rather than across the whole canvas.
#
# NOT part of the core four-primitive comparison (Wendland / Gaussian /
# Shepard / Triangle Soup) used in the report -- this is a standalone
# exploration of whether segmentation improves Shepard's control-point
# allocation, using its own non-standard initialization by design.
#
# Usage:
#   python shepard_segmented.py --image imgs/cat.png --n 100 --iters 100 --seg-size 0.2
#
# Args:
#   --image     target image path
#   --n         control points per segment
#   --iters     training iterations per segment
#   --seg-size  Mean Shift bandwidth (larger = fewer, bigger segments)
#   --seed      random seed, for reproducibility
import argparse
import os
import time
import numpy as np
import torch
import skimage.io
import skimage.color
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.cluster import MeanShift
import pydiffvg
import diffvg

parser = argparse.ArgumentParser()
parser.add_argument('--image', default='imgs/fruit_basket.png', help='Target image path')
parser.add_argument('--n', type=int, default=100, help='Control points per segment')
parser.add_argument('--iters', type=int, default=100, help='Training iterations per segment')
parser.add_argument('--seg-size', type=float, default=0.2,
                     help='Mean Shift bandwidth -- larger value means fewer but bigger segments')
parser.add_argument('--seed', type=int, default=0, help='Random seed')
args = parser.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)

N_per_segment = args.n
q = 3.0  # Shepard falloff sharpness, matches the other Shepard scripts
iters = args.iters
seg_size = args.seg_size

OUTDIR = 'results/shepard_segmented'
os.makedirs(OUTDIR, exist_ok=True)

# --- Load target image ---
target_np = skimage.io.imread(args.image).astype(np.float32) / 255.0
target_np = target_np[:, :, :3]
canvas_height, canvas_width = target_np.shape[0], target_np.shape[1]
pydiffvg.imwrite(torch.from_numpy(target_np), f'{OUTDIR}/target.png', gamma=1.0)

# --- Mean Shift segmentation ---
# Each pixel is represented as (x, y, r, g, b) so segments respect both
# spatial proximity and colour similarity.
print('Running mean shift segmentation...')
h, w = target_np.shape[:2]
coords = np.column_stack([
    np.indices((h, w))[1].ravel() / w,
    np.indices((h, w))[0].ravel() / h,
    target_np.reshape(-1, 3)
])
start = time.time()
ms = MeanShift(bandwidth=seg_size, bin_seeding=True)
ms.fit(coords)
labels = ms.labels_
n_segments = len(np.unique(labels))
print(f'Segmentation done in {time.time()-start:.1f}s, {n_segments} segments found')

labels_2d = labels.reshape(h, w)
seg_viz = skimage.color.label2rgb(labels_2d, target_np, kind='avg')
pydiffvg.imwrite(torch.from_numpy(seg_viz.astype(np.float32)), f'{OUTDIR}/segments.png', gamma=1.0)

os.makedirs(f'{OUTDIR}/segments', exist_ok=True)
for seg_id in range(n_segments):
    mask = (labels_2d == seg_id)
    seg_img = np.zeros_like(target_np)
    seg_img[mask] = target_np[mask]
    pydiffvg.imwrite(torch.from_numpy(seg_img.astype(np.float32)),
                      f'{OUTDIR}/segments/seg_{seg_id}.png', gamma=1.0)

# --- Per-segment Shepard optimization ---
# Each segment gets its own independent set of control points,
# confined to that segment's bounding box and trained only against the
# pixels inside it (via mask_tensor in the loss below).
print('Running per-segment Shepard optimization...')
diffvg.reset_shepard_timing()
final_image = np.zeros_like(target_np)
target_tensor = torch.from_numpy(target_np)
all_positions = []
segment_losses = []

for seg_id in range(n_segments):
    mask = (labels_2d == seg_id)
    if mask.sum() == 0:
        continue

    ys, xs = np.where(mask)
    x_min, x_max = xs.min() / canvas_width,  xs.max() / canvas_width
    y_min, y_max = ys.min() / canvas_height, ys.max() / canvas_height

    positions_n = torch.zeros(N_per_segment, 2)
    positions_n[:, 0] = torch.rand(N_per_segment) * (x_max - x_min) + x_min
    positions_n[:, 1] = torch.rand(N_per_segment) * (y_max - y_min) + y_min
    positions_n = positions_n.clone().requires_grad_(True)
    colors = torch.rand(N_per_segment, 3).clamp(0, 1).clone().requires_grad_(True)
    optimizer = torch.optim.Adam([positions_n, colors], lr=1e-2)

    mask_tensor = torch.from_numpy(mask).unsqueeze(-1).expand(-1, -1, 3)

    for t in range(iters):
        optimizer.zero_grad()
        positions = positions_n * torch.tensor([canvas_width, canvas_height])
        img = pydiffvg.ShepardRenderFunction.apply(positions, colors, q, canvas_width, canvas_height)
        loss = ((img - target_tensor) * mask_tensor).pow(2).sum()  # loss only on this segment's pixels
        loss.backward()
        optimizer.step()

        if t % 10 == 0:
            with torch.no_grad():
                debug_img = pydiffvg.ShepardRenderFunction.apply(
                    positions_n * torch.tensor([canvas_width, canvas_height]),
                    colors, q, canvas_width, canvas_height)
                debug_frame = final_image.copy()
                debug_frame[mask] = debug_img.clamp(0, 1).numpy()[mask]
                os.makedirs(f'{OUTDIR}/seg_{seg_id}', exist_ok=True)
                pydiffvg.imwrite(torch.from_numpy(debug_frame.astype(np.float32)),
                                  f'{OUTDIR}/seg_{seg_id}/iter_{t//10 + 1}.png', gamma=1.0)

        with torch.no_grad():
            positions_n.clamp_(0.0, 1.0)
            colors.clamp_(0.0, 1.0)

    print(f'segment {seg_id+1}/{n_segments} done, loss: {loss.item():.2f}')
    segment_losses.append(loss.item())

    with torch.no_grad():
        positions = positions_n * torch.tensor([canvas_width, canvas_height])
        img = pydiffvg.ShepardRenderFunction.apply(positions, colors, q, canvas_width, canvas_height)
        final_image[mask] = img.clamp(0, 1).numpy()[mask]
        all_positions.append(positions_n.detach() * torch.tensor([canvas_width, canvas_height]))
        pydiffvg.imwrite(torch.from_numpy(final_image.astype(np.float32)),
                          f'{OUTDIR}/iter_{seg_id}.png', gamma=1.0)

# --- Save results ---
final_tensor = torch.from_numpy(final_image.astype(np.float32))
pydiffvg.imwrite(final_tensor, f'{OUTDIR}/final.png', gamma=1.0)
print('saved final.png')

diffvg.print_shepard_timing()
fwd_ms, fwd_calls, bwd_ms, bwd_calls = diffvg.get_shepard_timing()
with open(f'{OUTDIR}/timing.txt', 'w') as f:
    f.write(f"render_shepard timing (N_per_segment={N_per_segment}, {n_segments} segments, "
            f"{iters} iters/segment, {canvas_width}x{canvas_height})\n")
    f.write(f"Forward:  {fwd_ms:.3f} ms total, {fwd_calls} pixel-calls, {fwd_ms/fwd_calls:.6f} ms/pixel\n")
    f.write(f"Backward: {bwd_ms:.3f} ms total, {bwd_calls} pixel-calls, {bwd_ms/bwd_calls:.6f} ms/pixel\n")

fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(range(len(segment_losses)), segment_losses)
ax.set_xlabel('Segment ID')
ax.set_ylabel('Final loss')
ax.set_title(f'Final loss per segment (N={N_per_segment} pts, {iters} iters)')
plt.savefig(f'{OUTDIR}/loss_per_segment.png', bbox_inches='tight', dpi=150)
plt.close(fig)
print('saved loss_per_segment.png')

# Overlay every segment's control points on the final render
fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(final_image)
for pos in all_positions:
    pos_np = pos.cpu().numpy()
    ax.scatter(pos_np[:, 0], pos_np[:, 1], c='red', s=10, edgecolors='white', linewidths=0.5)
ax.set_xlim(0, canvas_width)
ax.set_ylim(canvas_height, 0)
fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
ax.axis('off')
plt.savefig(f'{OUTDIR}/final_labeled.png', bbox_inches='tight', pad_inches=0, dpi=150)
plt.close(fig)
print('saved final_labeled.png')

fig, ax = plt.subplots(figsize=(8, 6))
error_map = ((target_np - final_image) ** 2).mean(axis=2)
im = ax.imshow(error_map, cmap='inferno')
ax.axis('off')
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
plt.savefig(f'{OUTDIR}/error_heatmap.png', bbox_inches='tight', dpi=150)
plt.close(fig)
print('saved error_heatmap.png')

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
plt.savefig(f'{OUTDIR}/all_comparison.png', bbox_inches='tight', dpi=150)
plt.close(fig)
print('saved all_comparison.png')

from subprocess import call
call(["ffmpeg", "-framerate", "3", "-i",
    f"{OUTDIR}/iter_%d.png", "-vb", "20M",
    f"{OUTDIR}/out.mp4"])