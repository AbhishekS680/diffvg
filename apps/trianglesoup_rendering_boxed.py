# trianglesoup_rendering_boxed.py
#
# Fits N independent triangles (no shared vertices/edges, unlike a mesh)
# directly to a single target image, using the tile-grid accelerated
# renderer. Each triangle has a flat colour and a learnable opacity,
# composited via alpha-over in index order. Core primitive script --
# part of the four-way comparison (Wendland / Gaussian / Shepard /
# Triangle Soup) used in the report. Uses plain uniform-random
# initialization, matching comparison_trianglesoup_boxed.py -- see
# trianglesoup_rendering_boxed_edge_weighted_init.py for the separate
# edge-weighted-initialization ablation.
#
# Usage:
#   python trianglesoup_rendering_boxed.py --image imgs/cat.png --n 1000 --iters 200
#
# Args:
#   --image   target image path
#   --n       number of triangles
#   --iters   number of training iterations
#   --seed    random seed, for reproducibility across primitives
#
# After the main training loop, runs an optional focus phase: a second
# round of iterations on the same triangles, reweighting the loss by
# pass one's error heatmap so poorly-reconstructed pixels get more
# gradient pull.
import argparse
import os
import torch
import numpy as np
import skimage.io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
import pydiffvg
import diffvg

parser = argparse.ArgumentParser()
parser.add_argument('--image', default='imgs/fruit_basket.png', help='Target image path')
parser.add_argument('--n', type=int, default=1000, help='Number of triangles')
parser.add_argument('--iters', type=int, default=200, help='Number of training iterations')
parser.add_argument('--seed', type=int, default=0, help='Random seed')
args = parser.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)

N = args.n
iters = args.iters

# --- Softness annealing ---
# Coverage uses a sigmoid-smoothed edge test with width SOFTNESS pixels.
# Wide softness early gives strong gradients so triangles can move large
# distances quickly; narrow softness late gives sharp edges in the final
# render. Annealed linearly across the run since softness is a plain
# float argument here, not a learnable tensor.
SOFTNESS_START = 4.0
SOFTNESS_END   = 0.5

FOCUS_ITERS = 200
FOCUS_WEIGHT_SCALE = 6.0  # how much extra weight the worst pixels get, relative to the best

OUTDIR = 'results/trianglesoup_rendering_boxed'
os.makedirs(OUTDIR, exist_ok=True)
os.makedirs(f'{OUTDIR}/focus_iters', exist_ok=True)

pydiffvg.set_use_gpu(torch.cuda.is_available())

target = torch.from_numpy(skimage.io.imread(args.image)).to(torch.float32) / 255.0
target = target[:, :, :3]
canvas_height, canvas_width = target.shape[0], target.shape[1]
pydiffvg.imwrite(target.cpu(), f'{OUTDIR}/target.png', gamma=1.0)

# --- Initialize N triangles (vertices, colour, opacity) uniformly at random ---
vertices_n = torch.rand(N, 3, 2).clone().requires_grad_(True)  # normalized [0,1]
colours    = torch.rand(N, 3).clone().requires_grad_(True)
# Opacity logit: unconstrained real parameter, squashed through sigmoid
# before being used. Init at 0 -> sigmoid(0) = 0.5, a neutral starting
# opacity -- either extreme gives weak initial gradients.
opacity_logit = torch.zeros(N).clone().requires_grad_(True)
optimizer = torch.optim.Adam([vertices_n, colours, opacity_logit], lr=1e-2)
loss_history = []
diffvg.reset_trianglesoup_boxed_timing()

open(f'{OUTDIR}/softness_log.txt', 'w').close()
open(f'{OUTDIR}/opacity_log.txt', 'w').close()

# --------------------------------------
# Main training loop.
# --------------------------------------
for t in range(iters):
    optimizer.zero_grad()
    softness = SOFTNESS_START + (SOFTNESS_END - SOFTNESS_START) * (t / max(iters - 1, 1))
    vertices_px = vertices_n * torch.tensor([canvas_width, canvas_height])
    opacity = torch.sigmoid(opacity_logit)
    img = pydiffvg.TriangleSoupBoxedRenderFunction.apply(
        vertices_px, colours, opacity, softness, None, canvas_width, canvas_height)
    loss = (img - target).pow(2).sum()
    loss_history.append(loss.item())
    loss.backward()

    print('iter', t, 'loss', loss.item(), 'softness', round(softness, 3))
    with open(f'{OUTDIR}/softness_log.txt', 'a') as f:
        f.write(f'iter {t}: softness={softness:.4f} loss={loss.item():.4f}\n')
    with torch.no_grad():
        opacity_current = torch.sigmoid(opacity_logit)
    with open(f'{OUTDIR}/opacity_log.txt', 'a') as f:
        f.write(f'iter {t}: opacity[min={opacity_current.min().item():.3f} '
                f'max={opacity_current.max().item():.3f} '
                f'mean={opacity_current.mean().item():.3f}]\n')

    optimizer.step()

    if t == iters - 2:
        second_last_vertices_px = (vertices_n.detach() * torch.tensor([canvas_width, canvas_height])).clone().numpy()

    with torch.no_grad():
        vertices_n.clamp_(0.0, 1.0)
        colours.clamp_(0.0, 1.0)
        # No clamp needed for opacity_logit -- sigmoid already bounds the
        # actual opacity to [0,1] regardless of the logit's raw value.

    pydiffvg.imwrite(img.detach().clamp(0, 1).cpu(), f'{OUTDIR}/iter_{t}.png', gamma=1.0)

print(f'final loss: {loss.item():.4f}')
with open(f'{OUTDIR}/final_loss.txt', 'w') as f:
    f.write(str(loss.item()))

diffvg.print_trianglesoup_boxed_timing()
fwd_ms, fwd_calls, bwd_ms, bwd_calls = diffvg.get_trianglesoup_boxed_timing()
with open(f'{OUTDIR}/timing.txt', 'w') as f:
    f.write(f"render_trianglesoup_boxed timing (N={N}, {iters} iters, {canvas_width}x{canvas_height})\n")
    f.write(f"Forward:  {fwd_ms:.3f} ms total, {fwd_calls} pixel-calls, {fwd_ms/fwd_calls:.6f} ms/pixel\n")
    f.write(f"Backward: {bwd_ms:.3f} ms total, {bwd_calls} pixel-calls, {bwd_ms/bwd_calls:.6f} ms/pixel\n")

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(loss_history)
ax.set_xlabel('Iteration')
ax.set_ylabel('Loss')
ax.set_title(f'Convergence (N={N} triangles, boxed/tile-grid, learnable opacity, softness {SOFTNESS_START}->{SOFTNESS_END}px)')
plt.savefig(f'{OUTDIR}/loss_curve.png', bbox_inches='tight', dpi=150)
plt.close(fig)
print('saved loss_curve.png')

# --- Final render (pass one) ---
vertices_px = vertices_n * torch.tensor([canvas_width, canvas_height])
opacity_final = torch.sigmoid(opacity_logit)
final = pydiffvg.TriangleSoupBoxedRenderFunction.apply(
    vertices_px, colours, opacity_final, SOFTNESS_END, None, canvas_width, canvas_height)
pydiffvg.imwrite(final.detach().clamp(0, 1).cpu(), f'{OUTDIR}/final.png', gamma=1.0)

# Overlay triangle outlines on final render, edge opacity reflecting
# each triangle's learned opacity
fig, ax = plt.subplots(figsize=(8, 8))
display_img = final.detach().clamp(0, 1).cpu().numpy()
ax.imshow(display_img)
verts_np = vertices_px.detach().cpu().numpy()
opacity_np = opacity_final.detach().cpu().numpy()
for idx in range(N):
    tri = verts_np[idx]
    ax.add_patch(MplPolygon(tri, closed=True, facecolor='none',
                             edgecolor='lime', linewidth=0.4,
                             alpha=float(np.clip(opacity_np[idx], 0.05, 1.0))))
ax.set_xlim(0, canvas_width)
ax.set_ylim(canvas_height, 0)
fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
ax.axis('off')
plt.savefig(f'{OUTDIR}/final_labeled.png', bbox_inches='tight', pad_inches=0, dpi=150)
plt.close(fig)
print('saved final_labeled.png')

final_np = final.detach().clamp(0, 1).cpu().numpy()

# Quiver plot: direction each vertex moved, second-to-last -> final
final_vertices_px = vertices_px.detach().numpy().reshape(-1, 2)
second_last_flat = second_last_vertices_px.reshape(-1, 2)
u = final_vertices_px[:, 0] - second_last_flat[:, 0]
v = final_vertices_px[:, 1] - second_last_flat[:, 1]
fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(final_np, alpha=0.6)
ax.quiver(second_last_flat[:, 0], second_last_flat[:, 1], u, v,
          angles='xy', scale_units='xy', scale=1, color='red', width=0.002)
ax.scatter(second_last_flat[:, 0], second_last_flat[:, 1], c='cyan', s=4, label='second-to-last')
ax.scatter(final_vertices_px[:, 0], final_vertices_px[:, 1], c='red', s=4, label='final')
ax.set_xlim(0, canvas_width)
ax.set_ylim(canvas_height, 0)
ax.legend(loc='upper right')
ax.axis('off')
plt.savefig(f'{OUTDIR}/movement_quiver.png', bbox_inches='tight', dpi=150)
plt.close(fig)
print('saved movement_quiver.png')

# Opacity distribution histogram
fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(opacity_np, bins=40, color='#3C896D')
ax.set_xlabel('Learned opacity')
ax.set_ylabel('Count')
ax.set_title('Final opacity distribution across triangles')
plt.savefig(f'{OUTDIR}/opacity_histogram.png', bbox_inches='tight', dpi=150)
plt.close(fig)
print('saved opacity_histogram.png')

# Per-pixel error heatmap (pass one, pre-focus)
fig, ax = plt.subplots(figsize=(8, 6))
target_np = target.cpu().numpy()
error_map = ((target_np - final_np) ** 2).mean(axis=2)
im = ax.imshow(error_map, cmap='inferno')
ax.axis('off')
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
plt.savefig(f'{OUTDIR}/error_heatmap.png', bbox_inches='tight', dpi=150)
plt.close(fig)
print('saved error_heatmap.png')

# --------------------------------------
# Focus phase.
# --------------------------------------
# Baseline weight of 1.0 ensures every pixel keeps some supervision --
# without it, already-good pixels get near-zero weight and nothing
# protects them from drifting while the optimizer chases the worst
# error, which actively degrades the reconstruction.
weight_np = 1.0 + FOCUS_WEIGHT_SCALE * (error_map / (error_map.max() + 1e-8))
weight_map = torch.from_numpy(weight_np).to(torch.float32).unsqueeze(-1)  # (H, W, 1), broadcasts over RGB

fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(weight_np, cmap='viridis')
ax.axis('off')
ax.set_title(f'Focus weight map (scale={FOCUS_WEIGHT_SCALE}, max weight={weight_np.max():.2f})')
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
plt.savefig(f'{OUTDIR}/focus_weight_map.png', bbox_inches='tight', dpi=150)
plt.close(fig)
print('saved focus_weight_map.png')

focus_optimizer = torch.optim.Adam([vertices_n, colours, opacity_logit], lr=5e-3)
focus_loss_history = []
for t in range(FOCUS_ITERS):
    focus_optimizer.zero_grad()
    vertices_px = vertices_n * torch.tensor([canvas_width, canvas_height])
    opacity = torch.sigmoid(opacity_logit)
    img = pydiffvg.TriangleSoupBoxedRenderFunction.apply(
        vertices_px, colours, opacity, SOFTNESS_END, None, canvas_width, canvas_height)
    focus_loss = ((img - target).pow(2) * weight_map).sum()
    focus_loss_history.append(focus_loss.item())
    focus_loss.backward()
    print('focus iter', t, 'loss', focus_loss.item())
    focus_optimizer.step()
    with torch.no_grad():
        vertices_n.clamp_(0.0, 1.0)
        colours.clamp_(0.0, 1.0)
    pydiffvg.imwrite(img.detach().clamp(0, 1).cpu(), f'{OUTDIR}/focus_iters/iter_{t}.png', gamma=1.0)

print(f'final focus loss: {focus_loss.item():.4f}')
with open(f'{OUTDIR}/focus_final_loss.txt', 'w') as f:
    f.write(str(focus_loss.item()))

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(focus_loss_history)
ax.set_xlabel('Focus iteration')
ax.set_ylabel('Weighted loss')
ax.set_title(f'Focus phase convergence (weight scale={FOCUS_WEIGHT_SCALE}, {FOCUS_ITERS} iters)')
plt.savefig(f'{OUTDIR}/focus_loss_curve.png', bbox_inches='tight', dpi=150)
plt.close(fig)
print('saved focus_loss_curve.png')

# --- Final render (post-focus) ---
vertices_px = vertices_n * torch.tensor([canvas_width, canvas_height])
opacity_final = torch.sigmoid(opacity_logit)
final_focused = pydiffvg.TriangleSoupBoxedRenderFunction.apply(
    vertices_px, colours, opacity_final, SOFTNESS_END, None, canvas_width, canvas_height)
final_focused = final_focused.clamp(0, 1)
pydiffvg.imwrite(final_focused.detach().cpu(), f'{OUTDIR}/final_focused.png', gamma=1.0)
final_focused_np = final_focused.detach().clamp(0, 1).cpu().numpy()

# Error heatmap against target, post-focus -- same colour scale as the
# pre-focus error heatmap so the two are directly comparable.
error_map_focused = ((target_np - final_focused_np) ** 2).mean(axis=2)
shared_focus_vmax = max(error_map.max(), error_map_focused.max())
fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(error_map_focused, cmap='inferno', vmin=0, vmax=shared_focus_vmax)
ax.axis('off')
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
plt.savefig(f'{OUTDIR}/error_heatmap_focused.png', bbox_inches='tight', dpi=150)
plt.close(fig)
print('saved error_heatmap_focused.png')

print(f'pre-focus mean error:  {error_map.mean():.6f}')
print(f'post-focus mean error: {error_map_focused.mean():.6f}')
with open(f'{OUTDIR}/focus_summary.txt', 'w') as f:
    f.write(f'pre-focus mean error:  {error_map.mean():.6f}\n')
    f.write(f'post-focus mean error: {error_map_focused.mean():.6f}\n')
    f.write(f'change: {error_map_focused.mean() - error_map.mean():.6f}\n')

# Before/after focus comparison
fig, axes = plt.subplots(1, 4, figsize=(24, 6))
axes[0].imshow(final_np)
axes[0].set_title('Reconstruction (pre-focus)')
axes[0].axis('off')
im0 = axes[1].imshow(error_map, cmap='inferno', vmin=0, vmax=shared_focus_vmax)
axes[1].set_title(f'Error (pre-focus), mean={error_map.mean():.5f}')
axes[1].axis('off')
fig.colorbar(im0, ax=axes[1], fraction=0.046, pad=0.04)
axes[2].imshow(final_focused_np)
axes[2].set_title(f'Reconstruction (post-focus, {FOCUS_ITERS} iters)')
axes[2].axis('off')
im1 = axes[3].imshow(error_map_focused, cmap='inferno', vmin=0, vmax=shared_focus_vmax)
axes[3].set_title(f'Error (post-focus), mean={error_map_focused.mean():.5f}')
axes[3].axis('off')
fig.colorbar(im1, ax=axes[3], fraction=0.046, pad=0.04)
plt.savefig(f'{OUTDIR}/focus_comparison.png', bbox_inches='tight', dpi=150)
plt.close(fig)
print('saved focus_comparison.png')

# --- Comparison: target | rendered | error heatmap (pre-focus) ---
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
axes[0].imshow(target_np)
axes[0].set_title('Target')
axes[0].axis('off')
axes[1].imshow(final_np)
axes[1].set_title('Rendered (Triangle soup, boxed, learnable opacity)')
axes[1].axis('off')
im = axes[2].imshow(error_map, cmap='inferno')
axes[2].set_title('Error heatmap')
axes[2].axis('off')
fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
plt.savefig(f'{OUTDIR}/all_comparison.png', bbox_inches='tight', dpi=150)
plt.close(fig)
print('saved all_comparison.png')

from subprocess import call
call(["ffmpeg", "-framerate", "24", "-i",
    f"{OUTDIR}/iter_%d.png", "-vb", "20M",
    f"{OUTDIR}/out.mp4"])