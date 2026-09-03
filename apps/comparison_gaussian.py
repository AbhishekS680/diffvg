# comparison_gaussian.py
#
# Direct reconstruction: canvas starts as the blurry (degraded) image,
# ellipses composite directly on top of it. Plain O(N) renderer -- see
# comparison_gaussian_boxed.py for the tile-grid accelerated version.
# Core primitive script -- part of the four-way comparison (Wendland /
# Gaussian / Shepard / Triangle Soup) used in the report.
#
# Usage:
#   python comparison_gaussian.py --target imgs/level_0.png --degraded imgs/level_1.png \
#       --outdir results/comparison_gaussian --n 1000 --iters 200
#
# Args:
#   --target    sharp target image path
#   --degraded  degraded/blurred image, used as the starting canvas
#   --outdir    output directory
#   --n         number of ellipses
#   --iters     number of training iterations
#   --seed      random seed, for reproducibility across primitives
import argparse
import os
import torch
import numpy as np
import skimage.io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import pydiffvg
import diffvg

parser = argparse.ArgumentParser()
parser.add_argument('--target', required=True, help='Sharp target image path')
parser.add_argument('--degraded', required=True, help='Degraded/blurred image, used as starting canvas')
parser.add_argument('--outdir', default='results/comparison_gaussian')
parser.add_argument('--n', type=int, default=1000, help='Number of ellipses')
parser.add_argument('--iters', type=int, default=200, help='Number of training iterations')
parser.add_argument('--seed', type=int, default=0, help='Random seed')
args = parser.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)

os.makedirs(args.outdir, exist_ok=True)
os.makedirs(f'{args.outdir}/iters', exist_ok=True)

N = args.n
iters = args.iters

pydiffvg.set_use_gpu(torch.cuda.is_available())

# --- Load images ---
original = torch.from_numpy(skimage.io.imread(args.target)).to(torch.float32) / 255.0
original = original[:, :, :3]
canvas_height, canvas_width = original.shape[0], original.shape[1]
pydiffvg.imwrite(original.cpu(), f'{args.outdir}/target_original.png', gamma=1.0)

degraded_np = skimage.io.imread(args.degraded).astype(np.float32) / 255.0
degraded_np = degraded_np[:, :, :3]
degraded = torch.from_numpy(degraded_np)
pydiffvg.imwrite(degraded.cpu(), f'{args.outdir}/init_source_degraded.png', gamma=1.0)

assert degraded_np.shape[0] == canvas_height and degraded_np.shape[1] == canvas_width, \
    'Degraded and original images must be the same size'

# --- Baseline error: degraded vs original, before any reconstruction ---
original_np = original.cpu().numpy()
degraded_error_map = ((original_np - degraded_np) ** 2).mean(axis=2)
print(f'baseline (degraded) mean error: {degraded_error_map.mean():.6f}')
with open(f'{args.outdir}/baseline_error.txt', 'w') as f:
    f.write(str(degraded_error_map.mean()))

fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(degraded_error_map, cmap='inferno')
ax.axis('off')
ax.set_title(f'Baseline error (degraded vs original), mean={degraded_error_map.mean():.5f}')
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
plt.savefig(f'{args.outdir}/degraded_error_heatmap.png', bbox_inches='tight', dpi=150)
plt.close(fig)
print('saved degraded_error_heatmap.png')

# --- Initialize N ellipses, matching wendland_rendering.py's convention ---
positions_n = torch.rand(N, 2).clone().requires_grad_(True)
colors = torch.rand(N, 3).clone().requires_grad_(True)
log_a = torch.full((N,), torch.log(torch.tensor(0.05))).clone().requires_grad_(True)
log_b = torch.full((N,), torch.log(torch.tensor(0.05))).clone().requires_grad_(True)
theta = torch.zeros(N).clone().requires_grad_(True)
optimizer = torch.optim.Adam([positions_n, colors, log_a, log_b, theta], lr=1e-2)
loss_history = []
diffvg.reset_ellipse_gaussian_timing()

# --------------------------------------
# Optimization loop: ellipses composited on top of the degraded canvas.
# --------------------------------------
for t in range(iters):
    optimizer.zero_grad()
    positions_px = positions_n * torch.tensor([canvas_width, canvas_height])
    a_px = torch.exp(log_a) * canvas_width
    b_px = torch.exp(log_b) * canvas_width
    img = pydiffvg.EllipseGaussianRenderFunction.apply(
        positions_px, colors, a_px, b_px, theta, degraded, canvas_width, canvas_height)
    loss = (img - original).pow(2).sum()
    loss_history.append(loss.item())
    loss.backward()
    print('iter', t, 'loss', loss.item())

    optimizer.step()

    if t == iters - 2:
        second_last_positions_px = (positions_n.detach() * torch.tensor([canvas_width, canvas_height])).clone().numpy()

    with torch.no_grad():
        positions_n.clamp_(0.0, 1.0)
        colors.clamp_(0.0, 1.0)
        log_a.clamp_(torch.log(torch.tensor(0.01)), torch.log(torch.tensor(1.0)))
        log_b.clamp_(torch.log(torch.tensor(0.01)), torch.log(torch.tensor(1.0)))

    pydiffvg.imwrite(img.detach().clamp(0, 1).cpu(), f'{args.outdir}/iters/iter_{t}.png', gamma=1.0)

print(f'final loss: {loss.item():.4f}')
diffvg.print_ellipse_gaussian_timing()
fwd_ms, fwd_calls, bwd_ms, bwd_calls = diffvg.get_ellipse_gaussian_timing()
with open(f'{args.outdir}/timing.txt', 'w') as f:
    f.write(f"render_ellipse_gaussian timing (N={N}, {iters} iters, {canvas_width}x{canvas_height})\n")
    f.write(f"Forward:  {fwd_ms:.3f} ms total, {fwd_calls} pixel-calls, {fwd_ms/fwd_calls:.6f} ms/pixel\n")
    f.write(f"Backward: {bwd_ms:.3f} ms total, {bwd_calls} pixel-calls, {bwd_ms/bwd_calls:.6f} ms/pixel\n")
with open(f'{args.outdir}/final_loss.txt', 'w') as f:
    f.write(str(loss.item()))

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(loss_history)
ax.set_xlabel('Iteration')
ax.set_ylabel('Loss')
ax.set_title(f'Convergence (N={N}, ellipses composited on blurry image)')
plt.savefig(f'{args.outdir}/loss_curve.png', bbox_inches='tight', dpi=150)
plt.close(fig)
print('saved loss_curve.png')

# --- Final render ---
positions_px = positions_n * torch.tensor([canvas_width, canvas_height])
a_px = torch.exp(log_a) * canvas_width
b_px = torch.exp(log_b) * canvas_width
final = pydiffvg.EllipseGaussianRenderFunction.apply(
    positions_px, colors, a_px, b_px, theta, degraded, canvas_width, canvas_height)
final = final.clamp(0, 1)
pydiffvg.imwrite(final.detach().cpu(), f'{args.outdir}/final.png', gamma=1.0)

# Overlay control points on final render
fig, ax = plt.subplots(figsize=(8, 8))
display_img = final.detach().clamp(0, 1).cpu().numpy()
ax.imshow(display_img)
pos_np = (positions_n.detach() * torch.tensor([canvas_width, canvas_height])).cpu().numpy()
a_np = a_px.detach().cpu().numpy()
b_np = b_px.detach().cpu().numpy()
theta_np = theta.detach().cpu().numpy()
ax.scatter(pos_np[:, 0], pos_np[:, 1], c='red', s=15, edgecolors='white', linewidths=0.5)
for idx, (x, y) in enumerate(pos_np):
    ax.add_patch(Ellipse((x, y), width=2*a_np[idx], height=2*b_np[idx],
                          angle=np.degrees(theta_np[idx]),
                          facecolor='none', edgecolor='lime', linewidth=0.8))
    ax.annotate(str(idx), (x, y), color='yellow', fontsize=8,
                xytext=(3, 3), textcoords='offset points')
ax.set_xlim(0, canvas_width)
ax.set_ylim(canvas_height, 0)
fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
ax.axis('off')
plt.savefig(f'{args.outdir}/final_labeled.png', bbox_inches='tight', pad_inches=0, dpi=150)
plt.close(fig)
print('saved final_labeled.png')

final_np = final.detach().clamp(0, 1).cpu().numpy()

# Quiver plot: direction each point moved, second-to-last -> final
final_positions_px = (positions_n.detach() * torch.tensor([canvas_width, canvas_height])).numpy()
u = final_positions_px[:, 0] - second_last_positions_px[:, 0]
v = final_positions_px[:, 1] - second_last_positions_px[:, 1]
fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(final_np, alpha=0.6)
ax.quiver(second_last_positions_px[:, 0], second_last_positions_px[:, 1], u, v,
          angles='xy', scale_units='xy', scale=1, color='red', width=0.003)
ax.scatter(second_last_positions_px[:, 0], second_last_positions_px[:, 1], c='cyan', s=8, label='second-to-last')
ax.scatter(final_positions_px[:, 0], final_positions_px[:, 1], c='red', s=8, label='final')
ax.set_xlim(0, canvas_width)
ax.set_ylim(canvas_height, 0)
ax.legend(loc='upper right')
ax.axis('off')
plt.savefig(f'{args.outdir}/movement_quiver.png', bbox_inches='tight', dpi=150)
plt.close(fig)
print('saved movement_quiver.png')

# Error heatmap against the original
fig, ax = plt.subplots(figsize=(8, 6))
error_map = ((original_np - final_np) ** 2).mean(axis=2)
im = ax.imshow(error_map, cmap='inferno')
ax.axis('off')
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
plt.savefig(f'{args.outdir}/error_heatmap.png', bbox_inches='tight', dpi=150)
plt.close(fig)
print('saved error_heatmap.png')

# --- Comparison grid: degraded | degraded-error | original | reconstruction | reconstruction-error ---
# Both error panels share one colour scale so brightness is directly
# comparable between "before" and "after" -- matches comparison_gaussian_boxed.py.
shared_vmax = max(degraded_error_map.max(), error_map.max())
fig, axes = plt.subplots(1, 5, figsize=(30, 6))
axes[0].imshow(degraded_np)
axes[0].set_title('Degraded (starting canvas)')
axes[0].axis('off')
im0 = axes[1].imshow(degraded_error_map, cmap='inferno', vmin=0, vmax=shared_vmax)
axes[1].set_title(f'Degraded error (mean={degraded_error_map.mean():.5f})')
axes[1].axis('off')
fig.colorbar(im0, ax=axes[1], fraction=0.046, pad=0.04)
axes[2].imshow(original_np)
axes[2].set_title('Original (target)')
axes[2].axis('off')
axes[3].imshow(final_np)
axes[3].set_title('Reconstruction')
axes[3].axis('off')
im1 = axes[4].imshow(error_map, cmap='inferno', vmin=0, vmax=shared_vmax)
axes[4].set_title(f'Reconstruction error (mean={error_map.mean():.5f})')
axes[4].axis('off')
fig.colorbar(im1, ax=axes[4], fraction=0.046, pad=0.04)
plt.savefig(f'{args.outdir}/all_comparison.png', bbox_inches='tight', dpi=150)
plt.close(fig)
print('saved all_comparison.png')

from subprocess import call
call(["ffmpeg", "-framerate", "24", "-i",
    f"{args.outdir}/iters/iter_%d.png", "-vb", "20M",
    f"{args.outdir}/iters.mp4"])