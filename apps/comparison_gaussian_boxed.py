# comparison_gaussian_boxed.py
# Same as comparison_gaussian.py, but uses the tile-grid accelerated renderer

import argparse
import pydiffvg
import diffvg
import torch
import skimage.io
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from matplotlib.patches import Ellipse

parser = argparse.ArgumentParser()
parser.add_argument('--target', required=True)                     # sharper image
parser.add_argument('--degraded', required=True)                   # blurrier image (used as starting canvas)
parser.add_argument('--outdir', default='results/comparison_gaussian_boxed')
args = parser.parse_args()
os.makedirs(args.outdir, exist_ok=True)
os.makedirs(f'{args.outdir}/iters', exist_ok=True)

N = 10000
iters = 500

# --- Semi-axis size penalty ---
# Discourages a/b (in pixels) from growing past MAX_AXIS_PX. Soft quadratic
# penalty: zero below the threshold, then grows smoothly, so Adam gets a
# real gradient pushing oversized ellipses back down instead of a hard
# clamp (no gradient) or a discontinuous jump (unstable).
MAX_AXIS_PX = 10.0
AXIS_PENALTY_WEIGHT = 1.0

pydiffvg.set_use_gpu(torch.cuda.is_available())

# --- Load images ---
original = torch.from_numpy(skimage.io.imread(args.target)).to(torch.float32) / 255.0
original = original[:, :, :3]
canvas_height, canvas_width = original.shape[0], original.shape[1]
pydiffvg.imwrite(original.cpu(), f'{args.outdir}/target_original.png', gamma=1.0)
print('original shape:', original.shape)
degraded_np = skimage.io.imread(args.degraded).astype(np.float32) / 255.0
degraded_np = degraded_np[:, :, :3]
degraded = torch.from_numpy(degraded_np)
pydiffvg.imwrite(degraded.cpu(), f'{args.outdir}/init_source_degraded.png', gamma=1.0)
assert degraded_np.shape[0] == canvas_height and degraded_np.shape[1] == canvas_width, \
    'Degraded and original images must be the same size'

# Init: random positions/shape/colour
positions_n = (torch.rand(N, 2)).clone().requires_grad_(True)
colors = (torch.rand(N, 3)).clone().requires_grad_(True)
log_a = torch.full((N,), torch.log(torch.tensor(0.05))).clone().requires_grad_(True)
log_b = torch.full((N,), torch.log(torch.tensor(0.05))).clone().requires_grad_(True)
theta = torch.zeros(N).clone().requires_grad_(True)
optimizer = torch.optim.Adam([positions_n, colors, log_a, log_b, theta], lr=1e-2)
loss_history = []
diffvg.reset_ellipse_gaussian_boxed_timing()

# Clear old axis-penalty log before a fresh run
open(f'{args.outdir}/axis_penalty_log.txt', 'w').close()

# --- Optimization loop ---
for t in range(iters):
    optimizer.zero_grad()
    positions_px = positions_n * torch.tensor([canvas_width, canvas_height])
    a_px = torch.exp(log_a) * canvas_width
    b_px = torch.exp(log_b) * canvas_width
    img = pydiffvg.EllipseGaussianBoxedRenderFunction.apply(
        positions_px, colors, a_px, b_px, theta, degraded, canvas_width, canvas_height)
    loss = (img - original).pow(2).sum()

    # Soft penalty on oversized semi-axes: zero contribution while a/b stay
    # under MAX_AXIS_PX, quadratic growth past it.
    axis_penalty = (torch.clamp(a_px - MAX_AXIS_PX, min=0).pow(2).sum()
                    + torch.clamp(b_px - MAX_AXIS_PX, min=0).pow(2).sum())
    loss = loss + AXIS_PENALTY_WEIGHT * axis_penalty
    loss_history.append(loss.item())
    loss.backward()
    print('iter', t, 'loss', loss.item())
    a_current = torch.exp(log_a.detach())
    b_current = torch.exp(log_b.detach())
    print('a range:', a_current.min().item(), '-', a_current.max().item())
    print('b range:', b_current.min().item(), '-', b_current.max().item())
    n_over = int(((a_px.detach() > MAX_AXIS_PX) | (b_px.detach() > MAX_AXIS_PX)).sum().item())
    with open(f'{args.outdir}/axis_penalty_log.txt', 'a') as f:
        f.write(f'iter {t}: penalty={axis_penalty.item():.6f} '
                f'n_over_threshold={n_over}/{N} '
                f'a[min={a_px.detach().min().item():.3f} max={a_px.detach().max().item():.3f} '
                f'mean={a_px.detach().mean().item():.3f}] '
                f'b[min={b_px.detach().min().item():.3f} max={b_px.detach().max().item():.3f} '
                f'mean={b_px.detach().mean().item():.3f}]\n')
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

# Write timing results to a text file
diffvg.print_ellipse_gaussian_boxed_timing()
fwd_ms, fwd_calls, bwd_ms, bwd_calls = diffvg.get_ellipse_gaussian_boxed_timing()
with open(f'{args.outdir}/timing.txt', 'w') as f:
    f.write(f"render_ellipse_gaussian_boxed timing (N={N}, {iters} iters, {canvas_width}x{canvas_height})\n")
    f.write(f"Forward:  {fwd_ms:.3f} ms total, {fwd_calls} pixel-calls, {fwd_ms/fwd_calls:.6f} ms/pixel\n")
    f.write(f"Backward: {bwd_ms:.3f} ms total, {bwd_calls} pixel-calls, {bwd_ms/bwd_calls:.6f} ms/pixel\n")
with open(f'{args.outdir}/final_loss.txt', 'w') as f:
    f.write(str(loss.item()))
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(loss_history)
ax.set_xlabel('Iteration')
ax.set_ylabel('Loss')
ax.set_title(f'Convergence (N={N}, boxed/tile-grid Gaussian, composited on blurry image)')
plt.savefig(f'{args.outdir}/loss_curve.png', bbox_inches='tight', dpi=150)
plt.close(fig)
print('saved loss_curve.png')

# --- Final render ---
positions_px = positions_n * torch.tensor([canvas_width, canvas_height])
a_px = torch.exp(log_a) * canvas_width
b_px = torch.exp(log_b) * canvas_width
final = pydiffvg.EllipseGaussianBoxedRenderFunction.apply(
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

# Quiver plot: direction each point moved
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
original_np = original.cpu().numpy()
error_map = ((original_np - final_np) ** 2).mean(axis=2)
im = ax.imshow(error_map, cmap='inferno')
ax.axis('off')
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
plt.savefig(f'{args.outdir}/error_heatmap.png', bbox_inches='tight', dpi=150)
plt.close(fig)
print('saved error_heatmap.png')

# Comparison grid
fig, axes = plt.subplots(1, 4, figsize=(24, 6))
axes[0].imshow(degraded_np)
axes[0].set_title('Degraded (starting canvas)')
axes[0].axis('off')
axes[1].imshow(original_np)
axes[1].set_title('Original (target)')
axes[1].axis('off')
axes[2].imshow(final_np)
axes[2].set_title('Reconstruction')
axes[2].axis('off')
im = axes[3].imshow(error_map, cmap='inferno')
axes[3].set_title('Error heatmap')
axes[3].axis('off')
fig.colorbar(im, ax=axes[3], fraction=0.046, pad=0.04)
plt.savefig(f'{args.outdir}/all_comparison.png', bbox_inches='tight', dpi=150)
plt.close(fig)
print('saved all_comparison.png')

from subprocess import call
call(["ffmpeg", "-framerate", "24", "-i",
    f"{args.outdir}/iters/iter_%d.png", "-vb", "20M",
    f"{args.outdir}/iters.mp4"])