# wendland_rendering_boxed.py
#
# Same as wendland_rendering.py, but uses the tile-grid accelerated
# renderer (see Section "Tile-Grid Acceleration" in the report). Core
# primitive script -- part of the four-way comparison (Wendland /
# Gaussian / Shepard / Triangle Soup) used in the report.
#
# Usage:
#   python wendland_rendering_boxed.py --image imgs/cat.png --n 1000 --iters 200
#
# Args:
#   --image   target image path
#   --n       number of ellipses
#   --iters   number of training iterations
#   --seed    random seed, for reproducibility across primitives
#
# After the main training loop, runs an optional focus phase: a second
# round of iterations on the same ellipses, reweighting the loss by
# pass one's error heatmap so poorly-reconstructed pixels get more
# gradient pull. Nothing is added or frozen -- the same ellipses are
# just pushed harder toward fixing their own mistakes.
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
parser.add_argument('--image', default='imgs/fruit_basket.png', help='Target image path')
parser.add_argument('--n', type=int, default=1000, help='Number of ellipses')
parser.add_argument('--iters', type=int, default=200, help='Number of training iterations')
parser.add_argument('--seed', type=int, default=0, help='Random seed')
args = parser.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)

N = args.n
iters = args.iters

FOCUS_ITERS = 200
FOCUS_WEIGHT_SCALE = 6.0  # how much extra weight the worst pixels get, relative to the best

OUTDIR = 'results/wendland_rendering_boxed'
os.makedirs(OUTDIR, exist_ok=True)
os.makedirs(f'{OUTDIR}/focus_iters', exist_ok=True)

pydiffvg.set_use_gpu(torch.cuda.is_available())

target = torch.from_numpy(skimage.io.imread(args.image)).to(torch.float32) / 255.0
target = target[:, :, :3]
canvas_height, canvas_width = target.shape[0], target.shape[1]
pydiffvg.imwrite(target.cpu(), f'{OUTDIR}/target.png', gamma=1.0)

# Initialize N ellipses: normalized position, log-space semi-axes (stay
# positive, start as small circles), zero initial rotation. Matches
# wendland_rendering.py's convention.
positions_n = torch.rand(N, 2).clone().requires_grad_(True)
colors      = torch.rand(N, 3).clone().requires_grad_(True)
log_a       = torch.full((N,), torch.log(torch.tensor(0.05))).clone().requires_grad_(True)
log_b       = torch.full((N,), torch.log(torch.tensor(0.05))).clone().requires_grad_(True)
theta       = torch.zeros(N).clone().requires_grad_(True)
optimizer = torch.optim.Adam([positions_n, colors, log_a, log_b, theta], lr=1e-2)
loss_history = []
diffvg.reset_ellipse_wendland_boxed_timing()

# --------------------------------------
# Main training loop.
# --------------------------------------
for t in range(iters):
    optimizer.zero_grad()
    positions_px = positions_n * torch.tensor([canvas_width, canvas_height])
    a_px = torch.exp(log_a) * canvas_width
    b_px = torch.exp(log_b) * canvas_width
    img = pydiffvg.EllipseWendlandBoxedRenderFunction.apply(
        positions_px, colors, a_px, b_px, theta, None, canvas_width, canvas_height)
    loss = (img - target).pow(2).sum()
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

    pydiffvg.imwrite(img.clamp(0, 1).cpu(), f'{OUTDIR}/iter_{t}.png', gamma=1.0)

print(f'final loss: {loss.item():.4f}')
diffvg.print_ellipse_wendland_boxed_timing()
fwd_ms, fwd_calls, bwd_ms, bwd_calls = diffvg.get_ellipse_wendland_boxed_timing()
with open(f'{OUTDIR}/timing.txt', 'w') as f:
    f.write(f"render_ellipse_wendland_boxed timing (N={N}, {iters} iters, {canvas_width}x{canvas_height})\n")
    f.write(f"Forward:  {fwd_ms:.3f} ms total, {fwd_calls} pixel-calls, {fwd_ms/fwd_calls:.6f} ms/pixel\n")
    f.write(f"Backward: {bwd_ms:.3f} ms total, {bwd_calls} pixel-calls, {bwd_ms/bwd_calls:.6f} ms/pixel\n")

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(loss_history)
ax.set_xlabel('Iteration')
ax.set_ylabel('Loss')
ax.set_title(f'Convergence (N={N}, boxed/tile-grid Wendland)')
plt.savefig(f'{OUTDIR}/loss_curve.png', bbox_inches='tight', dpi=150)
plt.close(fig)
print('saved loss_curve.png')

# --- Final render (pass one) ---
positions_px = positions_n * torch.tensor([canvas_width, canvas_height])
a_px = torch.exp(log_a) * canvas_width
b_px = torch.exp(log_b) * canvas_width
final = pydiffvg.EllipseWendlandBoxedRenderFunction.apply(
    positions_px, colors, a_px, b_px, theta, None, canvas_width, canvas_height)
pydiffvg.imwrite(final.clamp(0, 1).cpu(), f'{OUTDIR}/final.png', gamma=1.0)

# --- Overlay ellipse boundaries and control point locations on final render ---
fig, ax = plt.subplots(figsize=(8, 8))
fig.patch.set_facecolor('black')
display_img = final.detach().clamp(0, 1).cpu().numpy()
ALPHA_CUTOFF = 0.02
alpha_mask = (display_img.max(axis=2) > ALPHA_CUTOFF).astype(np.float32)
rgba_img = np.dstack([display_img, alpha_mask])
ax.imshow(rgba_img)

# Binary search for the t value where the Wendland kernel drops to
# ALPHA_CUTOFF, so the drawn boundary matches where the ellipse visually
# fades out rather than the full geometric ellipse (t=1).
lo, hi = 0.0, 1.0
for _ in range(50):
    mid = (lo + hi) / 2
    val = (1 - mid)**4 * (4 * mid + 1)
    if val > ALPHA_CUTOFF:
        lo = mid
    else:
        hi = mid
t_boundary = mid

pos_np = (positions_n.detach() * torch.tensor([canvas_width, canvas_height])).cpu().numpy()
a_np = a_px.detach().cpu().numpy()
b_np = b_px.detach().cpu().numpy()
theta_np = theta.detach().cpu().numpy()
ax.scatter(pos_np[:, 0], pos_np[:, 1], c='red', s=15, edgecolors='white', linewidths=0.5)
for idx, (x, y) in enumerate(pos_np):
    ax.add_patch(Ellipse((x, y), width=2*a_np[idx]*t_boundary, height=2*b_np[idx]*t_boundary,
                          angle=np.degrees(theta_np[idx]),
                          facecolor='none', edgecolor='lime', linewidth=0.8))
    ax.annotate(str(idx), (x, y), color='yellow', fontsize=8,
                xytext=(3, 3), textcoords='offset points')
ax.set_xlim(0, canvas_width)
ax.set_ylim(canvas_height, 0)
fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
ax.axis('off')
plt.savefig(f'{OUTDIR}/final_labeled.png', bbox_inches='tight', pad_inches=0, dpi=150, facecolor='black')
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
plt.savefig(f'{OUTDIR}/movement_quiver.png', bbox_inches='tight', dpi=150)
plt.close(fig)
print('saved movement_quiver.png')

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
# error, which actively degrades the reconstruction. See focus_summary.txt.
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

focus_optimizer = torch.optim.Adam([positions_n, colors, log_a, log_b, theta], lr=5e-3)
focus_loss_history = []
for t in range(FOCUS_ITERS):
    focus_optimizer.zero_grad()
    positions_px = positions_n * torch.tensor([canvas_width, canvas_height])
    a_px = torch.exp(log_a) * canvas_width
    b_px = torch.exp(log_b) * canvas_width
    img = pydiffvg.EllipseWendlandBoxedRenderFunction.apply(
        positions_px, colors, a_px, b_px, theta, None, canvas_width, canvas_height)
    focus_loss = ((img - target).pow(2) * weight_map).sum()
    focus_loss_history.append(focus_loss.item())
    focus_loss.backward()
    print('focus iter', t, 'loss', focus_loss.item())
    focus_optimizer.step()
    with torch.no_grad():
        positions_n.clamp_(0.0, 1.0)
        colors.clamp_(0.0, 1.0)
        log_a.clamp_(torch.log(torch.tensor(0.01)), torch.log(torch.tensor(1.0)))
        log_b.clamp_(torch.log(torch.tensor(0.01)), torch.log(torch.tensor(1.0)))
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
positions_px = positions_n * torch.tensor([canvas_width, canvas_height])
a_px = torch.exp(log_a) * canvas_width
b_px = torch.exp(log_b) * canvas_width
final_focused = pydiffvg.EllipseWendlandBoxedRenderFunction.apply(
    positions_px, colors, a_px, b_px, theta, None, canvas_width, canvas_height)
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
axes[1].set_title('Rendered (boxed)')
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