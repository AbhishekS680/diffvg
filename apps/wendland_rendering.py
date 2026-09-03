# wendland_rendering.py
#
# Fits N anisotropic Wendland C2 kernel ellipses directly to a single
# target image. Each ellipse's falloff is compactly supported and
# direction-dependent (semi-axes a/b can differ, plus a rotation angle),
# unlike an isotropic kernel whose influence is the same in every
# direction. Core primitive script -- part of the four-way comparison
# (Wendland / Gaussian / Shepard / Triangle Soup) used in the report.
# Initialization convention matches ellipse_diffvg.py (the master-branch
# hard-edged baseline), so the two are directly comparable.
#
# Usage:
#   python wendland_rendering.py --image imgs/cat.png --n 1000 --iters 200
#
# Args:
#   --image   target image path
#   --n       number of ellipses
#   --iters   number of training iterations
#   --seed    random seed, for reproducibility across primitives
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

OUTDIR = 'results/wendland_rendering'
os.makedirs(OUTDIR, exist_ok=True)

pydiffvg.set_use_gpu(torch.cuda.is_available())

target = torch.from_numpy(skimage.io.imread(args.image)).to(torch.float32) / 255.0
target = target[:, :, :3]
canvas_height, canvas_width = target.shape[0], target.shape[1]
pydiffvg.imwrite(target.cpu(), f'{OUTDIR}/target.png', gamma=1.0)

# Initialize N ellipses: normalized position, log-space semi-axes (stay
# positive, start as small circles), zero initial rotation.
positions_n = torch.rand(N, 2).clone().requires_grad_(True)
colors      = torch.rand(N, 3).clone().requires_grad_(True)
log_a       = torch.full((N,), torch.log(torch.tensor(0.05))).clone().requires_grad_(True)
log_b       = torch.full((N,), torch.log(torch.tensor(0.05))).clone().requires_grad_(True)
theta       = torch.zeros(N).clone().requires_grad_(True)
optimizer = torch.optim.Adam([positions_n, colors, log_a, log_b, theta], lr=1e-2)
loss_history = []
diffvg.reset_ellipse_wendland_timing()

# --------------------------------------
# Main training loop.
# --------------------------------------
for t in range(iters):
    optimizer.zero_grad()
    positions_px = positions_n * torch.tensor([canvas_width, canvas_height])
    a_px = torch.exp(log_a) * canvas_width
    b_px = torch.exp(log_b) * canvas_width
    img = pydiffvg.EllipseWendlandRenderFunction.apply(
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
        # theta is intentionally unclamped -- rotation wraps naturally

    pydiffvg.imwrite(img.clamp(0, 1).cpu(), f'{OUTDIR}/iter_{t}.png', gamma=1.0)

print(f'final loss: {loss.item():.4f}')
diffvg.print_ellipse_wendland_timing()
fwd_ms, fwd_calls, bwd_ms, bwd_calls = diffvg.get_ellipse_wendland_timing()
with open(f'{OUTDIR}/timing.txt', 'w') as f:
    f.write(f"render_ellipse_wendland timing (N={N}, {iters} iters, {canvas_width}x{canvas_height})\n")
    f.write(f"Forward:  {fwd_ms:.3f} ms total, {fwd_calls} pixel-calls, {fwd_ms/fwd_calls:.6f} ms/pixel\n")
    f.write(f"Backward: {bwd_ms:.3f} ms total, {bwd_calls} pixel-calls, {bwd_ms/bwd_calls:.6f} ms/pixel\n")

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(loss_history)
ax.set_xlabel('Iteration')
ax.set_ylabel('Loss')
ax.set_title(f'Convergence (N={N})')
plt.savefig(f'{OUTDIR}/loss_curve.png', bbox_inches='tight', dpi=150)
plt.close(fig)

# --- Final render ---
positions_px = positions_n * torch.tensor([canvas_width, canvas_height])
a_px = torch.exp(log_a) * canvas_width
b_px = torch.exp(log_b) * canvas_width
final = pydiffvg.EllipseWendlandRenderFunction.apply(
    positions_px, colors, a_px, b_px, theta, None, canvas_width, canvas_height)
pydiffvg.imwrite(final.clamp(0, 1).cpu(), f'{OUTDIR}/final.png', gamma=1.0)

# --- Overlay control point locations on final render ---
fig, ax = plt.subplots(figsize=(8, 8))
fig.patch.set_facecolor('black')
display_img = final.detach().clamp(0, 1).cpu().numpy()
ALPHA_CUTOFF = 0.02
alpha_mask = (display_img.max(axis=2) > ALPHA_CUTOFF).astype(np.float32)  # near-black -> transparent
rgba_img = np.dstack([display_img, alpha_mask])
ax.imshow(rgba_img)

# Binary search for t where the Wendland kernel drops to ALPHA_CUTOFF, so
# the drawn outline matches the visible region rather than the full
# geometric ellipse (t=1), which extends past where colour actually shows.
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

# Per-pixel error heatmap
fig, ax = plt.subplots(figsize=(8, 6))
target_np = target.cpu().numpy()
error_map = ((target_np - final_np) ** 2).mean(axis=2)
im = ax.imshow(error_map, cmap='inferno')
ax.axis('off')
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
plt.savefig(f'{OUTDIR}/error_heatmap.png', bbox_inches='tight', dpi=150)
plt.close(fig)
print('saved error_heatmap.png')

# Comparison: target | rendered | error heatmap
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
axes[0].imshow(target_np)
axes[0].set_title('Target')
axes[0].axis('off')
axes[1].imshow(final_np)
axes[1].set_title('Rendered (Wendland C2)')
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