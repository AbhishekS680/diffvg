# trianglesoup_rendering_boxed_no_opacity.py
#
# Ablation: same as trianglesoup_rendering_boxed.py, but opacity is
# fixed at 1.0 (fully solid triangles) instead of being learned. Isolates
# what a learnable opacity actually contributes to reconstruction
# quality -- with opacity fixed, later-indexed triangles fully occlude
# earlier ones in the alpha-over compositing order, since there is no
# way for a triangle to become partially transparent.
#
# NOT part of the core four-primitive comparison used in the report --
# this is a targeted ablation on the opacity parameter specifically, so
# it deliberately keeps plain uniform-random initialization (same as
# the core comparison) to isolate opacity as the only variable changed.
#
# Usage:
#   python trianglesoup_rendering_boxed_no_opacity.py --image imgs/cat.png --n 1000 --iters 200
#
# Args:
#   --image   target image path
#   --n       number of triangles
#   --iters   number of training iterations
#   --seed    random seed, for reproducibility
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

SOFTNESS_START = 4.0
SOFTNESS_END   = 0.5

OUTDIR = 'results/trianglesoup_rendering_boxed_no_opacity'
os.makedirs(OUTDIR, exist_ok=True)

pydiffvg.set_use_gpu(torch.cuda.is_available())

target = torch.from_numpy(skimage.io.imread(args.image)).to(torch.float32) / 255.0
target = target[:, :, :3]
canvas_height, canvas_width = target.shape[0], target.shape[1]
pydiffvg.imwrite(target.cpu(), f'{OUTDIR}/target.png', gamma=1.0)

# --- Initialize N triangles (vertices, colour) uniformly at random ---
vertices_n = torch.rand(N, 3, 2).clone().requires_grad_(True)  # normalized [0,1]
colours    = torch.rand(N, 3).clone().requires_grad_(True)
opacity = torch.ones(N)  # fixed at full opacity, not in the optimizer
optimizer = torch.optim.Adam([vertices_n, colours], lr=1e-2)
loss_history = []
diffvg.reset_trianglesoup_boxed_timing()

open(f'{OUTDIR}/softness_log.txt', 'w').close()

# --------------------------------------
# Main training loop.
# --------------------------------------
for t in range(iters):
    optimizer.zero_grad()
    softness = SOFTNESS_START + (SOFTNESS_END - SOFTNESS_START) * (t / max(iters - 1, 1))
    vertices_px = vertices_n * torch.tensor([canvas_width, canvas_height])
    img = pydiffvg.TriangleSoupBoxedRenderFunction.apply(
        vertices_px, colours, opacity, softness, None, canvas_width, canvas_height)
    loss = (img - target).pow(2).sum()
    loss_history.append(loss.item())
    loss.backward()

    print('iter', t, 'loss', loss.item(), 'softness', round(softness, 3))
    with open(f'{OUTDIR}/softness_log.txt', 'a') as f:
        f.write(f'iter {t}: softness={softness:.4f} loss={loss.item():.4f}\n')

    optimizer.step()  # only moves vertices_n, colours -- opacity is fixed

    if t == iters - 2:
        second_last_vertices_px = (vertices_n.detach() * torch.tensor([canvas_width, canvas_height])).clone().numpy()

    with torch.no_grad():
        vertices_n.clamp_(0.0, 1.0)
        colours.clamp_(0.0, 1.0)

    pydiffvg.imwrite(img.detach().clamp(0, 1).cpu(), f'{OUTDIR}/iter_{t}.png', gamma=1.0)

print(f'final loss: {loss.item():.4f}')
with open(f'{OUTDIR}/final_loss.txt', 'w') as f:
    f.write(str(loss.item()))

diffvg.print_trianglesoup_boxed_timing()
fwd_ms, fwd_calls, bwd_ms, bwd_calls = diffvg.get_trianglesoup_boxed_timing()
with open(f'{OUTDIR}/timing.txt', 'w') as f:
    f.write(f"render_trianglesoup_boxed timing, no opacity (N={N}, {iters} iters, {canvas_width}x{canvas_height})\n")
    f.write(f"Forward:  {fwd_ms:.3f} ms total, {fwd_calls} pixel-calls, {fwd_ms/fwd_calls:.6f} ms/pixel\n")
    f.write(f"Backward: {bwd_ms:.3f} ms total, {bwd_calls} pixel-calls, {bwd_ms/bwd_calls:.6f} ms/pixel\n")

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(loss_history)
ax.set_xlabel('Iteration')
ax.set_ylabel('Loss')
ax.set_title(f'Convergence (N={N} triangles, boxed/tile-grid, no opacity, softness {SOFTNESS_START}->{SOFTNESS_END}px)')
plt.savefig(f'{OUTDIR}/loss_curve.png', bbox_inches='tight', dpi=150)
plt.close(fig)
print('saved loss_curve.png')

# --- Final render ---
vertices_px = vertices_n * torch.tensor([canvas_width, canvas_height])
final = pydiffvg.TriangleSoupBoxedRenderFunction.apply(
    vertices_px, colours, opacity, SOFTNESS_END, None, canvas_width, canvas_height)
pydiffvg.imwrite(final.detach().clamp(0, 1).cpu(), f'{OUTDIR}/final.png', gamma=1.0)

# Overlay triangle outlines on final render. No opacity variation to
# reflect, so every outline is drawn at full alpha (unlike the
# learnable-opacity version, which dims outlines by learned opacity).
fig, ax = plt.subplots(figsize=(8, 8))
display_img = final.detach().clamp(0, 1).cpu().numpy()
ax.imshow(display_img)
verts_np = vertices_px.detach().cpu().numpy()
for idx in range(N):
    tri = verts_np[idx]
    ax.add_patch(MplPolygon(tri, closed=True, facecolor='none', edgecolor='lime', linewidth=0.4))
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
axes[1].set_title('Rendered (Triangle soup, boxed, no opacity)')
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