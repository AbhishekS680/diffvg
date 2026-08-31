# comparison_trianglesoup_boxed.py
# Comparison-script version of trianglesoup_rendering_boxed.py
# Uses the C++ TriangleSoupBoxedRenderFunction
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
from matplotlib.patches import Polygon as MplPolygon

parser = argparse.ArgumentParser()
parser.add_argument('--target', required=True)                     # sharper image
parser.add_argument('--degraded', required=True)                   # blurrier image (used as starting canvas)
parser.add_argument('--outdir', default='results/comparison_trianglesoup_boxed')
parser.add_argument('--n', type=int, default=1000, help='Number of triangles')
parser.add_argument('--iters', type=int, default=200, help='Number of training iterations')
args = parser.parse_args()
os.makedirs(args.outdir, exist_ok=True)
os.makedirs(f'{args.outdir}/iters', exist_ok=True)

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

# --- Baseline error heatmap: degraded vs original, before any reconstruction ---
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

# Initialize N triangles (vertices, colour, opacity) randomly
vertices_n = torch.rand(N, 3, 2).clone().requires_grad_(True)  # normalized [0,1]
colours    = torch.rand(N, 3).clone().requires_grad_(True)
# Opacity logit: unconstrained real parameter, squashed through sigmoid
# before being used. Init at 0 -> sigmoid(0) = 0.5, a neutral starting
# opacity (neither fully opaque nor invisible), same reasoning as why
# a fresh triangle shouldn't start either fully blocking or fully
# transparent -- both extremes give weak initial gradients.
opacity_logit = torch.zeros(N).clone().requires_grad_(True)
optimizer = torch.optim.Adam([vertices_n, colours, opacity_logit], lr=1e-2)
loss_history = []
diffvg.reset_trianglesoup_boxed_timing()

# Clear old logs before a fresh run
open(f'{args.outdir}/softness_log.txt', 'w').close()
open(f'{args.outdir}/opacity_log.txt', 'w').close()

# --------------------------------------
# Run Adam iterations. Composited onto the degraded image, same as the
# Wendland/Gaussian boxed comparison scripts.
# --------------------------------------
for t in range(iters):
    optimizer.zero_grad()
    softness = SOFTNESS_START + (SOFTNESS_END - SOFTNESS_START) * (t / max(iters - 1, 1))
    vertices_px = vertices_n * torch.tensor([canvas_width, canvas_height])
    opacity = torch.sigmoid(opacity_logit)
    img = pydiffvg.TriangleSoupBoxedRenderFunction.apply(
        vertices_px, colours, opacity, softness, degraded, canvas_width, canvas_height)
    loss = (img - original).pow(2).sum()  # how wrong is the current render
    loss_history.append(loss.item())
    loss.backward()  # backward -> C++ fills gradients -> deposits into .grad

    print('iter', t, 'loss', loss.item(), 'softness', round(softness, 3))
    with open(f'{args.outdir}/softness_log.txt', 'a') as f:
        f.write(f'iter {t}: softness={softness:.4f} loss={loss.item():.4f}\n')
    with torch.no_grad():
        opacity_current = torch.sigmoid(opacity_logit)
    with open(f'{args.outdir}/opacity_log.txt', 'a') as f:
        f.write(f'iter {t}: opacity[min={opacity_current.min().item():.3f} '
                f'max={opacity_current.max().item():.3f} '
                f'mean={opacity_current.mean().item():.3f}]\n')

    optimizer.step()  # Adam reads .grad -> moves vertices_n, colours, opacity_logit

    if t == iters - 2:
        second_last_vertices_px = (vertices_n.detach() * torch.tensor([canvas_width, canvas_height])).clone().numpy()

    # Helps the optimized parameters stay inside their bounds after each optimizer.step()
    with torch.no_grad():
        vertices_n.clamp_(0.0, 1.0)
        colours.clamp_(0.0, 1.0)
        # No clamp needed for opacity_logit -- sigmoid already bounds the
        # actual opacity to [0,1] regardless of the logit's raw value.

    pydiffvg.imwrite(img.detach().clamp(0, 1).cpu(), f'{args.outdir}/iters/iter_{t}.png', gamma=1.0)

print(f'final loss: {loss.item():.4f}')
with open(f'{args.outdir}/final_loss.txt', 'w') as f:
    f.write(str(loss.item()))

diffvg.print_trianglesoup_boxed_timing()
fwd_ms, fwd_calls, bwd_ms, bwd_calls = diffvg.get_trianglesoup_boxed_timing()
with open(f'{args.outdir}/timing.txt', 'w') as f:
    f.write(f"render_trianglesoup_boxed timing (N={N}, {iters} iters, {canvas_width}x{canvas_height})\n")
    f.write(f"Forward:  {fwd_ms:.3f} ms total, {fwd_calls} pixel-calls, {fwd_ms/fwd_calls:.6f} ms/pixel\n")
    f.write(f"Backward: {bwd_ms:.3f} ms total, {bwd_calls} pixel-calls, {bwd_ms/bwd_calls:.6f} ms/pixel\n")

# --------------------------------------
# Plot loss convergence over iterations.
# --------------------------------------
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(loss_history)
ax.set_xlabel('Iteration')
ax.set_ylabel('Loss')
ax.set_title(f'Convergence (N={N} triangles, boxed/tile-grid, learnable opacity, softness {SOFTNESS_START}->{SOFTNESS_END}px, composited on blurry image)')
plt.savefig(f'{args.outdir}/loss_curve.png', bbox_inches='tight', dpi=150)
plt.close(fig)
print('saved loss_curve.png')

# --------------------------------------
# Render the final result.
# --------------------------------------
vertices_px = vertices_n * torch.tensor([canvas_width, canvas_height])
opacity_final = torch.sigmoid(opacity_logit)
final = pydiffvg.TriangleSoupBoxedRenderFunction.apply(
    vertices_px, colours, opacity_final, SOFTNESS_END, degraded, canvas_width, canvas_height)
final = final.clamp(0, 1)
pydiffvg.imwrite(final.detach().cpu(), f'{args.outdir}/final.png', gamma=1.0)

# --- Triangles-only render (blank background instead of degraded) ---
blank_canvas = torch.zeros_like(degraded)  # black background -- shows raw triangle coverage
triangles_only = pydiffvg.TriangleSoupBoxedRenderFunction.apply(
    vertices_px, colours, opacity_final, SOFTNESS_END, blank_canvas, canvas_width, canvas_height)
triangles_only = triangles_only.clamp(0, 1)
pydiffvg.imwrite(triangles_only.detach().cpu(), f'{args.outdir}/triangles_only.png', gamma=1.0)
print('saved triangles_only.png')

# -------------------------------------------------------------------
# Visualization: overlay triangle outlines on the final render, edge
# opacity reflecting each triangle's learned opacity.
# -------------------------------------------------------------------
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
ax.axis('off')  # Hides the plot's axis lines, ticks, and labels
plt.savefig(f'{args.outdir}/final_labeled.png', bbox_inches='tight', pad_inches=0, dpi=150)
plt.close(fig)  # Releases the figure from memory
print('saved final_labeled.png')

final_np = final.detach().clamp(0, 1).cpu().numpy()

# -------------------------------------------------------------------
# Quiver plot: direction each vertex moved, second-to-last -> final
# -------------------------------------------------------------------
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
plt.savefig(f'{args.outdir}/movement_quiver.png', bbox_inches='tight', dpi=150)
plt.close(fig)
print('saved movement_quiver.png')

# -------------------------------------------------------------------
# Opacity distribution histogram
# -------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(opacity_np, bins=40, color='#3C896D')
ax.set_xlabel('Learned opacity')
ax.set_ylabel('Count')
ax.set_title('Final opacity distribution across triangles')
plt.savefig(f'{args.outdir}/opacity_histogram.png', bbox_inches='tight', dpi=150)
plt.close(fig)
print('saved opacity_histogram.png')

# -------------------------------------------------------------------
# Per-pixel error heatmap against the original
# -------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 6))
error_map = ((original_np - final_np) ** 2).mean(axis=2)
im = ax.imshow(error_map, cmap='inferno')
ax.axis('off')
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
plt.savefig(f'{args.outdir}/error_heatmap.png', bbox_inches='tight', dpi=150)
plt.close(fig)
print('saved error_heatmap.png')

# -------------------------------------------------------------------
# Comparison grid: degraded | degraded-error | original | reconstruction | reconstruction-error
# Both error panels share one color scale so brightness is directly
# comparable between "before" and "after" -- same convention as the
# shared-scale multi-primitive comparison script.
# -------------------------------------------------------------------
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
axes[3].set_title('Reconstruction (triangle soup, boxed)')
axes[3].axis('off')

im1 = axes[4].imshow(error_map, cmap='inferno', vmin=0, vmax=shared_vmax)
axes[4].set_title(f'Reconstruction error (mean={error_map.mean():.5f})')
axes[4].axis('off')
fig.colorbar(im1, ax=axes[4], fraction=0.046, pad=0.04)

plt.savefig(f'{args.outdir}/all_comparison.png', bbox_inches='tight', dpi=150)
plt.close(fig)
print('saved all_comparison.png')

# Convert the intermediate renderings to a video.
from subprocess import call
call(["ffmpeg", "-framerate", "24", "-i",
    f"{args.outdir}/iters/iter_%d.png", "-vb", "20M",
    f"{args.outdir}/iters.mp4"])