# trianglesoup_rendering.py
# Independent triangles (no shared vertices/edges, unlike a mesh), each
# with a flat colour and a learnable opacity, composited via alpha-over
# in index order. Uses the C++ TriangleSoupRenderFunction ported from
# trianglesoup_prototype.py.
import pydiffvg
import diffvg
import torch
import skimage.io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import numpy as np
from matplotlib.patches import Polygon as MplPolygon

os.makedirs('results/trianglesoup_rendering', exist_ok=True)

N = 1000
iters = 200

# --- Softness annealing ---
# Coverage uses a sigmoid-smoothed edge test with width SOFTNESS pixels.
# Wide softness early gives strong gradients so triangles can move large
# distances quickly; narrow softness late gives sharp edges in the final
# render. Annealed linearly across the run since softness is a plain
# float argument here, not a learnable tensor.
SOFTNESS_START = 4.0
SOFTNESS_END   = 0.5

# Use GPU if available
pydiffvg.set_use_gpu(torch.cuda.is_available())

target = torch.from_numpy(skimage.io.imread('imgs/fruit_basket.png')).to(torch.float32) / 255.0
target = target[:, :, :3]  # keep RGB only
canvas_height, canvas_width = target.shape[0], target.shape[1]
pydiffvg.imwrite(target.cpu(), 'results/trianglesoup_rendering/target.png', gamma=1.0)

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

diffvg.reset_trianglesoup_timing()

# Clear old logs before a fresh run
open('results/trianglesoup_rendering/softness_log.txt', 'w').close()
open('results/trianglesoup_rendering/opacity_log.txt', 'w').close()

# --------------------------------------
# Run Adam iterations.
# --------------------------------------
for t in range(iters):
    optimizer.zero_grad()
    softness = SOFTNESS_START + (SOFTNESS_END - SOFTNESS_START) * (t / max(iters - 1, 1))
    vertices_px = vertices_n * torch.tensor([canvas_width, canvas_height])
    opacity = torch.sigmoid(opacity_logit)
    img = pydiffvg.TriangleSoupRenderFunction.apply(
        vertices_px, colours, opacity, softness, None, canvas_width, canvas_height)
    loss = (img - target).pow(2).sum()  # how wrong is the current render
    loss_history.append(loss.item())
    loss.backward()  # backward -> C++ fills gradients -> deposits into .grad
    print('iter', t, 'loss', loss.item(), 'softness', round(softness, 3))
    with open('results/trianglesoup_rendering/softness_log.txt', 'a') as f:
        f.write(f'iter {t}: softness={softness:.4f} loss={loss.item():.4f}\n')
    with torch.no_grad():
        opacity_current = torch.sigmoid(opacity_logit)
    with open('results/trianglesoup_rendering/opacity_log.txt', 'a') as f:
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

    pydiffvg.imwrite(img.detach().clamp(0, 1).cpu(),
                      'results/trianglesoup_rendering/iter_{}.png'.format(t), gamma=1.0)

print(f'final loss: {loss.item():.4f}')
with open('results/trianglesoup_rendering/final_loss.txt', 'w') as f:
    f.write(str(loss.item()))
diffvg.print_trianglesoup_timing()
fwd_ms, fwd_calls, bwd_ms, bwd_calls = diffvg.get_trianglesoup_timing()
with open('results/trianglesoup_rendering/timing.txt', 'w') as f:
    f.write(f"render_trianglesoup timing (N={N}, {iters} iters, {canvas_width}x{canvas_height})\n")
    f.write(f"Forward:  {fwd_ms:.3f} ms total, {fwd_calls} pixel-calls, {fwd_ms/fwd_calls:.6f} ms/pixel\n")
    f.write(f"Backward: {bwd_ms:.3f} ms total, {bwd_calls} pixel-calls, {bwd_ms/bwd_calls:.6f} ms/pixel\n")

# --------------------------------------
# Plot loss convergence over iterations.
# --------------------------------------
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(loss_history)
ax.set_xlabel('Iteration')
ax.set_ylabel('Loss')
ax.set_title(f'Convergence (N={N} triangles, learnable opacity, softness {SOFTNESS_START}->{SOFTNESS_END}px)')
plt.savefig('results/trianglesoup_rendering/loss_curve.png', bbox_inches='tight', dpi=150)
plt.close(fig)
print('saved loss_curve.png')

# --------------------------------------
# Render the final result.
# --------------------------------------
vertices_px = vertices_n * torch.tensor([canvas_width, canvas_height])
opacity_final = torch.sigmoid(opacity_logit)
final = pydiffvg.TriangleSoupRenderFunction.apply(
    vertices_px, colours, opacity_final, SOFTNESS_END, None, canvas_width, canvas_height)
pydiffvg.imwrite(final.detach().clamp(0, 1).cpu(), 'results/trianglesoup_rendering/final.png', gamma=1.0)

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
plt.savefig('results/trianglesoup_rendering/final_labeled.png', bbox_inches='tight', pad_inches=0, dpi=150)
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
plt.savefig('results/trianglesoup_rendering/movement_quiver.png', bbox_inches='tight', dpi=150)
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
plt.savefig('results/trianglesoup_rendering/opacity_histogram.png', bbox_inches='tight', dpi=150)
plt.close(fig)
print('saved opacity_histogram.png')

# -------------------------------------------------------------------
# Per-pixel error heatmap
# -------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 6))
target_np = target.cpu().numpy()
# Per-pixel error: mean squared difference across RGB channels
error_map = ((target_np - final_np) ** 2).mean(axis=2)
im = ax.imshow(error_map, cmap='inferno')
ax.axis('off')
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
plt.savefig('results/trianglesoup_rendering/error_heatmap.png', bbox_inches='tight', dpi=150)
plt.close(fig)
print('saved error_heatmap.png')

# -------------------------------------------------------------------
# Comparison: target | rendered | error heatmap
# -------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
axes[0].imshow(target_np)
axes[0].set_title('Target')
axes[0].axis('off')
axes[1].imshow(final_np)
axes[1].set_title('Rendered (Triangle soup, learnable opacity)')
axes[1].axis('off')
im = axes[2].imshow(error_map, cmap='inferno')
axes[2].set_title('Error heatmap')
axes[2].axis('off')
fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
plt.savefig('results/trianglesoup_rendering/all_comparison.png', bbox_inches='tight', dpi=150)
plt.close(fig)
print('saved all_comparison.png')

# Convert the intermediate renderings to a video.
from subprocess import call
call(["ffmpeg", "-framerate", "24", "-i",
    "results/trianglesoup_rendering/iter_%d.png", "-vb", "20M",
    "results/trianglesoup_rendering/out.mp4"])