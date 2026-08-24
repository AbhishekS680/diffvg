# comparison_shepard.py
# Direct reconstruction: control points fit directly to the clear (original) target image
import argparse
import pydiffvg
import torch
import skimage.io
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import diffvg

parser = argparse.ArgumentParser()
parser.add_argument('--target', required=True)                     # sharper image
parser.add_argument('--degraded', required=True)                   # blurrier image (reference only)
parser.add_argument('--outdir', default='results/comparison_shepard')
args = parser.parse_args()
os.makedirs(args.outdir, exist_ok=True)
os.makedirs(f'{args.outdir}/iters', exist_ok=True)

N = 1000
q = 3.0
iters = 200

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

# Init: random positions/colors
positions_n = (torch.rand(N, 2)).clone().requires_grad_(True)
colors = (torch.rand(N, 3)).clone().requires_grad_(True)
optimizer = torch.optim.Adam([positions_n, colors], lr=1e-2)
loss_history = []
open(f'{args.outdir}/gradient_log.txt', 'w').close()
diffvg.reset_shepard_timing()

# --- Optimization loop: points fit directly against the clear image ---
for t in range(iters):
    optimizer.zero_grad()
    positions = positions_n * torch.tensor([canvas_width, canvas_height])
    img = pydiffvg.ShepardRenderFunction.apply(positions, colors, q, canvas_width, canvas_height)
    loss = (img - original).pow(2).sum()
    # Repulsion: move control points that are too close to each other
    # positions_px = positions_n * torch.tensor([canvas_width, canvas_height])
    # diff = positions_px.unsqueeze(0) - positions_px.unsqueeze(1)
    # dist_sq = diff.pow(2).sum(dim=2) + 0.001
    # eye_mask = 1 - torch.eye(N)
    # repulsion = (1.0 / dist_sq * eye_mask).sum()
    # loss = loss + 0.01 * repulsion
    loss_history.append(loss.item())
    loss.backward()

    pos_grad_norms = positions_n.grad.norm(dim=1)
    color_grad_norms = colors.grad.norm(dim=1)
    print('iter', t, 'loss', loss.item())
    print('position grad — min/max/mean:',
      pos_grad_norms.min().item(), pos_grad_norms.max().item(), pos_grad_norms.mean().item())
    print('color grad — min/max/mean:',
      color_grad_norms.min().item(), color_grad_norms.max().item(), color_grad_norms.mean().item())
    with open(f'{args.outdir}/gradient_log.txt', 'a') as f:
        f.write(f'iter {t}\n')
        f.write(f'  position grad norms: {pos_grad_norms.detach().numpy().tolist()}\n')
        f.write(f'  color grad norms: {color_grad_norms.detach().numpy().tolist()}\n')

    optimizer.step()

    if t == iters - 2:
        second_last_positions_px = (positions_n.detach() * torch.tensor([canvas_width, canvas_height])).clone().numpy()

    with torch.no_grad():
        positions_n.clamp_(0.0, 1.0)
        colors.clamp_(0.0, 1.0)

    pydiffvg.imwrite(img.detach().clamp(0, 1).cpu(), f'{args.outdir}/iters/iter_{t}.png', gamma=1.0)

print(f'final loss: {loss.item():.4f}')
with open(f'{args.outdir}/final_loss.txt', 'w') as f:
    f.write(str(loss.item()))

diffvg.print_shepard_timing()
fwd_ms, fwd_calls, bwd_ms, bwd_calls = diffvg.get_shepard_timing()
with open(f'{args.outdir}/timing.txt', 'w') as f:
    f.write(f"render_shepard timing (N={N}, {iters} iters, {canvas_width}x{canvas_height})\n")
    f.write(f"Forward:  {fwd_ms:.3f} ms total, {fwd_calls} pixel-calls, {fwd_ms/fwd_calls:.6f} ms/pixel\n")
    f.write(f"Backward: {bwd_ms:.3f} ms total, {bwd_calls} pixel-calls, {bwd_ms/bwd_calls:.6f} ms/pixel\n")

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(loss_history)
ax.set_xlabel('Iteration')
ax.set_ylabel('Loss')
ax.set_title(f'Convergence (N={N}, q={q}, fitting directly to target)')
plt.savefig(f'{args.outdir}/loss_curve.png', bbox_inches='tight', dpi=150)
plt.close(fig)
print('saved loss_curve.png')

# --- Final render ---
final = pydiffvg.ShepardRenderFunction.apply(
    positions_n * torch.tensor([canvas_width, canvas_height]), colors, q, canvas_width, canvas_height)
final = final.clamp(0, 1)
pydiffvg.imwrite(final.detach().cpu(), f'{args.outdir}/final.png', gamma=1.0)

# Overlay control points on final render
fig, ax = plt.subplots(figsize=(8, 8))
final_np = final.detach().clamp(0, 1).cpu().numpy()
ax.imshow(final_np)
pos_np = (positions_n.detach() * torch.tensor([canvas_width, canvas_height])).cpu().numpy()
ax.scatter(pos_np[:, 0], pos_np[:, 1], c='red', s=15, edgecolors='white', linewidths=0.5)
for idx, (x, y) in enumerate(pos_np):
    ax.annotate(str(idx), (x, y), color='yellow', fontsize=8,
                xytext=(3, 3), textcoords='offset points')
ax.set_xlim(0, canvas_width)
ax.set_ylim(canvas_height, 0)
fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
ax.axis('off')
plt.savefig(f'{args.outdir}/final_labeled.png', bbox_inches='tight', pad_inches=0, dpi=150)
plt.close(fig)
print('saved final_labeled.png')

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
error_map = ((original_np - final_np) ** 2).mean(axis=2)
im = ax.imshow(error_map, cmap='inferno')
ax.axis('off')
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
plt.savefig(f'{args.outdir}/error_heatmap.png', bbox_inches='tight', dpi=150)
plt.close(fig)
print('saved error_heatmap.png')

# Comparison grid: degraded | degraded-error | original | reconstruction | reconstruction-error
# Both error panels share one color scale so brightness is directly
# comparable between "before" and "after" -- same idea as the shared-scale
# multi-primitive comparison script.
shared_vmax = max(degraded_error_map.max(), error_map.max())

fig, axes = plt.subplots(1, 5, figsize=(30, 6))
axes[0].imshow(degraded_np)
axes[0].set_title('Degraded (reference only)')
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