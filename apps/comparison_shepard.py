# comparison_shepard.py
# Shepard reconstruction initialized from a degraded photo,
# but optimized against the original (clean) target image.
import pydiffvg
import torch
import skimage.io
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

os.makedirs('results/comparison_shepard', exist_ok=True)

N = 100       # Number of control points
q = 3.0      # Controls how sharply the falloff happens for each control point
iters = 250

pydiffvg.set_use_gpu(torch.cuda.is_available())

# --- Load images ---
original = torch.from_numpy(skimage.io.imread('imgs/arch.jpg')).to(torch.float32) / 255.0
original = original[:, :, :3]  # keep RGB only
canvas_height, canvas_width = original.shape[0], original.shape[1]
pydiffvg.imwrite(original.cpu(), 'results/comparison_shepard/target_original.png', gamma=1.0)
print('original shape:', original.shape)

degraded_np = skimage.io.imread('imgs/arch_blurry.jpg').astype(np.float32) / 255.0
degraded_np = degraded_np[:, :, :3]
pydiffvg.imwrite(torch.from_numpy(degraded_np), 'results/comparison_shepard/init_source_degraded.png', gamma=1.0)
assert degraded_np.shape[0] == canvas_height and degraded_np.shape[1] == canvas_width, \
    'Degraded and original images must be the same size'

# --- Initialize: random positions, colors = degraded photo's average color ---
background_color = degraded_np.reshape(-1, 3).mean(axis=0)  # single RGB value, head start

positions_n = (torch.rand(N, 2)).clone().requires_grad_(True)  # normalized [0,1]
colors = torch.tensor(background_color, dtype=torch.float32).unsqueeze(0).repeat(N, 1)
colors = (colors + torch.rand_like(colors) * 0.05).clamp(0, 1).clone().requires_grad_(True)

optimizer = torch.optim.Adam([positions_n, colors], lr=1e-2)
loss_history = []

# Clear old gradient log before a fresh run
open('results/comparison_shepard/gradient_log.txt', 'w').close()

# --- Optimization loop: target is the ORIGINAL image, not the degraded one ---
for t in range(iters):
    optimizer.zero_grad()
    positions = positions_n * torch.tensor([canvas_width, canvas_height])
    img = pydiffvg.ShepardRenderFunction.apply(positions, colors, q, canvas_width, canvas_height)
    loss = (img - original).pow(2).sum()

    # Repulsion: move control points that are too close to each other
    positions_px = positions_n * torch.tensor([canvas_width, canvas_height])
    diff = positions_px.unsqueeze(0) - positions_px.unsqueeze(1)  # (N, N, 2)
    dist_sq = diff.pow(2).sum(dim=2) + 0.001
    eye_mask = 1 - torch.eye(N)
    repulsion = (1.0 / dist_sq * eye_mask).sum()
    loss = loss + 0.01 * repulsion

    loss_history.append(loss.item())
    loss.backward()

    pos_grad_norms = positions_n.grad.norm(dim=1)
    color_grad_norms = colors.grad.norm(dim=1)
    print('iter', t, 'loss', loss.item())
    print('position grad — min/max/mean:',
      pos_grad_norms.min().item(), pos_grad_norms.max().item(), pos_grad_norms.mean().item())
    print('color grad — min/max/mean:',
      color_grad_norms.min().item(), color_grad_norms.max().item(), color_grad_norms.mean().item())
    with open('results/comparison_shepard/gradient_log.txt', 'a') as f:
        f.write(f'iter {t}\n')
        f.write(f'  position grad norms: {pos_grad_norms.detach().numpy().tolist()}\n')
        f.write(f'  color grad norms: {color_grad_norms.detach().numpy().tolist()}\n')

    optimizer.step()
    with torch.no_grad():
        positions_n.clamp_(0.0, 1.0)
        colors.clamp_(0.0, 1.0)
    pydiffvg.imwrite(img.clamp(0, 1).cpu(), 'results/comparison_shepard/iter_{}.png'.format(t), gamma=1.0)

print(f'final loss: {loss.item():.4f}')

# Save final loss number for later cross-script comparison
with open('results/comparison_shepard/final_loss.txt', 'w') as f:
    f.write(str(loss.item()))

# --------------------------------------
# Plot loss convergence over iterations.
# --------------------------------------
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(loss_history)
ax.set_xlabel('Iteration')
ax.set_ylabel('Loss')
ax.set_title(f'Convergence (N={N}, q={q}, init=degraded avg colour)')
plt.savefig('results/comparison_shepard/loss_curve.png', bbox_inches='tight', dpi=150)
plt.close(fig)
print('saved loss_curve.png')

# Render the final result.
final = pydiffvg.ShepardRenderFunction.apply(
    positions_n * torch.tensor([canvas_width, canvas_height]), colors, q, canvas_width, canvas_height)
pydiffvg.imwrite(final.clamp(0, 1).cpu(), 'results/comparison_shepard/final.png', gamma=1.0)

# -------------------------------------------------------------------
# Visualization: overlay control point locations on the final render.
# -------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 8))
display_img = final.detach().clamp(0, 1).cpu().numpy()
ax.imshow(display_img)
pos_np = (positions_n.detach() * torch.tensor([canvas_width, canvas_height])).cpu().numpy()
ax.scatter(pos_np[:, 0], pos_np[:, 1], c='red', s=15, edgecolors='white', linewidths=0.5)
for idx, (x, y) in enumerate(pos_np):
    ax.annotate(str(idx), (x, y), color='yellow', fontsize=8,
                xytext=(3, 3), textcoords='offset points')
ax.set_xlim(0, canvas_width)
ax.set_ylim(canvas_height, 0)
fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
ax.axis('off')
plt.savefig('results/comparison_shepard/final_labeled.png', bbox_inches='tight', pad_inches=0, dpi=150)
plt.close(fig)
print('saved final_labeled.png')

# -------------------------------------------------------------------
# Per-pixel error heatmap (against the ORIGINAL)
# -------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 6))
original_np = original.cpu().numpy()
final_np = final.detach().clamp(0, 1).cpu().numpy()
error_map = ((original_np - final_np) ** 2).mean(axis=2)
im = ax.imshow(error_map, cmap='inferno')
ax.axis('off')
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
plt.savefig('results/comparison_shepard/error_heatmap.png', bbox_inches='tight', dpi=150)
plt.close(fig)
print('saved error_heatmap.png')

# -------------------------------------------------------------------
# Comparison: degraded (init source) | original (target) | rendered | error heatmap
# -------------------------------------------------------------------
fig, axes = plt.subplots(1, 4, figsize=(24, 6))
axes[0].imshow(degraded_np)
axes[0].set_title('Degraded (init source)')
axes[0].axis('off')
axes[1].imshow(original_np)
axes[1].set_title('Original (target)')
axes[1].axis('off')
axes[2].imshow(final_np)
axes[2].set_title('Reconstructed (Shepard)')
axes[2].axis('off')
im = axes[3].imshow(error_map, cmap='inferno')
axes[3].set_title('Error heatmap')
axes[3].axis('off')
fig.colorbar(im, ax=axes[3], fraction=0.046, pad=0.04)
plt.savefig('results/comparison_shepard/all_comparison.png', bbox_inches='tight', dpi=150)
plt.close(fig)
print('saved all_comparison.png')

# Convert the intermediate renderings to a video.
from subprocess import call
call(["ffmpeg", "-framerate", "24", "-i",
    "results/comparison_shepard/iter_%d.png", "-vb", "20M",
    "results/comparison_shepard/out.mp4"])