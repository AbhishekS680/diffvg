# comparison_wendland.py
# Residual reconstruction: ellipses fit the signed error between original
# and degraded, then get added on top of the degraded photo.
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

os.makedirs('results/comparison_wendland', exist_ok=True)
os.makedirs('results/comparison_wendland/error_only', exist_ok=True)
os.makedirs('results/comparison_wendland/combined', exist_ok=True)

N = 100  # Number of ellipses
iters = 250

pydiffvg.set_use_gpu(torch.cuda.is_available())

# --- Load images ---
original = torch.from_numpy(skimage.io.imread('imgs/arch.jpg')).to(torch.float32) / 255.0
original = original[:, :, :3]
canvas_height, canvas_width = original.shape[0], original.shape[1]
pydiffvg.imwrite(original.cpu(), 'results/comparison_wendland/target_original.png', gamma=1.0)
print('original shape:', original.shape)

degraded_np = skimage.io.imread('imgs/arch_blurry.jpg').astype(np.float32) / 255.0
degraded_np = degraded_np[:, :, :3]
degraded = torch.from_numpy(degraded_np)
pydiffvg.imwrite(degraded.cpu(), 'results/comparison_wendland/init_source_degraded.png', gamma=1.0)

assert degraded_np.shape[0] == canvas_height and degraded_np.shape[1] == canvas_width, \
    'Degraded and original images must be the same size'

# Signed residual, not squared
error_image = original - degraded
pydiffvg.imwrite((error_image * 0.5 + 0.5).clamp(0, 1).cpu(),
                  'results/comparison_wendland/error_image_visualized.png', gamma=1.0)

# Init: random positions/shape, colors near zero (additive correction)
positions_n = (torch.rand(N, 2)).clone().requires_grad_(True)
colors = (torch.zeros(N, 3) + torch.rand(N, 3) * 0.05 - 0.025).clone().requires_grad_(True)

log_a = torch.full((N,), torch.log(torch.tensor(0.15))).clone().requires_grad_(True)
log_b = torch.full((N,), torch.log(torch.tensor(0.15))).clone().requires_grad_(True)
theta = torch.zeros(N).clone().requires_grad_(True)

optimizer = torch.optim.Adam([positions_n, colors, log_a, log_b, theta], lr=1e-2)
loss_history = []
diffvg.reset_ellipse_wendland_timing()

# --- Optimization loop: fit ellipses to the error image ---
for t in range(iters):
    optimizer.zero_grad()
    positions_px = positions_n * torch.tensor([canvas_width, canvas_height])
    a_px = torch.exp(log_a) * canvas_width
    b_px = torch.exp(log_b) * canvas_width
    img = pydiffvg.EllipseWendlandRenderFunction.apply(
        positions_px, colors, a_px, b_px, theta, canvas_width, canvas_height)
    loss = (img - error_image).pow(2).sum()
    loss_history.append(loss.item())
    loss.backward()

    print('iter', t, 'loss', loss.item())
    a_current = torch.exp(log_a.detach())
    b_current = torch.exp(log_b.detach())
    print('a range:', a_current.min().item(), '-', a_current.max().item())
    print('b range:', b_current.min().item(), '-', b_current.max().item())

    optimizer.step()

    if t == iters - 2:
        second_last_positions_px = (positions_n.detach() * torch.tensor([canvas_width, canvas_height])).clone().numpy()

    with torch.no_grad():
        positions_n.clamp_(0.0, 1.0)
        colors.clamp_(-1.0, 1.0)
        log_a.clamp_(torch.log(torch.tensor(0.01)), torch.log(torch.tensor(1.0)))
        log_b.clamp_(torch.log(torch.tensor(0.01)), torch.log(torch.tensor(1.0)))

    # Raw ellipse output only, no degraded photo
    raw_error_preview = (img.detach() * 0.5 + 0.5).clamp(0, 1)
    pydiffvg.imwrite(raw_error_preview.cpu(), 'results/comparison_wendland/error_only/iter_{}.png'.format(t), gamma=1.0)

    # Combined: degraded photo + ellipse correction
    combined_preview = (degraded + img.detach()).clamp(0, 1)
    pydiffvg.imwrite(combined_preview.cpu(), 'results/comparison_wendland/combined/iter_{}.png'.format(t), gamma=1.0)

print(f'final loss: {loss.item():.4f}')

diffvg.print_ellipse_wendland_timing()
fwd_ms, fwd_calls, bwd_ms, bwd_calls = diffvg.get_ellipse_wendland_timing()
with open('results/comparison_wendland/timing.txt', 'w') as f:
    f.write(f"render_ellipse_wendland timing (N={N}, {iters} iters, {canvas_width}x{canvas_height})\n")
    f.write(f"Forward:  {fwd_ms:.3f} ms total, {fwd_calls} pixel-calls, {fwd_ms/fwd_calls:.6f} ms/pixel\n")
    f.write(f"Backward: {bwd_ms:.3f} ms total, {bwd_calls} pixel-calls, {bwd_ms/bwd_calls:.6f} ms/pixel\n")

with open('results/comparison_wendland/final_loss.txt', 'w') as f:
    f.write(str(loss.item()))

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(loss_history)
ax.set_xlabel('Iteration')
ax.set_ylabel('Loss')
ax.set_title(f'Convergence (N={N}, fitting residual error)')
plt.savefig('results/comparison_wendland/loss_curve.png', bbox_inches='tight', dpi=150)
plt.close(fig)
print('saved loss_curve.png')

# --- Final render: raw error, combined result, both saved ---
positions_px = positions_n * torch.tensor([canvas_width, canvas_height])
a_px = torch.exp(log_a) * canvas_width
b_px = torch.exp(log_b) * canvas_width
reconstructed_error = pydiffvg.EllipseWendlandRenderFunction.apply(
    positions_px, colors, a_px, b_px, theta, canvas_width, canvas_height)

pydiffvg.imwrite((reconstructed_error.detach() * 0.5 + 0.5).clamp(0, 1).cpu(),
                  'results/comparison_wendland/reconstructed_error_only.png', gamma=1.0)

final = (degraded + reconstructed_error).clamp(0, 1)
pydiffvg.imwrite(final.detach().cpu(), 'results/comparison_wendland/final.png', gamma=1.0)

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
plt.savefig('results/comparison_wendland/final_labeled.png', bbox_inches='tight', pad_inches=0, dpi=150)
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
plt.savefig('results/comparison_wendland/movement_quiver.png', bbox_inches='tight', dpi=150)
plt.close(fig)
print('saved movement_quiver.png')

# Error heatmap against the original
fig, ax = plt.subplots(figsize=(8, 6))
original_np = original.cpu().numpy()
error_map = ((original_np - final_np) ** 2).mean(axis=2)
im = ax.imshow(error_map, cmap='inferno')
ax.axis('off')
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
plt.savefig('results/comparison_wendland/error_heatmap.png', bbox_inches='tight', dpi=150)
plt.close(fig)
print('saved error_heatmap.png')

# Comparison grid
fig, axes = plt.subplots(1, 4, figsize=(24, 6))
axes[0].imshow(degraded_np)
axes[0].set_title('Degraded (base)')
axes[0].axis('off')
axes[1].imshow(original_np)
axes[1].set_title('Original (target)')
axes[1].axis('off')
axes[2].imshow(final_np)
axes[2].set_title('Degraded + reconstructed error')
axes[2].axis('off')
im = axes[3].imshow(error_map, cmap='inferno')
axes[3].set_title('Error heatmap')
axes[3].axis('off')
fig.colorbar(im, ax=axes[3], fraction=0.046, pad=0.04)
plt.savefig('results/comparison_wendland/all_comparison.png', bbox_inches='tight', dpi=150)
plt.close(fig)
print('saved all_comparison.png')

from subprocess import call
call(["ffmpeg", "-framerate", "24", "-i",
    "results/comparison_wendland/error_only/iter_%d.png", "-vb", "20M",
    "results/comparison_wendland/error_only.mp4"])
call(["ffmpeg", "-framerate", "24", "-i",
    "results/comparison_wendland/combined/iter_%d.png", "-vb", "20M",
    "results/comparison_wendland/combined.mp4"])