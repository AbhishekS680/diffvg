# comparison_wendland.py
# Ellipse Wendland reconstruction initialized from a degraded photo,
# but optimized against the original (clean) target image.
import pydiffvg
import torch
import skimage.io
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

os.makedirs('results/comparison_wendland', exist_ok=True)

N = 100 # Number of ellipses
iters = 250

pydiffvg.set_use_gpu(torch.cuda.is_available())

# --- Load images ---
original = torch.from_numpy(skimage.io.imread('imgs/arch.jpg')).to(torch.float32) / 255.0
original = original[:, :, :3]  # keep RGB only
canvas_height, canvas_width = original.shape[0], original.shape[1]
pydiffvg.imwrite(original.cpu(), 'results/comparison_wendland/target_original.png', gamma=1.0)
print('original shape:', original.shape)

degraded_np = skimage.io.imread('imgs/arch_blurry.jpg').astype(np.float32) / 255.0
degraded_np = degraded_np[:, :, :3]
pydiffvg.imwrite(torch.from_numpy(degraded_np), 'results/comparison_wendland/init_source_degraded.png', gamma=1.0)
assert degraded_np.shape[0] == canvas_height and degraded_np.shape[1] == canvas_width, \
    'Degraded and original images must be the same size'

# --- Initialize: random positions/shape, colors = degraded photo's average color ---
background_color = degraded_np.reshape(-1, 3).mean(axis=0)  # single RGB value, head start

positions_n = (torch.rand(N, 2)).clone().requires_grad_(True)  # normalized [0,1]
colors = torch.tensor(background_color, dtype=torch.float32).unsqueeze(0).repeat(N, 1)
colors = (colors + torch.rand_like(colors) * 0.05).clamp(0, 1).clone().requires_grad_(True)

# Ellipses start as small circles, same convention as ellipse_wendland_rendering.py
log_a = torch.full((N,), torch.log(torch.tensor(0.15))).clone().requires_grad_(True)
log_b = torch.full((N,), torch.log(torch.tensor(0.15))).clone().requires_grad_(True)
theta = torch.zeros(N).clone().requires_grad_(True)

optimizer = torch.optim.Adam([positions_n, colors, log_a, log_b, theta], lr=1e-2)
loss_history = []

# --- Optimization loop: target is the ORIGINAL image, not the degraded one ---
for t in range(iters):
    optimizer.zero_grad()
    positions_px = positions_n * torch.tensor([canvas_width, canvas_height])
    a_px = torch.exp(log_a) * canvas_width
    b_px = torch.exp(log_b) * canvas_width
    img = pydiffvg.EllipseWendlandRenderFunction.apply(
        positions_px, colors, a_px, b_px, theta, canvas_width, canvas_height)
    loss = (img - original).pow(2).sum()
    loss_history.append(loss.item())
    loss.backward()

    print('iter', t, 'loss', loss.item())
    a_current = torch.exp(log_a.detach())
    b_current = torch.exp(log_b.detach())
    print('a range:', a_current.min().item(), '-', a_current.max().item())
    print('b range:', b_current.min().item(), '-', b_current.max().item())

    optimizer.step()
    with torch.no_grad():
        positions_n.clamp_(0.0, 1.0)
        colors.clamp_(0.0, 1.0)
        log_a.clamp_(torch.log(torch.tensor(0.01)), torch.log(torch.tensor(1.0)))
        log_b.clamp_(torch.log(torch.tensor(0.01)), torch.log(torch.tensor(1.0)))
        # theta is unclamped, rotation naturally wraps
    pydiffvg.imwrite(img.clamp(0, 1).cpu(), 'results/comparison_wendland/iter_{}.png'.format(t), gamma=1.0)

print(f'final loss: {loss.item():.4f}')

# Save final loss number for later cross-script comparison
with open('results/comparison_wendland/final_loss.txt', 'w') as f:
    f.write(str(loss.item()))

# --------------------------------------
# Plot loss convergence over iterations.
# --------------------------------------
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(loss_history)
ax.set_xlabel('Iteration')
ax.set_ylabel('Loss')
ax.set_title(f'Convergence (N={N}, init=degraded avg colour)')
plt.savefig('results/comparison_wendland/loss_curve.png', bbox_inches='tight', dpi=150)
plt.close(fig)
print('saved loss_curve.png')

# Render the final result.
positions_px = positions_n * torch.tensor([canvas_width, canvas_height])
a_px = torch.exp(log_a) * canvas_width
b_px = torch.exp(log_b) * canvas_width
final = pydiffvg.EllipseWendlandRenderFunction.apply(
    positions_px, colors, a_px, b_px, theta, canvas_width, canvas_height)
pydiffvg.imwrite(final.clamp(0, 1).cpu(), 'results/comparison_wendland/final.png', gamma=1.0)

# -------------------------------------------------------------------
# Visualization: overlay control point locations on the final render.
# -------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 8))
display_img = final.detach().clamp(0, 1).cpu().numpy()
ax.imshow(display_img)
pos_np = (positions_n.detach() * torch.tensor([canvas_width, canvas_height])).cpu().numpy()
ax.scatter(pos_np[:, 0], pos_np[:, 1], c='red', s=15, edgecolors='white', linewidths=0.5)
ax.set_xlim(0, canvas_width)
ax.set_ylim(canvas_height, 0)
fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
ax.axis('off')
plt.savefig('results/comparison_wendland/final_labeled.png', bbox_inches='tight', pad_inches=0, dpi=150)
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
plt.savefig('results/comparison_wendland/error_heatmap.png', bbox_inches='tight', dpi=150)
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
axes[2].set_title('Reconstructed (Ellipse Wendland)')
axes[2].axis('off')
im = axes[3].imshow(error_map, cmap='inferno')
axes[3].set_title('Error heatmap')
axes[3].axis('off')
fig.colorbar(im, ax=axes[3], fraction=0.046, pad=0.04)
plt.savefig('results/comparison_wendland/all_comparison.png', bbox_inches='tight', dpi=150)
plt.close(fig)
print('saved all_comparison.png')

# Convert the intermediate renderings to a video.
from subprocess import call
call(["ffmpeg", "-framerate", "24", "-i",
    "results/comparison_wendland/iter_%d.png", "-vb", "20M",
    "results/comparison_wendland/out.mp4"])