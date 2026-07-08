# shepard_rendering.py
# Implemented using diffvg's ShepardField C++ renderer

import pydiffvg
import torch

import skimage.io

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

N = 30 # Number of control points
q = 3.0 # Controls how sharply the falloff happens for each control point
iters = 100

# Use GPU if available
pydiffvg.set_use_gpu(torch.cuda.is_available())

target = torch.from_numpy(skimage.io.imread('imgs/fruit_basket.png')).to(torch.float32) / 255.0
target = target[:, :, :3]  # keep RGB only
canvas_height, canvas_width = target.shape[0], target.shape[1]
pydiffvg.imwrite(target.cpu(), 'results/shepard_rendering/target.png', gamma=1.0)
print('target shape:', target.shape) # used to confirm if image loaded has expected (H, W, 3) shape

# Initialize N control points and colors randomly
positions_n = (torch.rand(N, 2)).clone().requires_grad_(True) # normalized [0,1]
colors      = (torch.rand(N, 3)).clone().requires_grad_(True)
optimizer   = torch.optim.Adam([positions_n, colors], lr=1e-2)
loss_history = []

# Clear old gradient log before a fresh run
open('results/shepard_rendering/gradient_log.txt', 'w').close()

# Run Adam iterations.
for t in range(iters):
    optimizer.zero_grad()
    positions = positions_n * torch.tensor([canvas_width, canvas_height])
    img = pydiffvg.ShepardRenderFunction.apply(positions, colors, q, canvas_width, canvas_height) # forward → C++ render_shepard
    loss = (img - target).pow(2).sum() # how wrong is the current render

    # Repulsion: move control points that are too close to each other
    positions_px = positions_n * torch.tensor([canvas_width, canvas_height])
    diff = positions_px.unsqueeze(0) - positions_px.unsqueeze(1)  # (N, N, 2)
    dist_sq = diff.pow(2).sum(dim=2) + 0.001  # (N, N) squared distances, +1 prevents division by zero
    eye_mask = 1 - torch.eye(N)
    repulsion = (1.0 / dist_sq * eye_mask).sum()
    loss = loss + 0.01 * repulsion

    loss_history.append(loss.item())
    loss.backward() # backward → C++ fills d_positions, d_colours → deposits into .grad -> d_render_image created

    # Per-point gradient magnitudes, how hard is each control point being pulled?
    pos_grad_norms = positions_n.grad.norm(dim=1) # (N,) one value per point
    color_grad_norms = colors.grad.norm(dim=1) # (N,) one value per point

    print('iter', t, 'loss', loss.item())
    print('position grad — min/max/mean:',
      pos_grad_norms.min().item(), pos_grad_norms.max().item(), pos_grad_norms.mean().item())
    print('color grad — min/max/mean:',
      color_grad_norms.min().item(), color_grad_norms.max().item(), color_grad_norms.mean().item())

    with open('results/shepard_rendering/gradient_log.txt', 'a') as f:
        f.write(f'iter {t}\n')
        f.write(f'  position grad norms: {pos_grad_norms.detach().numpy().tolist()}\n')
        f.write(f'  color grad norms: {color_grad_norms.detach().numpy().tolist()}\n')

    optimizer.step() # Adam reads .grad → moves positions_n and colors

    # Helps the optimized parameters stay inside the bounds[0,1] after each optimizer.step()
    with torch.no_grad():
        positions_n.clamp_(0.0, 1.0)
        colors.clamp_(0.0, 1.0)

    pydiffvg.imwrite(img.clamp(0, 1).cpu(), 'results/shepard_rendering/iter_{}.png'.format(t), gamma=1.0)

print(f'final loss: {loss.item():.4f}')

# --------------------------------------
# Plot loss convergence over iterations.
# --------------------------------------
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(loss_history)
ax.set_xlabel('Iteration')
ax.set_ylabel('Loss')
ax.set_title(f'Convergence (N={N}, q={q})')
plt.savefig('results/shepard_rendering/loss_curve.png', bbox_inches='tight', dpi=150)
plt.close(fig)
print('saved loss_curve.png')

# Render the final result.
final = pydiffvg.ShepardRenderFunction.apply(positions_n * torch.tensor([canvas_width, canvas_height]), colors, q, canvas_width, canvas_height)
pydiffvg.imwrite(final.clamp(0, 1).cpu(), 'results/shepard_rendering/final.png', gamma=1.0)

# -------------------------------------------------------------------
# Visualization: overlay control point locations on the final render.
# -------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 8)) # Creates a matplotlib figure to draw on
display_img = final.detach().clamp(0, 1).cpu().numpy() # Gets the final rendered image from PyTorch
ax.imshow(display_img)

# Gets the pixel coordinates of the control points and plots small red dots at each of those coordinates
pos_np = (positions_n.detach() * torch.tensor([canvas_width, canvas_height])).cpu().numpy()
ax.scatter(pos_np[:, 0], pos_np[:, 1], c='red', s=15, edgecolors='white', linewidths=0.5)

# Label each control point with its index
for idx, (x, y) in enumerate(pos_np):
    ax.annotate(str(idx), (x, y), color='yellow', fontsize=8,
                xytext=(3, 3), textcoords='offset points')

ax.set_xlim(0, canvas_width)
ax.set_ylim(canvas_height, 0)
fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

ax.axis('off') # Hides the plot's axis lines, ticks, and labels
plt.savefig('results/shepard_rendering/final_labeled.png', bbox_inches='tight', pad_inches=0, dpi=150)
plt.close(fig) # Releases the figure from memory
print('saved final_labeled.png')

# -------------------------------------------------------------------
# Per-pixel error heatmap
# -------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 6))

target_np = target.cpu().numpy()
final_np = final.detach().clamp(0, 1).cpu().numpy()

# Per-pixel error: mean squared difference across RGB channels
error_map = ((target_np - final_np) ** 2).mean(axis=2)

im = ax.imshow(error_map, cmap='inferno')
ax.axis('off')
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
plt.savefig('results/shepard_rendering/error_heatmap.png', bbox_inches='tight', dpi=150)
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
axes[1].set_title('Rendered')
axes[1].axis('off')

im = axes[2].imshow(error_map, cmap='inferno')
axes[2].set_title('Error heatmap')
axes[2].axis('off')
fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

plt.savefig('results/shepard_rendering/all_comparison.png', bbox_inches='tight', dpi=150)
plt.close(fig)
print('saved all_comparison.png')

# Convert the intermediate renderings to a video.
from subprocess import call
call(["ffmpeg", "-framerate", "24", "-i",
    "results/shepard_rendering/iter_%d.png", "-vb", "20M",
    "results/shepard_rendering/out.mp4"])