# wendland_rendering.py
# Wendland C2 kernel field renderer, pure PyTorch autodiff

import pydiffvg
import torch

import skimage.io

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

N = 150 # Number of control points
iters = 100

# Use GPU if available
pydiffvg.set_use_gpu(torch.cuda.is_available())

target = torch.from_numpy(skimage.io.imread('imgs/fruit_basket.png')).to(torch.float32) / 255.0
target = target[:, :, :3]  # keep RGB only
canvas_height, canvas_width = target.shape[0], target.shape[1]
pydiffvg.imwrite(target.cpu(), 'results/wendland_rendering/target.png', gamma=1.0)
print('target shape:', target.shape) # used to confirm if image loaded has expected (H, W, 3) shape

# Initialize N control points and colors randomly
positions_n = (torch.rand(N, 2)).clone().requires_grad_(True) # normalized [0,1]
colors      = (torch.rand(N, 3)).clone().requires_grad_(True)
radius       = torch.full((N,), torch.log(torch.tensor(0.15))).clone().requires_grad_(True)
optimizer = torch.optim.Adam([positions_n, colors, radius], lr=1e-2)
loss_history = []

def wendland_render(positions_n, colors, log_r, canvas_width, canvas_height):
    ys, xs = torch.meshgrid(
        torch.linspace(0, 1, canvas_height),
        torch.linspace(0, 1, canvas_width),
        indexing='ij'
    )
    coords = torch.stack([xs, ys], dim=-1)               # (H, W, 2)

    r = torch.exp(log_r)                                  # (N,)
    diff = coords.unsqueeze(0) - positions_n.view(N, 1, 1, 2)
    dist = torch.norm(diff, dim=-1)                        # (N, H, W)
    t = dist / r.view(N, 1, 1)

    inside = t < 1
    w = torch.where(inside, (1 - t).clamp(min=0)**4 * (4 * t + 1), torch.zeros_like(t))

    w_sum = w.sum(dim=0) + 1e-8
    color_sum = (w.unsqueeze(-1) * colors.view(N, 1, 1, 3)).sum(dim=0)
    return color_sum / w_sum.unsqueeze(-1)

# Run Adam iterations.
for t in range(iters):
    optimizer.zero_grad()
    img = wendland_render(positions_n, colors, radius, canvas_width, canvas_height)
    loss = (img - target).pow(2).sum() # how wrong is the current render
    loss_history.append(loss.item())
    loss.backward() # backward → PyTorch autodiff computes .grad for positions, colors, radius
    
    print('iter', t, 'loss', loss.item())
    print('positions.grad norm:', positions_n.grad.norm().item())
    r_current = torch.exp(radius.detach())
    print('radius range:', r_current.min().item(), '-', r_current.max().item())
    optimizer.step() # Adam reads .grad → moves positions_n and colors

    # Helps the optimized parameters stay inside the bounds[0,1] after each optimizer.step()
    with torch.no_grad():
        positions_n.clamp_(0.0, 1.0)
        colors.clamp_(0.0, 1.0)
        radius.clamp_(torch.log(torch.tensor(0.01)), torch.log(torch.tensor(1.0)))

    pydiffvg.imwrite(img.clamp(0, 1).cpu(), 'results/wendland_rendering/iter_{}.png'.format(t), gamma=1.0)

print(f'final loss: {loss.item():.4f}')

# --------------------------------------
# Plot loss convergence over iterations.
# --------------------------------------
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(loss_history)
ax.set_xlabel('Iteration')
ax.set_ylabel('Loss')
ax.set_title(f'Convergence (N={N})')
plt.savefig('results/wendland_rendering/loss_curve.png', bbox_inches='tight', dpi=150)
plt.close(fig)
print('saved loss_curve.png')

# Render the final result.
final = wendland_render(positions_n, colors, radius, canvas_width, canvas_height)
pydiffvg.imwrite(final.clamp(0, 1).cpu(), 'results/wendland_rendering/final.png', gamma=1.0)

# -------------------------------------------------------------------
# Visualization: overlay control point locations on the final render.
# -------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 8)) # Creates a matplotlib figure to draw on
display_img = final.detach().clamp(0, 1).cpu().numpy() # Gets the final rendered image from PyTorch
ax.imshow(display_img)

# Gets the pixel coordinates of the control points and plots small red dots at each of those coordinates
pos_np = (positions_n.detach() * torch.tensor([canvas_width, canvas_height])).cpu().numpy()
ax.scatter(pos_np[:, 0], pos_np[:, 1], c='red', s=15, edgecolors='white', linewidths=0.5)

ax.set_xlim(0, canvas_width)
ax.set_ylim(canvas_height, 0)
fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

ax.axis('off') # Hides the plot's axis lines, ticks, and labels
plt.savefig('results/wendland_rendering/final_labeled.png', bbox_inches='tight', pad_inches=0, dpi=150)
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
plt.savefig('results/wendland_rendering/error_heatmap.png', bbox_inches='tight', dpi=150)
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

plt.savefig('results/wendland_rendering/all_comparison.png', bbox_inches='tight', dpi=150)
plt.close(fig)
print('saved all_comparison.png')

# Convert the intermediate renderings to a video.
from subprocess import call
call(["ffmpeg", "-framerate", "24", "-i",
    "results/wendland_rendering/iter_%d.png", "-vb", "20M",
    "results/wendland_rendering/out.mp4"])