# ellipse_shepard_refine.py
# Residual refinement: ellipse Wendland corrects Shepard's leftover error
# Shepard's output is a fixed base layer
# only the ellipse correction layer is trained

import pydiffvg
import torch
import skimage.io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

os.makedirs('results/ellipse_shepard_refine', exist_ok=True)

N = 150
iters = 100

pydiffvg.set_use_gpu(torch.cuda.is_available())

target = torch.from_numpy(skimage.io.imread('imgs/fruit_basket.png')).to(torch.float32) / 255.0
target = target[:, :, :3]
canvas_height, canvas_width = target.shape[0], target.shape[1]

# Shepard's already-optimized output — fixed, not re-optimized here
shepard_base = torch.from_numpy(skimage.io.imread('results/shepard_rendering/final.png')).to(torch.float32) / 255.0
shepard_base = shepard_base[:, :, :3]

baseline_mse = ((target - shepard_base) ** 2).mean().item()
print('Shepard baseline MSE:', baseline_mse)

# Ellipse control points, initialized to represent small corrections
positions_n = (torch.rand(N, 2)).clone().requires_grad_(True)
colors      = (torch.randn(N, 3) * 0.1).clone().requires_grad_(True)   # signed correction, starts subtle
log_a       = torch.full((N,), torch.log(torch.tensor(0.15))).clone().requires_grad_(True)
log_b       = torch.full((N,), torch.log(torch.tensor(0.15))).clone().requires_grad_(True)
theta       = torch.zeros(N).clone().requires_grad_(True)

optimizer = torch.optim.Adam([positions_n, colors, log_a, log_b, theta], lr=1e-2)
loss_history = []

for t in range(iters):
    optimizer.zero_grad()
    positions_px = positions_n * torch.tensor([canvas_width, canvas_height])
    a_px = torch.exp(log_a) * canvas_width
    b_px = torch.exp(log_b) * canvas_width
    correction = pydiffvg.EllipseWendlandRenderFunction.apply(
        positions_px, colors, a_px, b_px, theta, canvas_width, canvas_height)

    combined = shepard_base + correction
    loss = (combined - target).pow(2).sum()
    loss_history.append(loss.item())
    loss.backward()

    print('iter', t, 'loss', loss.item())
    optimizer.step()

    with torch.no_grad():
        positions_n.clamp_(0.0, 1.0)
        colors.clamp_(-1.0, 1.0)   # corrections can be negative, unlike raw paint colors
        log_a.clamp_(torch.log(torch.tensor(0.01)), torch.log(torch.tensor(1.0)))
        log_b.clamp_(torch.log(torch.tensor(0.01)), torch.log(torch.tensor(1.0)))

    pydiffvg.imwrite((shepard_base + correction).clamp(0, 1).detach().cpu(),
                      'results/ellipse_shepard_refine/iter_{}.png'.format(t), gamma=1.0)

print(f'final loss: {loss.item():.4f}')

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(loss_history)
ax.set_xlabel('Iteration')
ax.set_ylabel('Loss')
ax.set_title(f'Convergence (N={N})')
plt.savefig('results/ellipse_shepard_refine/loss_curve.png', bbox_inches='tight', dpi=150)
plt.close(fig)

positions_px = positions_n * torch.tensor([canvas_width, canvas_height])
a_px = torch.exp(log_a) * canvas_width
b_px = torch.exp(log_b) * canvas_width
correction = pydiffvg.EllipseWendlandRenderFunction.apply(
    positions_px, colors, a_px, b_px, theta, canvas_width, canvas_height)
final = (shepard_base + correction).clamp(0, 1)
pydiffvg.imwrite(final.detach().cpu(), 'results/ellipse_shepard_refine/final.png', gamma=1.0)

final_mse = ((target - final) ** 2).mean().item()
print('Shepard baseline MSE:', baseline_mse)
print('Shepard + ellipse refinement MSE:', final_mse)
print('Improvement:', baseline_mse - final_mse)

target_np = target.cpu().numpy()
shepard_np = shepard_base.cpu().numpy()
final_np = final.detach().cpu().numpy()
error_map = ((target_np - final_np) ** 2).mean(axis=2)

# Standalone error heatmap
fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(error_map, cmap='inferno')
ax.set_title('Remaining error after Shepard + ellipse refinement')
ax.axis('off')
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
plt.savefig('results/ellipse_shepard_refine/error_heatmap.png', bbox_inches='tight', dpi=150)
plt.close(fig)
print('saved error_heatmap.png')

fig, axes = plt.subplots(1, 4, figsize=(22, 6))
axes[0].imshow(target_np); axes[0].set_title('Target'); axes[0].axis('off')
axes[1].imshow(shepard_np); axes[1].set_title(f'Shepard alone (MSE {baseline_mse:.5f})'); axes[1].axis('off')
axes[2].imshow(final_np); axes[2].set_title(f'Shepard + ellipse (MSE {final_mse:.5f})'); axes[2].axis('off')
im = axes[3].imshow(error_map, cmap='inferno'); axes[3].set_title('Remaining error'); axes[3].axis('off')
fig.colorbar(im, ax=axes[3], fraction=0.046, pad=0.04)
plt.savefig('results/ellipse_shepard_refine/all_comparison.png', bbox_inches='tight', dpi=150)
plt.close(fig)
print('saved all_comparison.png')

from subprocess import call
call(["ffmpeg", "-framerate", "24", "-i",
    "results/ellipse_shepard_refine/iter_%d.png", "-vb", "20M",
    "results/ellipse_shepard_refine/out.mp4"])