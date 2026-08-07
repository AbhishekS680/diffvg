# ellipse_diffvg.py
# Baseline comparison: standard diffvg Ellipse shapes (hard-edged, axis-aligned,
# Monte Carlo edge-sampled gradients) fit to the same target as the Wendland
# renderer, for a fair fidelity/speed comparison.

import pydiffvg
import torch
import skimage.io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import numpy as np
from matplotlib.patches import Ellipse

os.makedirs('results/ellipse_diffvg', exist_ok=True)
N = 1000
iters = 500

pydiffvg.set_use_gpu(torch.cuda.is_available())

target = torch.from_numpy(skimage.io.imread('imgs/fruit_basket.png')).to(torch.float32) / 255.0
target = target[:, :, :3]
canvas_height, canvas_width = target.shape[0], target.shape[1]
pydiffvg.imwrite(target.cpu(), 'results/ellipse_diffvg/target.png', gamma=1.0)

# Same init convention as the Wendland script: normalized position [0,1],
# log-space semi-axes so they stay positive, start as circles.
positions_n = torch.rand(N, 2).clone().requires_grad_(True)
log_a = torch.full((N,), torch.log(torch.tensor(0.05))).clone().requires_grad_(True)
log_b = torch.full((N,), torch.log(torch.tensor(0.05))).clone().requires_grad_(True)

# RGB is learnable; alpha is fixed at 1.0 (not in the optimizer) so these
# stay genuinely hard-edged/opaque, matching the "baseline" framing --
# letting alpha drift would make this a soft-blend comparison instead.
colors_rgb = torch.rand(N, 3).clone().requires_grad_(True)
alpha_fixed = torch.ones(N, 1)
optimizer = torch.optim.Adam([positions_n, log_a, log_b, colors_rgb], lr=1e-2)
loss_history = []

for t in range(iters):
    optimizer.zero_grad()
    positions_px = positions_n * torch.tensor([canvas_width, canvas_height])
    a_px = torch.exp(log_a) * canvas_width
    b_px = torch.exp(log_b) * canvas_width
    colors = torch.cat([colors_rgb, alpha_fixed], dim=1)  # rebuild RGBA each iter
    shapes = []
    shape_groups = []
    for i in range(N):
        ellipse = pydiffvg.Ellipse(radius=torch.stack([a_px[i], b_px[i]]),
                                   center=positions_px[i])
        shapes.append(ellipse)
        shape_groups.append(pydiffvg.ShapeGroup(
            shape_ids=torch.tensor([i]),
            fill_color=colors[i]))
    scene_args = pydiffvg.RenderFunction.serialize_scene(
        canvas_width, canvas_height, shapes, shape_groups)
    img = pydiffvg.RenderFunction.apply(canvas_width, canvas_height, 2, 2, t, None, *scene_args)
    img = img[:, :, :3]  # drop alpha channel for loss against RGB target
    loss = (img - target).pow(2).sum()
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
        colors_rgb.clamp_(0.0, 1.0)
        log_a.clamp_(torch.log(torch.tensor(0.01)), torch.log(torch.tensor(1.0)))
        log_b.clamp_(torch.log(torch.tensor(0.01)), torch.log(torch.tensor(1.0)))
    pydiffvg.imwrite(img.clamp(0, 1).detach().cpu(), 'results/ellipse_diffvg/iter_{}.png'.format(t), gamma=1.0)
print(f'final loss: {loss.item():.4f}')

with open('results/ellipse_diffvg/final_loss.txt', 'w') as f:
    f.write(str(loss.item()))

# -------------------------------------------------------------------
# Loss curve
# -------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(loss_history)
ax.set_xlabel('Iteration')
ax.set_ylabel('Loss')
ax.set_title(f'Convergence (N={N}, diffvg baseline)')
plt.savefig('results/ellipse_diffvg/loss_curve.png', bbox_inches='tight', dpi=150)
plt.close(fig)
print('saved loss_curve.png')

# --- Final render ---
positions_px = positions_n * torch.tensor([canvas_width, canvas_height])
a_px = torch.exp(log_a) * canvas_width
b_px = torch.exp(log_b) * canvas_width
colors = torch.cat([colors_rgb, alpha_fixed], dim=1)
shapes = []
shape_groups = []
for i in range(N):
    ellipse = pydiffvg.Ellipse(radius=torch.stack([a_px[i], b_px[i]]),
                               center=positions_px[i])
    shapes.append(ellipse)
    shape_groups.append(pydiffvg.ShapeGroup(
        shape_ids=torch.tensor([i]),
        fill_color=colors[i]))
scene_args = pydiffvg.RenderFunction.serialize_scene(
    canvas_width, canvas_height, shapes, shape_groups)
final = pydiffvg.RenderFunction.apply(canvas_width, canvas_height, 2, 2, iters, None, *scene_args)
final = final[:, :, :3]
pydiffvg.imwrite(final.clamp(0, 1).detach().cpu(), 'results/ellipse_diffvg/final.png', gamma=1.0)

# -------------------------------------------------------------------
# Visualization: overlay control point locations + edges on final render.
# No theta here — standard diffvg Ellipse is axis-aligned only, so no
# rotation to plot (angle=0 for every patch).
# -------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 8))
display_img = final.detach().clamp(0, 1).cpu().numpy()
ax.imshow(display_img)
pos_np = positions_px.detach().cpu().numpy()
a_np = a_px.detach().cpu().numpy()
b_np = b_px.detach().cpu().numpy()
ax.scatter(pos_np[:, 0], pos_np[:, 1], c='red', s=15, edgecolors='white', linewidths=0.5)
for idx, (x, y) in enumerate(pos_np):
    ax.add_patch(Ellipse((x, y), width=2*a_np[idx], height=2*b_np[idx],
                          angle=0,
                          facecolor='none', edgecolor='lime', linewidth=0.8))
    ax.annotate(str(idx), (x, y), color='yellow', fontsize=8,
                xytext=(3, 3), textcoords='offset points')
ax.set_xlim(0, canvas_width)
ax.set_ylim(canvas_height, 0)
fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
ax.axis('off')
plt.savefig('results/ellipse_diffvg/final_labeled.png', bbox_inches='tight', pad_inches=0, dpi=150)
plt.close(fig)
print('saved final_labeled.png')
final_np = final.detach().clamp(0, 1).cpu().numpy()

# -------------------------------------------------------------------
# Quiver plot: direction each point moved, second-to-last -> final
# -------------------------------------------------------------------
final_positions_px = positions_px.detach().numpy()
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
plt.savefig('results/ellipse_diffvg/movement_quiver.png', bbox_inches='tight', dpi=150)
plt.close(fig)
print('saved movement_quiver.png')

# -------------------------------------------------------------------
# Per-pixel error heatmap
# -------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 6))
target_np = target.cpu().numpy()
error_map = ((target_np - final_np) ** 2).mean(axis=2)
im = ax.imshow(error_map, cmap='inferno')
ax.axis('off')
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
plt.savefig('results/ellipse_diffvg/error_heatmap.png', bbox_inches='tight', dpi=150)
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
axes[1].set_title('Rendered (diffvg baseline)')
axes[1].axis('off')
im = axes[2].imshow(error_map, cmap='inferno')
axes[2].set_title('Error heatmap')
axes[2].axis('off')
fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
plt.savefig('results/ellipse_diffvg/all_comparison.png', bbox_inches='tight', dpi=150)
plt.close(fig)
print('saved all_comparison.png')

# Convert the intermediate renderings to a video.
from subprocess import call
call(["ffmpeg", "-framerate", "24", "-i",
    "results/ellipse_diffvg/iter_%d.png", "-vb", "20M",
    "results/ellipse_diffvg/out.mp4"])