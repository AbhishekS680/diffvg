# shepard_rendering.py
# Implemented using diffvg's ShepardField C++ renderer
import pydiffvg
import torch
import skimage.io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import diffvg

N = 1000 # Number of control points
q = 3.0 # Controls how sharply the falloff happens for each control point
iters = 200

# --- Focus phase ---
# After the normal training loop finishes, run a second phase of
# iterations on the SAME control points, but reweight the per-pixel
# loss using the error heatmap from pass one -- pixels that were
# reconstructed badly get more gradient pull, pixels that were already
# good get left mostly alone. Nothing is frozen, nothing new is added,
# the existing points just get pushed harder toward fixing their own
# mistakes.
FOCUS_ITERS = 100
FOCUS_WEIGHT_SCALE = 5.0  # how much extra weight the worst pixels get, relative to the best

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
diffvg.reset_shepard_timing()

# Run Adam iterations.
for t in range(iters):
    optimizer.zero_grad()
    positions = positions_n * torch.tensor([canvas_width, canvas_height])
    img = pydiffvg.ShepardRenderFunction.apply(positions, colors, q, canvas_width, canvas_height) # forward → C++ render_shepard
    loss = (img - target).pow(2).sum() # how wrong is the current render
    # Repulsion: move control points that are too close to each other
    # positions_px = positions_n * torch.tensor([canvas_width, canvas_height])
    # diff = positions_px.unsqueeze(0) - positions_px.unsqueeze(1)  # (N, N, 2)
    # dist_sq = diff.pow(2).sum(dim=2) + 0.001  # (N, N) squared distances, +1 prevents division by zero
    # eye_mask = 1 - torch.eye(N)
    # repulsion = (1.0 / dist_sq * eye_mask).sum()
    # loss = loss + 0.01 * repulsion
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

    if t == iters - 2:
        second_last_positions_px = (positions_n.detach() * torch.tensor([canvas_width, canvas_height])).clone().numpy()

    # Helps the optimized parameters stay inside the bounds[0,1] after each optimizer.step()
    with torch.no_grad():
        positions_n.clamp_(0.0, 1.0)
        colors.clamp_(0.0, 1.0)

    pydiffvg.imwrite(img.clamp(0, 1).cpu(), 'results/shepard_rendering/iter_{}.png'.format(t), gamma=1.0)

print(f'final loss: {loss.item():.4f}')
diffvg.print_shepard_timing()
fwd_ms, fwd_calls, bwd_ms, bwd_calls = diffvg.get_shepard_timing()
with open('results/shepard_rendering/timing.txt', 'w') as f:
    f.write(f"render_shepard timing (N={N}, {iters} iters, {canvas_width}x{canvas_height})\n")
    f.write(f"Forward:  {fwd_ms:.3f} ms total, {fwd_calls} pixel-calls, {fwd_ms/fwd_calls:.6f} ms/pixel\n")
    f.write(f"Backward: {bwd_ms:.3f} ms total, {bwd_calls} pixel-calls, {bwd_ms/bwd_calls:.6f} ms/pixel\n")

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

# Render the final result (pass one).
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

final_np = final.detach().clamp(0, 1).cpu().numpy()

# -------------------------------------------------------------------
# Quiver plot: direction each point moved, initial -> final
# -------------------------------------------------------------------
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
plt.savefig('results/shepard_rendering/movement_quiver.png', bbox_inches='tight', dpi=150)
plt.close(fig)
print('saved movement_quiver.png')

# -------------------------------------------------------------------
# Per-pixel error heatmap (pass one, pre-focus)
# -------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 6))
target_np = target.cpu().numpy()
# Per-pixel error: mean squared difference across RGB channels
error_map = ((target_np - final_np) ** 2).mean(axis=2)
im = ax.imshow(error_map, cmap='inferno')
ax.axis('off')
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
plt.savefig('results/shepard_rendering/error_heatmap.png', bbox_inches='tight', dpi=150)
plt.close(fig)
print('saved error_heatmap.png')

# -------------------------------------------------------------------
# Focus phase: continue training the SAME positions_n/colors, but
# reweight the loss using pass one's error heatmap. Pixels with high
# error get a bigger weight, so their gradient contribution dominates;
# pixels that were already reconstructed well get weight close to 1
# and are mostly left alone.
# -------------------------------------------------------------------
import os
weight_np = 1.0 + FOCUS_WEIGHT_SCALE * (error_map / (error_map.max() + 1e-8))
weight_map = torch.from_numpy(weight_np).to(torch.float32).unsqueeze(-1)  # (H, W, 1), broadcasts over RGB

fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(weight_np, cmap='viridis')
ax.axis('off')
ax.set_title(f'Focus weight map (scale={FOCUS_WEIGHT_SCALE}, max weight={weight_np.max():.2f})')
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
plt.savefig('results/shepard_rendering/focus_weight_map.png', bbox_inches='tight', dpi=150)
plt.close(fig)
print('saved focus_weight_map.png')

focus_loss_history = []
os.makedirs('results/shepard_rendering/focus_iters', exist_ok=True)

for t in range(FOCUS_ITERS):
    optimizer.zero_grad()
    positions = positions_n * torch.tensor([canvas_width, canvas_height])
    img = pydiffvg.ShepardRenderFunction.apply(positions, colors, q, canvas_width, canvas_height)
    focus_loss = ((img - target).pow(2) * weight_map).sum()
    focus_loss_history.append(focus_loss.item())
    focus_loss.backward()

    print('focus iter', t, 'loss', focus_loss.item())

    optimizer.step()

    with torch.no_grad():
        positions_n.clamp_(0.0, 1.0)
        colors.clamp_(0.0, 1.0)

    pydiffvg.imwrite(img.detach().clamp(0, 1).cpu(), 'results/shepard_rendering/focus_iters/iter_{}.png'.format(t), gamma=1.0)

print(f'final focus loss: {focus_loss.item():.4f}')
with open('results/shepard_rendering/focus_final_loss.txt', 'w') as f:
    f.write(str(focus_loss.item()))

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(focus_loss_history)
ax.set_xlabel('Focus iteration')
ax.set_ylabel('Weighted loss')
ax.set_title(f'Focus phase convergence (weight scale={FOCUS_WEIGHT_SCALE}, {FOCUS_ITERS} iters)')
plt.savefig('results/shepard_rendering/focus_loss_curve.png', bbox_inches='tight', dpi=150)
plt.close(fig)
print('saved focus_loss_curve.png')

# Render the final result (post-focus).
final_focused = pydiffvg.ShepardRenderFunction.apply(positions_n * torch.tensor([canvas_width, canvas_height]), colors, q, canvas_width, canvas_height)
final_focused = final_focused.clamp(0, 1)
pydiffvg.imwrite(final_focused.detach().cpu(), 'results/shepard_rendering/final_focused.png', gamma=1.0)
final_focused_np = final_focused.detach().clamp(0, 1).cpu().numpy()

# Error heatmap against target, post-focus -- same color scale as the
# pre-focus error heatmap so the two are directly comparable.
error_map_focused = ((target_np - final_focused_np) ** 2).mean(axis=2)
shared_focus_vmax = max(error_map.max(), error_map_focused.max())

fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(error_map_focused, cmap='inferno', vmin=0, vmax=shared_focus_vmax)
ax.axis('off')
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
plt.savefig('results/shepard_rendering/error_heatmap_focused.png', bbox_inches='tight', dpi=150)
plt.close(fig)
print('saved error_heatmap_focused.png')

print(f'pre-focus mean error:  {error_map.mean():.6f}')
print(f'post-focus mean error: {error_map_focused.mean():.6f}')
with open('results/shepard_rendering/focus_summary.txt', 'w') as f:
    f.write(f'pre-focus mean error:  {error_map.mean():.6f}\n')
    f.write(f'post-focus mean error: {error_map_focused.mean():.6f}\n')
    f.write(f'change: {error_map_focused.mean() - error_map.mean():.6f}\n')

# Before/after focus comparison: reconstruction | reconstruction-error | focused | focused-error
fig, axes = plt.subplots(1, 4, figsize=(24, 6))
axes[0].imshow(final_np)
axes[0].set_title('Reconstruction (pre-focus)')
axes[0].axis('off')

im0 = axes[1].imshow(error_map, cmap='inferno', vmin=0, vmax=shared_focus_vmax)
axes[1].set_title(f'Error (pre-focus), mean={error_map.mean():.5f}')
axes[1].axis('off')
fig.colorbar(im0, ax=axes[1], fraction=0.046, pad=0.04)

axes[2].imshow(final_focused_np)
axes[2].set_title(f'Reconstruction (post-focus, {FOCUS_ITERS} iters)')
axes[2].axis('off')

im1 = axes[3].imshow(error_map_focused, cmap='inferno', vmin=0, vmax=shared_focus_vmax)
axes[3].set_title(f'Error (post-focus), mean={error_map_focused.mean():.5f}')
axes[3].axis('off')
fig.colorbar(im1, ax=axes[3], fraction=0.046, pad=0.04)

plt.savefig('results/shepard_rendering/focus_comparison.png', bbox_inches='tight', dpi=150)
plt.close(fig)
print('saved focus_comparison.png')

# -------------------------------------------------------------------
# Comparison: target | rendered | error heatmap (pre-focus, unchanged
# from original script)
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