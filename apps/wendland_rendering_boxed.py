# wendland_rendering_boxed.py
# Same as wendland_rendering.py, but uses the tile-grid accelerated
# renderer
import pydiffvg
import diffvg
import torch
import skimage.io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import numpy as np
from matplotlib.patches import Ellipse

os.makedirs('results/wendland_rendering_boxed', exist_ok=True)

N = 10000
iters = 500

# --- Semi-axis size penalty ---
# Discourages a/b (in pixels) from growing past MAX_AXIS_PX. Soft quadratic
# penalty: zero below the threshold, then grows smoothly, so Adam gets a
# real gradient pushing oversized ellipses back down instead of a hard
# clamp (no gradient) or a discontinuous jump (unstable).
MAX_AXIS_PX = 10.0
AXIS_PENALTY_WEIGHT = 1.0

# Use GPU if available
pydiffvg.set_use_gpu(torch.cuda.is_available())

target = torch.from_numpy(skimage.io.imread('imgs/fruit_basket.png')).to(torch.float32) / 255.0
target = target[:, :, :3]  # keep RGB only
canvas_height, canvas_width = target.shape[0], target.shape[1]
pydiffvg.imwrite(target.cpu(), 'results/wendland_rendering_boxed/target.png', gamma=1.0)

# Initialize N ellipses (position, color, semi-axes, rotation) randomly
positions_n = (torch.rand(N, 2)).clone().requires_grad_(True)  # normalized [0,1]
colors      = (torch.rand(N, 3)).clone().requires_grad_(True)
log_a       = torch.full((N,), torch.log(torch.tensor(0.05))).clone().requires_grad_(True)
log_b       = torch.full((N,), torch.log(torch.tensor(0.05))).clone().requires_grad_(True)
theta       = torch.zeros(N).clone().requires_grad_(True)
optimizer = torch.optim.Adam([positions_n, colors, log_a, log_b, theta], lr=1e-2)
loss_history = []

diffvg.reset_ellipse_wendland_boxed_timing()

# Clear old axis-penalty log before a fresh run
open('results/wendland_rendering_boxed/axis_penalty_log.txt', 'w').close()

# --------------------------------------
# Run Adam iterations.
# --------------------------------------
for t in range(iters):
    optimizer.zero_grad()
    positions_px = positions_n * torch.tensor([canvas_width, canvas_height])
    a_px = torch.exp(log_a) * canvas_width
    b_px = torch.exp(log_b) * canvas_width
    img = pydiffvg.EllipseWendlandBoxedRenderFunction.apply(positions_px, colors, a_px, b_px, theta, None, canvas_width, canvas_height)
    loss = (img - target).pow(2).sum()  # how wrong is the current render

    # Soft penalty on oversized semi-axes: zero contribution while a/b stay
    # under MAX_AXIS_PX, quadratic growth past it.
    axis_penalty = (torch.clamp(a_px - MAX_AXIS_PX, min=0).pow(2).sum()
                    + torch.clamp(b_px - MAX_AXIS_PX, min=0).pow(2).sum())
    loss = loss + AXIS_PENALTY_WEIGHT * axis_penalty

    loss_history.append(loss.item())
    loss.backward()  # backward -> C++ fills gradients -> deposits into .grad
    print('iter', t, 'loss', loss.item())
    a_current = torch.exp(log_a.detach())
    b_current = torch.exp(log_b.detach())
    print('a range:', a_current.min().item(), '-', a_current.max().item())
    print('b range:', b_current.min().item(), '-', b_current.max().item())
    n_over = int(((a_px.detach() > MAX_AXIS_PX) | (b_px.detach() > MAX_AXIS_PX)).sum().item())
    with open('results/wendland_rendering_boxed/axis_penalty_log.txt', 'a') as f:
        f.write(f'iter {t}: penalty={axis_penalty.item():.6f} '
                f'n_over_threshold={n_over}/{N} '
                f'a[min={a_px.detach().min().item():.3f} max={a_px.detach().max().item():.3f} '
                f'mean={a_px.detach().mean().item():.3f}] '
                f'b[min={b_px.detach().min().item():.3f} max={b_px.detach().max().item():.3f} '
                f'mean={b_px.detach().mean().item():.3f}]\n')
    optimizer.step()  # Adam reads .grad -> moves positions_n, colors, log_a, log_b, theta

    if t == iters - 2:
        second_last_positions_px = (positions_n.detach() * torch.tensor([canvas_width, canvas_height])).clone().numpy()

    # Helps the optimized parameters stay inside their bounds after each optimizer.step()
    with torch.no_grad():
        positions_n.clamp_(0.0, 1.0)
        colors.clamp_(0.0, 1.0)
        log_a.clamp_(torch.log(torch.tensor(0.01)), torch.log(torch.tensor(1.0)))
        log_b.clamp_(torch.log(torch.tensor(0.01)), torch.log(torch.tensor(1.0)))

    pydiffvg.imwrite(img.clamp(0, 1).cpu(), 'results/wendland_rendering_boxed/iter_{}.png'.format(t), gamma=1.0)

print(f'final loss: {loss.item():.4f}')
diffvg.print_ellipse_wendland_boxed_timing()
fwd_ms, fwd_calls, bwd_ms, bwd_calls = diffvg.get_ellipse_wendland_boxed_timing()
with open('results/wendland_rendering_boxed/timing.txt', 'w') as f:
    f.write(f"render_ellipse_wendland_boxed timing (N={N}, {iters} iters, {canvas_width}x{canvas_height})\n")
    f.write(f"Forward:  {fwd_ms:.3f} ms total, {fwd_calls} pixel-calls, {fwd_ms/fwd_calls:.6f} ms/pixel\n")
    f.write(f"Backward: {bwd_ms:.3f} ms total, {bwd_calls} pixel-calls, {bwd_ms/bwd_calls:.6f} ms/pixel\n")

# --------------------------------------
# Plot loss convergence over iterations.
# --------------------------------------
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(loss_history)
ax.set_xlabel('Iteration')
ax.set_ylabel('Loss')
ax.set_title(f'Convergence (N={N}, boxed/tile-grid Wendland)')
plt.savefig('results/wendland_rendering_boxed/loss_curve.png', bbox_inches='tight', dpi=150)
plt.close(fig)
print('saved loss_curve.png')

# --------------------------------------
# Render the final result.
# --------------------------------------
positions_px = positions_n * torch.tensor([canvas_width, canvas_height])
a_px = torch.exp(log_a) * canvas_width
b_px = torch.exp(log_b) * canvas_width
final = pydiffvg.EllipseWendlandBoxedRenderFunction.apply(positions_px, colors, a_px, b_px, theta, None, canvas_width, canvas_height)
pydiffvg.imwrite(final.clamp(0, 1).cpu(), 'results/wendland_rendering_boxed/final.png', gamma=1.0)

# -------------------------------------------------------------------
# Visualization: overlay ellipse boundaries and control point locations
# on the final render.
# -------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 8))
fig.patch.set_facecolor('black')
display_img = final.detach().clamp(0, 1).cpu().numpy()
ALPHA_CUTOFF = 0.02
alpha_mask = (display_img.max(axis=2) > ALPHA_CUTOFF).astype(np.float32)
rgba_img = np.dstack([display_img, alpha_mask])
ax.imshow(rgba_img)

# Binary search for the t value where the Wendland kernel drops to ALPHA_CUTOFF,
# so the drawn ellipse boundary matches where the ellipse visually fades out.
lo, hi = 0.0, 1.0
for _ in range(50):
    mid = (lo + hi) / 2
    val = (1 - mid)**4 * (4 * mid + 1)
    if val > ALPHA_CUTOFF:
        lo = mid
    else:
        hi = mid
t_boundary = mid

pos_np = (positions_n.detach() * torch.tensor([canvas_width, canvas_height])).cpu().numpy()
a_np = a_px.detach().cpu().numpy()
b_np = b_px.detach().cpu().numpy()
theta_np = theta.detach().cpu().numpy()
ax.scatter(pos_np[:, 0], pos_np[:, 1], c='red', s=15, edgecolors='white', linewidths=0.5)
# Label each control point with its index and draw its effective ellipse boundary
for idx, (x, y) in enumerate(pos_np):
    ax.add_patch(Ellipse((x, y), width=2*a_np[idx]*t_boundary, height=2*b_np[idx]*t_boundary,
                          angle=np.degrees(theta_np[idx]),
                          facecolor='none', edgecolor='lime', linewidth=0.8))
    ax.annotate(str(idx), (x, y), color='yellow', fontsize=8,
                xytext=(3, 3), textcoords='offset points')
ax.set_xlim(0, canvas_width)
ax.set_ylim(canvas_height, 0)
fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
ax.axis('off')  # Hides the plot's axis lines, ticks, and labels
plt.savefig('results/wendland_rendering_boxed/final_labeled.png', bbox_inches='tight', pad_inches=0, dpi=150,
            facecolor='black')
plt.close(fig)  # Releases the figure from memory
print('saved final_labeled.png')

final_np = final.detach().clamp(0, 1).cpu().numpy()

# -------------------------------------------------------------------
# Quiver plot: direction each point moved, second-to-last -> final
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
plt.savefig('results/wendland_rendering_boxed/movement_quiver.png', bbox_inches='tight', dpi=150)
plt.close(fig)
print('saved movement_quiver.png')

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
plt.savefig('results/wendland_rendering_boxed/error_heatmap.png', bbox_inches='tight', dpi=150)
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
axes[1].set_title('Rendered (boxed)')
axes[1].axis('off')
im = axes[2].imshow(error_map, cmap='inferno')
axes[2].set_title('Error heatmap')
axes[2].axis('off')
fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
plt.savefig('results/wendland_rendering_boxed/all_comparison.png', bbox_inches='tight', dpi=150)
plt.close(fig)
print('saved all_comparison.png')

# Convert the intermediate renderings to a video.
from subprocess import call
call(["ffmpeg", "-framerate", "24", "-i",
    "results/wendland_rendering_boxed/iter_%d.png", "-vb", "20M",
    "results/wendland_rendering_boxed/out.mp4"])