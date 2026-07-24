# polynomial_kernel_rendering.py
# Anisotropic ellipse renderer using a learnable global polynomial kernel instead of the fixed Wendland formula
# f(t) = a*t^4 + b*t^3 + c*t^2 + d*t + e
# a..e start at 1 and are optimized jointly with position/shape/color via EllipsePolyRenderFunction

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

os.makedirs('results/polynomial_kernel_rendering', exist_ok=True)

N = 100
iters = 250

pydiffvg.set_use_gpu(torch.cuda.is_available())

target = torch.from_numpy(skimage.io.imread('imgs/fruit_basket.png')).to(torch.float32) / 255.0
target = target[:, :, :3]
canvas_height, canvas_width = target.shape[0], target.shape[1]
pydiffvg.imwrite(target.cpu(), 'results/polynomial_kernel_rendering/target.png', gamma=1.0)

positions_n = torch.rand(N, 2).clone().requires_grad_(True) # normalized [0,1]
colors      = torch.rand(N, 3).clone().requires_grad_(True)
log_a       = torch.full((N,), torch.log(torch.tensor(0.15))).clone().requires_grad_(True)
log_b       = torch.full((N,), torch.log(torch.tensor(0.15))).clone().requires_grad_(True)
theta       = torch.zeros(N).clone().requires_grad_(True)

# Polynomial kernel coefficients: f(t) = a*t^4 + b*t^3 + c*t^2 + d*t + e
# Global: one set shared across every ellipse. All start at 1.
poly_coeffs = torch.tensor([1.0, -4.0, 6.0, -4.0, 1.0]).clone().requires_grad_(True) # (1-t)^4

optimizer = torch.optim.Adam([
    {'params': [positions_n, colors, log_a, log_b, theta], 'lr': 1e-2},
    {'params': [poly_coeffs], 'lr': 1e-4},
], )
loss_history = []
coeff_history = []
diffvg.reset_ellipse_poly_timing()

for t in range(iters):
    optimizer.zero_grad()
    positions_px = positions_n * torch.tensor([canvas_width, canvas_height])
    a_px = torch.exp(log_a) * canvas_width
    b_px = torch.exp(log_b) * canvas_width
    img = pydiffvg.EllipsePolyRenderFunction.apply(
        positions_px, colors, a_px, b_px, theta, poly_coeffs, canvas_width, canvas_height)
    loss = (img - target).pow(2).sum()
    loss_history.append(loss.item())
    coeff_history.append(poly_coeffs.detach().clone().numpy())
    loss.backward()

    print('iter', t, 'loss', loss.item())
    print('poly coeffs [a,b,c,d,e]:', poly_coeffs.detach().numpy())
    a_current = torch.exp(log_a.detach())
    b_current = torch.exp(log_b.detach())
    print('a range:', a_current.min().item(), '-', a_current.max().item())
    print('b range:', b_current.min().item(), '-', b_current.max().item())

    optimizer.step()

    if t == iters - 2:
        second_last_positions_px = (positions_n.detach() * torch.tensor([canvas_width, canvas_height])).clone().numpy()

    with torch.no_grad():
        positions_n.clamp_(0.0, 1.0)
        colors.clamp_(0.0, 1.0)
        log_a.clamp_(torch.log(torch.tensor(0.01)), torch.log(torch.tensor(1.0)))
        log_b.clamp_(torch.log(torch.tensor(0.01)), torch.log(torch.tensor(1.0)))
        # theta and poly_coeffs unclamped

    pydiffvg.imwrite(img.detach().clamp(0, 1).cpu(),
                      'results/polynomial_kernel_rendering/iter_{}.png'.format(t), gamma=1.0)

print(f'final loss: {loss.item():.4f}')
print('final poly coeffs [a,b,c,d,e]:', poly_coeffs.detach().numpy())
with open('results/polynomial_kernel_rendering/final_coeffs.txt', 'w') as f:
    names = ['a (t^4)', 'b (t^3)', 'c (t^2)', 'd (t^1)', 'e (t^0)']
    for name, val in zip(names, poly_coeffs.detach().numpy()):
        f.write(f'{name}: {val:.6f}\n')
with open('results/polynomial_kernel_rendering/final_loss.txt', 'w') as f:
    f.write(str(loss.item()))

diffvg.print_ellipse_poly_timing()
fwd_ms, fwd_calls, bwd_ms, bwd_calls = diffvg.get_ellipse_poly_timing()
with open('results/polynomial_kernel_rendering/timing.txt', 'w') as f:
    f.write(f"render_ellipse_poly timing (N={N}, {iters} iters, {canvas_width}x{canvas_height})\n")
    f.write(f"Forward:  {fwd_ms:.3f} ms total, {fwd_calls} pixel-calls, {fwd_ms/fwd_calls:.6f} ms/pixel\n")
    f.write(f"Backward: {bwd_ms:.3f} ms total, {bwd_calls} pixel-calls, {bwd_ms/bwd_calls:.6f} ms/pixel\n")

# -------------------------------------------------------------------
# Loss curve
# -------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(loss_history)
ax.set_xlabel('Iteration')
ax.set_ylabel('Loss')
ax.set_title(f'Convergence (N={N}, learnable polynomial kernel)')
plt.savefig('results/polynomial_kernel_rendering/loss_curve.png', bbox_inches='tight', dpi=150)
plt.close(fig)
print('saved loss_curve.png')

# -------------------------------------------------------------------
# Coefficient trajectories
# -------------------------------------------------------------------
coeff_history = np.array(coeff_history)  # (iters, 5)
fig, ax = plt.subplots(figsize=(8, 5))
labels = ['a (t^4)', 'b (t^3)', 'c (t^2)', 'd (t^1)', 'e (t^0)']
for j in range(5):
    ax.plot(coeff_history[:, j], label=labels[j])
ax.set_xlabel('Iteration')
ax.set_ylabel('Coefficient value')
ax.set_title('Polynomial coefficient trajectories')
ax.axhline(0, color='gray', linewidth=0.5, linestyle='--')
ax.legend()
plt.savefig('results/polynomial_kernel_rendering/coefficient_trajectories.png', bbox_inches='tight', dpi=150)
plt.close(fig)
print('saved coefficient_trajectories.png')

# -------------------------------------------------------------------
# Kernel shape plot: initial (all coeffs=1) vs learned vs Wendland reference,
# same clamp-to-[0,1].
# -------------------------------------------------------------------
t_range = torch.linspace(0, 1, 200)
final_coeffs = poly_coeffs.detach()

def eval_poly(coeffs, t):
    a, b, c, d, e = coeffs
    f = a * t**4 + b * t**3 + c * t**2 + d * t + e
    return f.clamp(0.0, 1.0)

fig, ax = plt.subplots(figsize=(8, 5))
initial_coeffs = torch.tensor([1.0, -4.0, 6.0, -4.0, 1.0])
ax.plot(t_range.numpy(), eval_poly(initial_coeffs, t_range).numpy(), '--', label='Initial ((1-t)^4)')
ax.plot(t_range.numpy(), eval_poly(final_coeffs, t_range).numpy(), label='Learned')
wendland = (1 - t_range).clamp(min=0)**4 * (4 * t_range + 1)
ax.plot(t_range.numpy(), wendland.numpy(), ':', label='Wendland C2 (reference)')
ax.set_xlabel('t (normalized distance from ellipse center)')
ax.set_ylabel('alpha (opacity)')
ax.set_title('Learned kernel shape vs initial vs Wendland reference')
ax.legend()
plt.savefig('results/polynomial_kernel_rendering/kernel_shape_comparison.png', bbox_inches='tight', dpi=150)
plt.close(fig)
print('saved kernel_shape_comparison.png')

# -------------------------------------------------------------------
# Final render
# -------------------------------------------------------------------
positions_px = positions_n * torch.tensor([canvas_width, canvas_height])
a_px = torch.exp(log_a) * canvas_width
b_px = torch.exp(log_b) * canvas_width
final = pydiffvg.EllipsePolyRenderFunction.apply(
    positions_px, colors, a_px, b_px, theta, poly_coeffs, canvas_width, canvas_height)
pydiffvg.imwrite(final.detach().clamp(0, 1).cpu(), 'results/polynomial_kernel_rendering/final.png', gamma=1.0)

# -------------------------------------------------------------------
# Labeled ellipses with edges
# -------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 8))
display_img = final.detach().clamp(0, 1).cpu().numpy()
ax.imshow(display_img)
pos_np = positions_px.detach().cpu().numpy()
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
plt.savefig('results/polynomial_kernel_rendering/final_labeled.png', bbox_inches='tight', pad_inches=0, dpi=150)
plt.close(fig)
print('saved final_labeled.png')

final_np = final.detach().clamp(0, 1).cpu().numpy()

# -------------------------------------------------------------------
# Quiver plot
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
plt.savefig('results/polynomial_kernel_rendering/movement_quiver.png', bbox_inches='tight', dpi=150)
plt.close(fig)
print('saved movement_quiver.png')

# -------------------------------------------------------------------
# Error heatmap + comparison grid
# -------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 6))
target_np = target.cpu().numpy()
error_map = ((target_np - final_np) ** 2).mean(axis=2)
im = ax.imshow(error_map, cmap='inferno')
ax.axis('off')
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
plt.savefig('results/polynomial_kernel_rendering/error_heatmap.png', bbox_inches='tight', dpi=150)
plt.close(fig)
print('saved error_heatmap.png')

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
axes[0].imshow(target_np)
axes[0].set_title('Target')
axes[0].axis('off')
axes[1].imshow(final_np)
axes[1].set_title('Rendered (polynomial kernel)')
axes[1].axis('off')
im = axes[2].imshow(error_map, cmap='inferno')
axes[2].set_title('Error heatmap')
axes[2].axis('off')
fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
plt.savefig('results/polynomial_kernel_rendering/all_comparison.png', bbox_inches='tight', dpi=150)
plt.close(fig)
print('saved all_comparison.png')

from subprocess import call
call(["ffmpeg", "-framerate", "24", "-i",
    "results/polynomial_kernel_rendering/iter_%d.png", "-vb", "20M",
    "results/polynomial_kernel_rendering/out.mp4"])