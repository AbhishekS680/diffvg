# shepard_rendering.py
# Implemented using diffvg's ShepardField C++ renderer

import pydiffvg
import torch

import skimage.io

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

N = 50 # Number of control points
q = 3.0 # Controls how sharply the falloff happens for each control point
iters = 500

# Use GPU if available
pydiffvg.set_use_gpu(torch.cuda.is_available())

target = torch.from_numpy(skimage.io.imread('imgs/shepard_ex.png')).to(torch.float32) / 255.0
target = target[:, :, :3]  # keep RGB only
canvas_height, canvas_width = target.shape[0], target.shape[1]
pydiffvg.imwrite(target.cpu(), 'results/shepard_rendering/target.png', gamma=1.0)
print('target shape:', target.shape)

# Initialize N control points and colors randomly
positions_n = (torch.rand(N, 2)).clone().requires_grad_(True) # normalized [0,1]
colors      = (torch.rand(N, 3)).clone().requires_grad_(True)
optimizer   = torch.optim.Adam([positions_n, colors], lr=1e-2)
loss_history = []


# Run Adam iterations.
for t in range(iters):
    optimizer.zero_grad()
    positions = positions_n * torch.tensor([canvas_width, canvas_height])
    img = pydiffvg.ShepardRenderFunction.apply(positions, colors, q, canvas_width, canvas_height) # forward → C++ render_shepard
    loss = (img - target).pow(2).sum() # how wrong is the current render
    loss_history.append(loss.item())
    loss.backward() # backward → C++ fills d_positions, d_colours → deposits into .grad -> d_render_image created
    
    print('iter', t, 'loss', loss.item())
    print('positions.grad norm:', positions_n.grad.norm().item())
    optimizer.step() # Adam reads .grad → moves positions_n and colors

    # Helps the optimized parameters stay inside the bounds[0,1] after each optimizer.step()
    with torch.no_grad():
        positions_n.clamp_(0.0, 1.0)
        colors.clamp_(0.0, 1.0)

    pydiffvg.imwrite(img.clamp(0, 1).cpu(), 'results/shepard_rendering/iter_{}.png'.format(t), gamma=2.2)

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
pydiffvg.imwrite(final.clamp(0, 1).cpu(), 'results/shepard_rendering/final.png', gamma=2.2)

# -------------------------------------------------------------------
# Visualization: overlay control point locations on the final render.
# -------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 8)) # Creates a matplotlib figure to draw on
display_img = final.detach().clamp(0, 1).cpu().numpy() ** (1 / 2.2) # Gets the final rendered image from PyTorch
ax.imshow(display_img)

# Gets the pixel coordinates of the control points and plots small red dots at each of those coordinates
pos_np = (positions_n.detach() * torch.tensor([canvas_width, canvas_height])).cpu().numpy()
ax.scatter(pos_np[:, 0], pos_np[:, 1], c='red', s=15, edgecolors='white', linewidths=0.5)

ax.set_xlim(0, canvas_width)
ax.set_ylim(canvas_height, 0)
fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

ax.axis('off') # Hides the plot's axis lines, ticks, and labels
plt.savefig('results/shepard_rendering/final_labeled.png', bbox_inches='tight', pad_inches=0, dpi=150)
plt.close(fig) # Releases the figure from memory
print('saved final_labeled.png')

# Convert the intermediate renderings to a video.
from subprocess import call
call(["ffmpeg", "-framerate", "24", "-i",
    "results/shepard_rendering/iter_%d.png", "-vb", "20M",
    "results/shepard_rendering/out.mp4"])