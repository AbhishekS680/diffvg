# single_ellipse_wendland.py
#
# Sanity-check / debug script: optimizes a single Wendland ellipse
# (N=1) to recover a known, hand-chosen target ellipse's parameters
# from a deliberately wrong initial guess. Confirms the renderer's
# gradients are well-behaved before scaling up to N=1000+ in the real
# comparison scripts.
#
# NOT part of the core four-primitive comparison used in the report --
# no CLI args, no target image, everything is hardcoded by design.
#
# Usage:
#   python single_ellipse_wendland.py
import os
import torch
import pydiffvg

OUTDIR = 'results/single_ellipse_wendland'
os.makedirs(OUTDIR, exist_ok=True)

canvas_width, canvas_height = 256, 256
iters = 500

# --- Known target ellipse, used to render the target image ---
true_position = torch.tensor([[128.0, 128.0]])
true_a        = torch.tensor([60.0])
true_b        = torch.tensor([30.0])
true_theta    = torch.tensor([0.0])
true_colour   = torch.tensor([[0.3, 0.6, 0.3]])
target = pydiffvg.EllipseWendlandRenderFunction.apply(
    true_position, true_colour, true_a, true_b, true_theta, None,
    canvas_width, canvas_height)
pydiffvg.imwrite(target.cpu(), f'{OUTDIR}/target.png', gamma=1.0)

# --- Deliberately wrong initial guess ---
position_n = torch.tensor([[100.0 / canvas_width, 150.0 / canvas_height]], requires_grad=True)
log_a  = torch.log(torch.tensor([20.0])).clone().requires_grad_(True)
log_b  = torch.log(torch.tensor([40.0])).clone().requires_grad_(True)
theta  = torch.tensor([0.3], requires_grad=True)
colour = torch.tensor([[0.3, 0.2, 0.8]], requires_grad=True)

position_px = position_n * torch.tensor([canvas_width, canvas_height])
img = pydiffvg.EllipseWendlandRenderFunction.apply(
    position_px, colour, torch.exp(log_a), torch.exp(log_b), theta, None,
    canvas_width, canvas_height)
pydiffvg.imwrite(img.detach().cpu(), f'{OUTDIR}/init.png', gamma=1.0)

# --------------------------------------
# Optimize toward the target.
# --------------------------------------
optimizer = torch.optim.Adam([position_n, log_a, log_b, theta, colour], lr=1e-2)
loss_history = []
for t in range(iters):
    optimizer.zero_grad()
    position_px = position_n * torch.tensor([canvas_width, canvas_height])
    img = pydiffvg.EllipseWendlandRenderFunction.apply(
        position_px, colour, torch.exp(log_a), torch.exp(log_b), theta, None,
        canvas_width, canvas_height)
    loss = (img - target).pow(2).sum()
    loss_history.append(loss.item())
    loss.backward()
    print('iter', t, 'loss', loss.item())

    optimizer.step()
    with torch.no_grad():
        position_n.clamp_(0.0, 1.0)
        colour.clamp_(0.0, 1.0)
        log_a.clamp_(torch.log(torch.tensor(1.0)), torch.log(torch.tensor(150.0)))
        log_b.clamp_(torch.log(torch.tensor(1.0)), torch.log(torch.tensor(150.0)))

    pydiffvg.imwrite(img.detach().cpu(), f'{OUTDIR}/iter_{t}.png', gamma=1.0)

print(f'final loss: {loss.item():.4f}')

# --- Final render ---
position_px = position_n * torch.tensor([canvas_width, canvas_height])
img = pydiffvg.EllipseWendlandRenderFunction.apply(
    position_px, colour, torch.exp(log_a), torch.exp(log_b), theta, None,
    canvas_width, canvas_height)
pydiffvg.imwrite(img.detach().cpu(), f'{OUTDIR}/final.png', gamma=1.0)

from subprocess import call
call(["ffmpeg", "-framerate", "24", "-i",
    f"{OUTDIR}/iter_%d.png", "-vb", "20M",
    f"{OUTDIR}/out.mp4"])