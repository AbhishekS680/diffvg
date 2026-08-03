# single_ellipse_wendland.py
# Optimize a single ellipse (N=1) using the
# EllipseWendlandRenderFunction

import pydiffvg
import torch
import os

os.makedirs('results/single_ellipse_wendland', exist_ok=True)

canvas_width, canvas_height = 256, 256
iters = 500

# ---- Known parameters, used to generate the target image ----
true_position = torch.tensor([[128.0, 128.0]])
true_a        = torch.tensor([60.0])
true_b        = torch.tensor([30.0])
true_theta    = torch.tensor([0.0])
true_colour   = torch.tensor([[0.3, 0.6, 0.3]])

target = pydiffvg.EllipseWendlandRenderFunction.apply(
    true_position, true_colour, true_a, true_b, true_theta, None, 
    canvas_width, canvas_height)
pydiffvg.imwrite(target.cpu(), 'results/single_ellipse_wendland/target.png', gamma=1.0)

# ---- Random initial guess ----
# position normalized to [0,1]
position_n = torch.tensor([[100.0 / canvas_width, 150.0 / canvas_height]], requires_grad=True)
log_a    = torch.log(torch.tensor([20.0])).clone().requires_grad_(True)
log_b    = torch.log(torch.tensor([40.0])).clone().requires_grad_(True)
theta    = torch.tensor([0.3], requires_grad=True)
colour   = torch.tensor([[0.3, 0.2, 0.8]], requires_grad=True)

position_px = position_n * torch.tensor([canvas_width, canvas_height])
img = pydiffvg.EllipseWendlandRenderFunction.apply(
    position_px, colour, torch.exp(log_a), torch.exp(log_b), theta, None,
    canvas_width, canvas_height)
pydiffvg.imwrite(img.detach().cpu(), 'results/single_ellipse_wendland/init.png', gamma=1.0)

# ---- Optimize ----
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
    print('position_n.grad:', position_n.grad)
    print('log_a.grad:', log_a.grad)
    print('log_b.grad:', log_b.grad)
    print('theta.grad:', theta.grad)
    print('colour.grad:', colour.grad)
    optimizer.step()
    with torch.no_grad():
        position_n.clamp_(0.0, 1.0)
        colour.clamp_(0.0, 1.0)
        log_a.clamp_(torch.log(torch.tensor(1.0)), torch.log(torch.tensor(150.0)))
        log_b.clamp_(torch.log(torch.tensor(1.0)), torch.log(torch.tensor(150.0)))
    pydiffvg.imwrite(img.detach().cpu(), 'results/single_ellipse_wendland/iter_{}.png'.format(t), gamma=1.0)

print(f'final loss: {loss.item():.4f}')

# ---- Final render ----
position_px = position_n * torch.tensor([canvas_width, canvas_height])
img = pydiffvg.EllipseWendlandRenderFunction.apply(
    position_px, colour, torch.exp(log_a), torch.exp(log_b), theta, None,
    canvas_width, canvas_height)
pydiffvg.imwrite(img.detach().cpu(), 'results/single_ellipse_wendland/final.png', gamma=1.0)

# ---- Convert intermediate renderings to a video ----
from subprocess import call
call(["ffmpeg", "-framerate", "24", "-i",
    "results/single_ellipse_wendland/iter_%d.png", "-vb", "20M",
    "results/single_ellipse_wendland/out.mp4"])