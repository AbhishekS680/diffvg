# n_vs_time_wendland.py
# Measures forward+backward render time as N (number of ellipses) increases
# One forward+backward pass per N value

import pydiffvg
import diffvg
import torch
import skimage.io
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import time

# Fixed target image so canvas size doesn't confound the N sweep
target = torch.from_numpy(skimage.io.imread('imgs/fruit_basket.png')).to(torch.float32) / 255.0
target = target[:, :, :3]
canvas_height, canvas_width = target.shape[0], target.shape[1]
pydiffvg.set_use_gpu(torch.cuda.is_available())
os.makedirs('results/n_vs_time', exist_ok=True)

N_values = [50, 100, 250, 500, 750, 1000, 1500, 2000, 3000, 4000, 5000]
forward_times = []
backward_times = []
total_times = []

for N in N_values:
    positions_n = torch.rand(N, 2).clone().requires_grad_(True)
    colors = torch.rand(N, 3).clone().requires_grad_(True)
    log_a = torch.full((N,), torch.log(torch.tensor(0.15))).clone().requires_grad_(True)
    log_b = torch.full((N,), torch.log(torch.tensor(0.15))).clone().requires_grad_(True)
    theta = torch.zeros(N).clone().requires_grad_(True)
    diffvg.reset_ellipse_wendland_timing()
    positions_px = positions_n * torch.tensor([canvas_width, canvas_height])
    a_px = torch.exp(log_a) * canvas_width
    b_px = torch.exp(log_b) * canvas_width
    img = pydiffvg.EllipseWendlandRenderFunction.apply(
        positions_px, colors, a_px, b_px, theta, None, canvas_width, canvas_height)
    loss = (img - target).pow(2).sum()
    loss.backward()
    fwd_ms, fwd_calls, bwd_ms, bwd_calls = diffvg.get_ellipse_wendland_timing()
    forward_times.append(fwd_ms)
    backward_times.append(bwd_ms)
    total_times.append(fwd_ms + bwd_ms)
    print(f'N={N}: forward={fwd_ms:.2f} ms, backward={bwd_ms:.2f} ms, total={fwd_ms + bwd_ms:.2f} ms')
    time.sleep(10)  # let the machine cool between runs

with open('results/n_vs_time/n_vs_time_wendland.txt', 'w') as f:
    f.write('N, forward_ms, backward_ms, total_ms\n')
    for N, fwd, bwd, tot in zip(N_values, forward_times, backward_times, total_times):
        f.write(f'{N}, {fwd:.3f}, {bwd:.3f}, {tot:.3f}\n')

print('saved results/n_vs_time/n_vs_time_wendland.txt')

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(N_values, total_times, marker='o', label='Total (fwd+bwd)')
ax.plot(N_values, forward_times, marker='o', label='Forward')
ax.plot(N_values, backward_times, marker='o', label='Backward')
ax.set_xlabel('N (number of ellipses)')
ax.set_ylabel('Time (ms)')
ax.set_title('Render time vs N (Wendland)')
ax.legend()
plt.savefig('results/n_vs_time/n_vs_time_wendland.png', bbox_inches='tight', dpi=150)
plt.close(fig)
print('saved results/n_vs_time/n_vs_time_wendland.png')