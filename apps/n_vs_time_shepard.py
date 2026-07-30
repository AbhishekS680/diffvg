# n_vs_time_shepard.py
# Measures forward+backward render time as N (number of control points) increases
# One forward+backward pass per N value
# Not a full training run, since we're isolating per-pass renderer cost

import pydiffvg
import diffvg
import torch
import skimage.io
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Fixed target image so canvas size doesn't confound the N sweep
target = torch.from_numpy(skimage.io.imread('imgs/fruit_basket.png')).to(torch.float32) / 255.0
target = target[:, :, :3]
canvas_height, canvas_width = target.shape[0], target.shape[1]
pydiffvg.set_use_gpu(torch.cuda.is_available())

q = 3.0
N_values = [50, 100, 250, 500, 1000, 2000, 4000, 10000]
forward_times = []
backward_times = []
total_times = []

for N in N_values:
    positions_n = torch.rand(N, 2).clone().requires_grad_(True)
    colors = torch.rand(N, 3).clone().requires_grad_(True)
    diffvg.reset_shepard_timing()
    positions_px = positions_n * torch.tensor([canvas_width, canvas_height])
    img = pydiffvg.ShepardRenderFunction.apply(positions_px, colors, q, canvas_width, canvas_height)
    loss = (img - target).pow(2).sum()
    loss.backward()
    fwd_ms, fwd_calls, bwd_ms, bwd_calls = diffvg.get_shepard_timing()
    forward_times.append(fwd_ms)
    backward_times.append(bwd_ms)
    total_times.append(fwd_ms + bwd_ms)
    print(f'N={N}: forward={fwd_ms:.2f} ms, backward={bwd_ms:.2f} ms, total={fwd_ms + bwd_ms:.2f} ms')

with open('results/n_vs_time_shepard.txt', 'w') as f:
    f.write('N, forward_ms, backward_ms, total_ms\n')
    for N, fwd, bwd, tot in zip(N_values, forward_times, backward_times, total_times):
        f.write(f'{N}, {fwd:.3f}, {bwd:.3f}, {tot:.3f}\n')
        
print('saved results/n_vs_time_shepard.txt')
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(N_values, total_times, marker='o', label='Total (fwd+bwd)')
ax.plot(N_values, forward_times, marker='o', label='Forward')
ax.plot(N_values, backward_times, marker='o', label='Backward')
ax.set_xlabel('N (number of control points)')
ax.set_ylabel('Time (ms)')
ax.set_title('Render time vs N (Shepard)')
ax.legend()
plt.savefig('results/n_vs_time_shepard.png', bbox_inches='tight', dpi=150)
plt.close(fig)
print('saved results/n_vs_time_shepard.png')