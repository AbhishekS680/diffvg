# n_vs_time_trianglesoup_boxed.py
# Measures forward+backward render time as N (number of triangles) increases,
# using the tile-grid accelerated renderer (TriangleSoupBoxedRenderFunction).
# One forward+backward pass per N value
import argparse
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

# --- Command-line arguments ---
# Lets the target image, the list of N values to sweep, and the cooldown
# between runs be set from the shell instead of hardcoded here:
#   python n_vs_time_trianglesoup_boxed.py --image imgs/cat.png --n-values 100,500,1000 --sleep 5
parser = argparse.ArgumentParser()
parser.add_argument('--image', default='imgs/fruit_basket.png', help='Target image path')
parser.add_argument('--n-values', default='50,100,250,500,750,1000,1500,2000,3000,4000,5000',
                     help='Comma-separated list of N values to sweep')
parser.add_argument('--sleep', type=float, default=10.0,
                     help='Seconds to sleep between N values, to let the machine cool')
args = parser.parse_args()

N_values = [int(n) for n in args.n_values.split(',')]

# Fixed target image so canvas size doesn't confound the N sweep
target = torch.from_numpy(skimage.io.imread(args.image)).to(torch.float32) / 255.0
target = target[:, :, :3]
canvas_height, canvas_width = target.shape[0], target.shape[1]
pydiffvg.set_use_gpu(torch.cuda.is_available())
os.makedirs('results/n_vs_time', exist_ok=True)

# Fixed softness for the timing pass -- this script measures raw
# forward/backward cost at a single softness value, not convergence, so
# no annealing is needed (unlike the full rendering scripts).
SOFTNESS = 0.5

forward_times = []
backward_times = []
total_times = []

for N in N_values:
    vertices_n = torch.rand(N, 3, 2).clone().requires_grad_(True)
    colours = torch.rand(N, 3).clone().requires_grad_(True)
    opacity_logit = torch.zeros(N).clone().requires_grad_(True)
    diffvg.reset_trianglesoup_boxed_timing()
    vertices_px = vertices_n * torch.tensor([canvas_width, canvas_height])
    opacity = torch.sigmoid(opacity_logit)
    img = pydiffvg.TriangleSoupBoxedRenderFunction.apply(
        vertices_px, colours, opacity, SOFTNESS, None, canvas_width, canvas_height)
    loss = (img - target).pow(2).sum()
    loss.backward()
    fwd_ms, fwd_calls, bwd_ms, bwd_calls = diffvg.get_trianglesoup_boxed_timing()
    forward_times.append(fwd_ms)
    backward_times.append(bwd_ms)
    total_times.append(fwd_ms + bwd_ms)
    print(f'N={N}: forward={fwd_ms:.2f} ms, backward={bwd_ms:.2f} ms, total={fwd_ms + bwd_ms:.2f} ms')
    time.sleep(args.sleep)  # let the machine cool between runs

with open('results/n_vs_time/n_vs_time_trianglesoup_boxed.txt', 'w') as f:
    f.write('N, forward_ms, backward_ms, total_ms\n')
    for N, fwd, bwd, tot in zip(N_values, forward_times, backward_times, total_times):
        f.write(f'{N}, {fwd:.3f}, {bwd:.3f}, {tot:.3f}\n')
print('saved results/n_vs_time/n_vs_time_trianglesoup_boxed.txt')

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(N_values, total_times, marker='o', label='Total (fwd+bwd)')
ax.plot(N_values, forward_times, marker='o', label='Forward')
ax.plot(N_values, backward_times, marker='o', label='Backward')
ax.set_xlabel('N (number of triangles)')
ax.set_ylabel('Time (ms)')
ax.set_title('Render time vs N (Triangle Soup, boxed/tile-grid)')
ax.legend()
plt.savefig('results/n_vs_time/n_vs_time_trianglesoup_boxed.png', bbox_inches='tight', dpi=150)
plt.close(fig)
print('saved results/n_vs_time/n_vs_time_trianglesoup_boxed.png')