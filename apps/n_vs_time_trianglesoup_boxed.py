# n_vs_time_trianglesoup_boxed.py
#
# Measures forward+backward render time for the tile-grid accelerated
# Triangle Soup renderer as N (number of triangles) increases -- one
# forward+backward pass per N value, no training. Paired with
# n_vs_time_trianglesoup.py to compute the boxed-vs-plain speedup
# reported in the "Performance Comparison" section of the report.
#
# Usage:
#   python n_vs_time_trianglesoup_boxed.py --image imgs/cat.png --n-values 100,500,1000 --sleep 5
#
# Args:
#   --image     target image path
#   --n-values  comma-separated list of N values to sweep
#   --sleep     seconds to sleep between N values, to let the machine cool
#   --seed      random seed, for reproducibility
#
# SOFTNESS is fixed at 0.5 (matches SOFTNESS_END in the full rendering
# scripts) -- this measures raw forward/backward cost at one softness
# value, not convergence, so no annealing is needed.
import argparse
import os
import time
import torch
import numpy as np
import skimage.io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pydiffvg
import diffvg

parser = argparse.ArgumentParser()
parser.add_argument('--image', default='imgs/fruit_basket.png', help='Target image path')
parser.add_argument('--n-values', default='50,100,250,500,750,1000,1500,2000,3000,4000,5000',
                     help='Comma-separated list of N values to sweep')
parser.add_argument('--sleep', type=float, default=10.0,
                     help='Seconds to sleep between N values, to let the machine cool')
parser.add_argument('--seed', type=int, default=0, help='Random seed')
args = parser.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)

N_values = [int(n) for n in args.n_values.split(',')]
SOFTNESS = 0.5

OUTDIR = 'results/n_vs_time'
os.makedirs(OUTDIR, exist_ok=True)

target = torch.from_numpy(skimage.io.imread(args.image)).to(torch.float32) / 255.0
target = target[:, :, :3]
canvas_height, canvas_width = target.shape[0], target.shape[1]
pydiffvg.set_use_gpu(torch.cuda.is_available())

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
    time.sleep(args.sleep)

with open(f'{OUTDIR}/n_vs_time_trianglesoup_boxed.txt', 'w') as f:
    f.write('N, forward_ms, backward_ms, total_ms\n')
    for N, fwd, bwd, tot in zip(N_values, forward_times, backward_times, total_times):
        f.write(f'{N}, {fwd:.3f}, {bwd:.3f}, {tot:.3f}\n')
print(f'saved {OUTDIR}/n_vs_time_trianglesoup_boxed.txt')

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(N_values, total_times, marker='o', label='Total (fwd+bwd)')
ax.plot(N_values, forward_times, marker='o', label='Forward')
ax.plot(N_values, backward_times, marker='o', label='Backward')
ax.set_xlabel('N (number of triangles)')
ax.set_ylabel('Time (ms)')
ax.set_title('Render time vs N (Triangle Soup, boxed/tile-grid)')
ax.legend()
plt.savefig(f'{OUTDIR}/n_vs_time_trianglesoup_boxed.png', bbox_inches='tight', dpi=150)
plt.close(fig)
print(f'saved {OUTDIR}/n_vs_time_trianglesoup_boxed.png')