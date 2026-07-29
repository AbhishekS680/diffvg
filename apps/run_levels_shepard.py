# run_levels_shepard.py
# Run this ON the shepard branch. Loops the 3 level hops (3->2, 2->1, 1->0),
# calling comparison_shepard.py each time

import subprocess
from score_results import score_path, LEVEL_IMAGES

for n in [3, 2, 1]:
    n_minus_1 = n - 1
    outdir = f'results/level_{n}_to_{n_minus_1}/shepard'
    subprocess.run([
        'python', 'comparison_shepard.py',
        '--target', LEVEL_IMAGES[n_minus_1],
        '--degraded', LEVEL_IMAGES[n],
        '--outdir', outdir,
    ], check=True)

    ssim_val, lpips_val, passed = score_path(f'{outdir}/final.png', LEVEL_IMAGES[n_minus_1])
    print(f"[{n}->{n_minus_1}] shepard: SSIM={ssim_val:.4f} LPIPS={lpips_val:.4f} PASS={passed}")