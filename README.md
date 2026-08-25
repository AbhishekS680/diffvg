# diffvg (fork) — Comparative Analysis of Geometric Primitives for Differentiable Image Reconstruction

This is a fork of [diffvg](https://people.csail.mit.edu/tzumao/diffvg), a differentiable
rasterizer for 2D vector graphics. This fork extends diffvg with several additional
kernel-based primitives and investigates which geometric representation offers the best
trade-off between reconstruction fidelity, geometric complexity, and convergence speed,
for the task of reconstructing raster (pixel) images.

This project was completed as part of an NSERC USRA at Carleton University's Graphics,
Imaging, and Games Lab (GIGL), supervised by Dr. David Mould and Dr. Oliver van Kaick.

## What's different in this fork

The original diffvg supports circles, ellipses, rectangles, polygons, curves, and paths,
optimized against a target image via gradient descent through a differentiable rasterizer.

This fork adds five additional primitive types, each implemented as its own C++ kernel
bolted onto the diffvg pipeline, and compares them against each other on image
reconstruction tasks:

- **Triangle meshes / triangle soups** — independent triangles, no shared vertices/edges,
  each with a flat colour and a learnable opacity, composited via soft-rasterized
  alpha-over compositing (`trianglesoup` branch)
- **Splatting kernels** — three variants, each a smooth radial falloff around a control
  point:
  - **Wendland C2** — compact-support polynomial kernel, `(1-t)^4(4t+1)` (`wendland` branch)
  - **Gaussian RBF** — standard Gaussian falloff, `exp(-t^2 / 2*sigma^2)` (`gaussian` branch)
  - **Shepard IDW** — inverse-distance-weighted global interpolation, `1/dist^q`
    (`shepard` branch)
- **Polynomial kernels** — per-primitive polynomial coefficient blending (see
  `poly` branch)

Each kernel branch has both a plain O(N) renderer and a tile-grid accelerated ("boxed")
variant that only evaluates primitives near each pixel, giving a large speedup for large
N. Comparison scripts (`comparison_*.py`) reconstruct a target image starting from a
degraded/blurred version, computing SSIM and LPIPS against the sharp original.

### Example output

*(Add your own result images here, e.g. from `results/.../all_comparison.png`. Suggested
layout below — replace the paths with images you've copied into `docs/images/` in this
repo.)*

![wendland reconstruction](docs/images/wendland_all_comparison.png)
![gaussian reconstruction](docs/images/gaussian_all_comparison.png)
![shepard reconstruction](docs/images/shepard_all_comparison.png)
![trianglesoup reconstruction](docs/images/trianglesoup_all_comparison.png)

## Branch structure

Each primitive lives on its own branch, built off diffvg's `master`:

| Branch | Primitive |
|---|---|
| `master` | Base diffvg (circles, ellipses, paths, etc.) |
| `wendland` | Wendland C2 splatting kernel |
| `gaussian` | Gaussian RBF splatting kernel |
| `shepard` | Shepard IDW global interpolation |
| `trianglesoup` | Independent triangle soup with learnable opacity |

Switching branches automatically rebuilds `pydiffvg` via a `post-checkout` git hook
(`.git/hooks/post-checkout`), since each branch's C++ extension is built separately and a
stale build causes `AttributeError`s from mismatched bindings.

## Install

Same base dependencies as upstream diffvg, plus a few extras used by the comparison
scripts:

```
git submodule update --init --recursive
conda create -n diffvg python=3.10
conda activate diffvg
conda install -y pytorch torchvision -c pytorch
conda install -y numpy
conda install -y scikit-image
conda install -y -c anaconda cmake
conda install -y -c conda-forge ffmpeg
conda install -y matplotlib
pip install svgwrite
pip install svgpathtools
pip install cssutils
pip install numba
pip install lpips
python setup.py install
```

**Rebuilding after switching branches:** if you're not relying on the `post-checkout`
hook, rebuild manually after any `git checkout`:
```
rm -rf build
python setup.py install
```

**Windows:** build from the x64 Native Tools Command Prompt with the conda environment
activated. See `INSTALL.md` for Windows-specific linker/CMake fixes (Python lib paths,
thrust/cub submodule pinning, etc.).

## Running a single primitive

```
cd apps
```

Each primitive has a standalone rendering script that fits N control points/shapes
directly to a target image:

```
python wendland_rendering_boxed.py
python gaussian_rendering_boxed.py
python shepard_rendering.py
python trianglesoup_rendering_boxed.py
```

Each writes its outputs (final render, loss curve, error heatmap, timing, and several
diagnostic visualizations) to `results/<script_name>/`.

## Running a comparison (degraded → sharp reconstruction)

The comparison scripts take a target (sharp) image and a degraded (blurred) image, and
optimize the primitive's parameters to reconstruct the sharp image starting from the
degraded one as the base canvas:

```
python comparison_wendland_boxed.py --target imgs/level_0.png --degraded imgs/level_1.png --outdir results/comparison_wendland_boxed
python comparison_gaussian_boxed.py --target imgs/level_0.png --degraded imgs/level_1.png --outdir results/comparison_gaussian_boxed
python comparison_shepard.py --target imgs/level_0.png --degraded imgs/level_1.png --outdir results/comparison_shepard
python comparison_trianglesoup_boxed.py --target imgs/level_0.png --degraded imgs/level_1.png --outdir results/comparison_trianglesoup_boxed
```

Each comparison run outputs, among other things:
- `all_comparison.png` — degraded | degraded-error | original | reconstruction | reconstruction-error, on a shared error color scale
- `baseline_error.txt` / `focus_summary.txt` — quantitative error summaries
- `timing.txt` — forward/backward pass timing

### Running the full sweep

`run_all_comparisons.sh` runs all four primitives across a chain of degradation levels,
checking out each branch automatically and scoring each hop's reconstruction against the
sharp target:

```
chmod +x run_all_comparisons.sh
caffeinate -i ./run_all_comparisons.sh   # caffeinate keeps macOS awake for the full run
```

Score everything afterward with:
```
python score_results.py
```
which reports SSIM/LPIPS pass/fail per primitive per hop, using thresholds `SSIM > 0.9`
and `LPIPS < 0.1`, and writes a summary table to `results/.../level_summary.txt`.

## Comparing primitives against each other

Two additional scripts (kept in a separate `diffvg-tools` repo, copied into `apps/`
before use) generate cross-primitive error comparisons:

```
python compare_error_heatmaps.py --target imgs/level_0.png --recon-a results/.../wendland/final.png --recon-b results/.../gaussian/final.png --label-a Wendland --label-b Gaussian
python compare_error_heatmaps_multi.py --target imgs/level_0.png --recon Wendland:results/.../final.png --recon Gaussian:results/.../final.png --recon Shepard:results/.../final.png
```

The first produces a red/blue difference map showing exactly which pixels each primitive
reconstructs better; the second puts any number of primitives' error heatmaps on one
shared color scale plus a mean-error bar chart, for report-ready figures.

## Building in debug mode

```
python setup.py build --debug install
```

---

## Original diffvg

The base rasterizer and its original usage (single-shape optimization, painterly
rendering, image vectorization, seam carving, generative models) is documented at
[https://people.csail.mit.edu/tzumao/diffvg](https://people.csail.mit.edu/tzumao/diffvg).
See `apps/` for the original example scripts (`single_circle.py`,
`painterly_rendering.py`, `refine_svg.py`, `seam_carving.py`, etc.), which still work
unmodified on the `master` branch.

If you use diffvg in your academic work, please cite the original paper:
```
@article{Li:2020:DVG,
    title = {Differentiable Vector Graphics Rasterization for Editing and Learning},
    author = {Li, Tzu-Mao and Luk\'{a}\v{c}, Michal and Gharbi Micha\"{e}l and Jonathan Ragan-Kelley},
    journal = {ACM Trans. Graph. (Proc. SIGGRAPH Asia)},
    volume = {39},
    number = {6},
    pages = {193:1--193:15},
    year = {2020}
}
```
