# trianglesoup_prototype.py
# Pure PyTorch prototype for the triangle soup primitive -- independent
# triangles with no shared vertices/edges (unlike a triangle mesh), each
# with its own flat color. No diffvg/C++ involvement yet -- this is the
# math-validation stage, same pattern as single_shepard.py and the
# Wendland PyTorch prototype: validate here first, port to C++ later.
#
# Triangles have hard edges (a plain point-in-triangle test is a step
# function), so a naive render gives zero gradient for vertex positions.
# This uses soft rasterization: each edge gets a sigmoid-smoothed "inside"
# test so coverage fades smoothly near triangle boundaries, which is what
# actually lets Adam move the vertices.
import torch
import skimage.io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

os.makedirs('results/trianglesoup_prototype', exist_ok=True)

N = 500          # number of triangles
iters = 100
SOFTNESS = 2.0    # pixels; larger = blurrier edges but stronger gradients

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

target = torch.from_numpy(skimage.io.imread('imgs/fruit_basket.png')).to(torch.float32) / 255.0
target = target[:, :, :3].to(device)  # keep RGB only
canvas_height, canvas_width = target.shape[0], target.shape[1]

# Pixel grid, computed once and reused every iteration (not inside the loop)
ys, xs = torch.meshgrid(
    torch.arange(canvas_height, dtype=torch.float32, device=device),
    torch.arange(canvas_width, dtype=torch.float32, device=device),
    indexing='ij')
pixel_coords = torch.stack([xs, ys], dim=-1)  # (H, W, 2)

# Initialize N triangles: random vertices (normalized [0,1]) and random flat colors.
# vertices_n: (N, 3, 2) -- 3 vertices per triangle, each (x, y) normalized
vertices_n = torch.rand(N, 3, 2, device=device).clone().requires_grad_(True)
colors = torch.rand(N, 3, device=device).clone().requires_grad_(True)

optimizer = torch.optim.Adam([vertices_n, colors], lr=1e-2)
loss_history = []


def soft_triangle_coverage(verts_px):
    """verts_px: (N, 3, 2) triangle vertices in pixel coordinates.
    Returns (N, H, W) soft coverage in [0, 1] -- 1.0 well inside a
    triangle, 0.0 well outside, smoothly blended near edges."""
    p0 = verts_px[:, 0, :].view(N, 1, 1, 2)
    p1 = verts_px[:, 1, :].view(N, 1, 1, 2)
    p2 = verts_px[:, 2, :].view(N, 1, 1, 2)
    pix = pixel_coords.view(1, canvas_height, canvas_width, 2)

    # Signed edge function for each of the 3 edges: positive on one side,
    # negative on the other. For a consistently-wound triangle, "inside"
    # means the same sign on all three.
    def edge_fn(a, b, p):
        return (p[..., 0] - a[..., 0]) * (b[..., 1] - a[..., 1]) \
             - (p[..., 1] - a[..., 1]) * (b[..., 0] - a[..., 0])

    e0 = edge_fn(p0, p1, pix)  # (N, H, W)
    e1 = edge_fn(p1, p2, pix)
    e2 = edge_fn(p2, p0, pix)

    # Soft "inside" test per edge: sigmoid turns the hard sign test into a
    # smooth ramp SOFTNESS pixels wide, so there's a real gradient near
    # the boundary instead of a flat zero.
    s0 = torch.sigmoid(e0 / SOFTNESS)
    s1 = torch.sigmoid(e1 / SOFTNESS)
    s2 = torch.sigmoid(e2 / SOFTNESS)

    # Triangle winding can be either orientation depending on random init,
    # so take whichever consistent sign combination gives coverage -- use
    # the product for one winding and its complement for the other, then
    # take the max (a triangle "belongs" to whichever winding is coherent).
    coverage_pos = s0 * s1 * s2
    coverage_neg = (1 - s0) * (1 - s1) * (1 - s2)
    return torch.maximum(coverage_pos, coverage_neg)


def render(verts_n, cols):
    """Composite N triangles via alpha-over in index order (painter's
    algorithm: triangle N-1 is painted last, on top).

    Vectorized instead of looping over triangles in Python: triangle i's
    surviving contribution is alpha_i * color_i * (product of (1-alpha)
    for every triangle painted on top of it, i.e. indices i+1..N-1). That
    running product is a cumulative product, computed once for all N
    triangles at once instead of N sequential Python steps -- this is
    what actually made the loop version slow, not the coverage math
    itself (which was already vectorized across N)."""
    verts_px = verts_n * torch.tensor([canvas_width, canvas_height], device=device)
    coverage = soft_triangle_coverage(verts_px)  # (N, H, W)
    alpha = coverage.unsqueeze(-1)  # (N, H, W, 1)
    one_minus_alpha = 1 - alpha

    # Exclusive suffix product: suffix[i] = product of one_minus_alpha[i+1:]
    # (i.e. everything painted after/on top of triangle i). Computed by
    # reversing along the triangle dimension, taking an exclusive cumprod,
    # then reversing back.
    rev = torch.flip(one_minus_alpha, dims=[0])
    cumprod_rev = torch.cumprod(rev, dim=0)
    ones_row = torch.ones_like(cumprod_rev[:1])
    shifted = torch.cat([ones_row, cumprod_rev[:-1]], dim=0)
    suffix = torch.flip(shifted, dims=[0])  # (N, H, W, 1)

    weight = alpha * suffix  # (N, H, W, 1)
    img = (weight * cols.view(N, 1, 1, 3)).sum(dim=0)  # (H, W, 3)
    return img


# --------------------------------------
# Run Adam iterations.
# --------------------------------------
for t in range(iters):
    optimizer.zero_grad()
    img = render(vertices_n, colors)
    loss = (img - target).pow(2).sum()
    loss_history.append(loss.item())
    loss.backward()
    print('iter', t, 'loss', loss.item())
    optimizer.step()

    with torch.no_grad():
        vertices_n.clamp_(0.0, 1.0)
        colors.clamp_(0.0, 1.0)

    if t % 10 == 0 or t == iters - 1:
        with torch.no_grad():
            preview = render(vertices_n, colors).clamp(0, 1).cpu().numpy()
        skimage.io.imsave(f'results/trianglesoup_prototype/iter_{t}.png',
                           (preview * 255).astype('uint8'))

print(f'final loss: {loss.item():.4f}')

# --------------------------------------
# Plot loss convergence over iterations.
# --------------------------------------
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(loss_history)
ax.set_xlabel('Iteration')
ax.set_ylabel('Loss')
ax.set_title(f'Convergence (N={N} triangles, soft rasterization, softness={SOFTNESS}px)')
plt.savefig('results/trianglesoup_prototype/loss_curve.png', bbox_inches='tight', dpi=150)
plt.close(fig)
print('saved loss_curve.png')

# --------------------------------------
# Final render.
# --------------------------------------
with torch.no_grad():
    final = render(vertices_n, colors).clamp(0, 1)
skimage.io.imsave('results/trianglesoup_prototype/final.png',
                   (final.cpu().numpy() * 255).astype('uint8'))
print('saved final.png')