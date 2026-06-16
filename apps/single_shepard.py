# single_shepard.py
# Implemented using PyTorch, not diffvg

import pydiffvg
import torch

N = 100 # Number of control points
q = 3.0 # Controls how sharply the falloff happens for each control point
iters = 100

def shepard_render(positions, colors, width, height, q, eps=1e-8):
    ys = torch.arange(height, dtype=torch.float32)
    xs = torch.arange(width,  dtype=torch.float32)
    gy, gx = torch.meshgrid(ys, xs, indexing='ij')
    coords = torch.stack([gx, gy], dim=-1)                      # (H,W,2)
    diff = coords[:, :, None, :] - positions[None, None, :, :]  # (H,W,N,2)
    dist = torch.sqrt((diff ** 2).sum(dim=-1))                  # (H,W,N)
    weight = 1.0 / (dist ** q + eps)                            # (H,W,N)
    numer = (weight[..., None] * colors).sum(dim=2)             # (H,W,3)
    denom = weight.sum(dim=2, keepdim=True)                     # (H,W,1)
    return numer / denom                                        # (H,W,3)

# Use GPU if available
pydiffvg.set_use_gpu(torch.cuda.is_available())

canvas_width = 256
canvas_height = 256
circle = pydiffvg.Circle(radius = torch.tensor(40.0),
                         center = torch.tensor([128.0, 128.0]))
shapes = [circle]
circle_group = pydiffvg.ShapeGroup(shape_ids = torch.tensor([0]),
    fill_color = torch.tensor([0.3, 0.6, 0.3, 1.0]))
shape_groups = [circle_group]
scene_args = pydiffvg.RenderFunction.serialize_scene(\
    canvas_width, canvas_height, shapes, shape_groups)

render = pydiffvg.RenderFunction.apply
img = render(canvas_width, # width
             canvas_height, # height
             2,   # num_samples_x
             2,   # num_samples_y
             0,   # seed
             None,
             *scene_args)
# The output image is in linear RGB space. Do Gamma correction before saving the image.
pydiffvg.imwrite(img.cpu(), 'results/single_shepard/target.png', gamma=2.2)
target = img.clone()[..., :3]   # drop alpha, match shepard's (H,W,3)
print('diffvg img shape:', img.shape)

# Initialize N control points and colors randomly.
# Positions are kept in normalized [0,1] coords and scaled to pixel space
# each iteration (helps Adam use a single learning rate for both tensors).
positions_n = (torch.rand(N, 2)).clone().requires_grad_(True)   # normalized [0,1]
colors      = (torch.rand(N, 3)).clone().requires_grad_(True)
optimizer   = torch.optim.Adam([positions_n, colors], lr=1e-2)


# Run 100 Adam iterations.
for t in range(iters):
    optimizer.zero_grad()
    positions = positions_n * canvas_width
    img = shepard_render(positions, colors, canvas_width, canvas_height, q)
    loss = (img - target).pow(2).sum()
    loss.backward()
    
    print('iter', t, 'loss', loss.item())
    print('positions.grad norm:', positions_n.grad.norm().item())
    optimizer.step()

    pydiffvg.imwrite(img.clamp(0, 1).cpu(), 'results/single_shepard/iter_{}.png'.format(t), gamma=2.2)

print(f'final loss: {loss.item():.4f}')

# Render the final result.
final = shepard_render(positions_n * canvas_width, colors, canvas_width, canvas_height, q)
pydiffvg.imwrite(final.clamp(0, 1).cpu(), 'results/single_shepard/final.png', gamma=2.2)

# Convert the intermediate renderings to a video.
from subprocess import call
call(["ffmpeg", "-framerate", "24", "-i",
    "results/single_shepard/iter_%d.png", "-vb", "20M",
    "results/single_shepard/out.mp4"])