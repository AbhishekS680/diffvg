# shepard_rendering.py
# Implemented using diffvg's ShepardField C++ renderer

import pydiffvg
import torch

N = 100 # Number of control points
q = 3.0 # Controls how sharply the falloff happens for each control point
iters = 100

# Use GPU if available
pydiffvg.set_use_gpu(torch.cuda.is_available())

canvas_width = 256
canvas_height = 256

# Replace with image later
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
pydiffvg.imwrite(img.cpu(), 'results/shepard_rendering/target.png', gamma=2.2)
target = img.clone()[..., :3]   # drop alpha, match shepard's (H,W,3)
print('diffvg img shape:', img.shape)

# Initialize N control points and colors randomly
positions_n = (torch.rand(N, 2)).clone().requires_grad_(True) # normalized [0,1]
colors      = (torch.rand(N, 3)).clone().requires_grad_(True)
optimizer   = torch.optim.Adam([positions_n, colors], lr=1e-2)


# Run Adam iterations.
for t in range(iters):
    optimizer.zero_grad()
    positions = positions_n * canvas_width
    img = pydiffvg.ShepardRenderFunction.apply(positions, colors, q, canvas_width, canvas_height) # forward → C++ render_shepard
    loss = (img - target).pow(2).sum() # how wrong is the current render
    loss.backward() # backward → C++ fills d_positions, d_colours → deposits into .grad -> d_render_image created
    
    print('iter', t, 'loss', loss.item())
    print('positions.grad norm:', positions_n.grad.norm().item())
    optimizer.step() # Adam reads .grad → moves positions_n and colors

    pydiffvg.imwrite(img.clamp(0, 1).cpu(), 'results/shepard_rendering/iter_{}.png'.format(t), gamma=2.2)

print(f'final loss: {loss.item():.4f}')

# Render the final result.
final = pydiffvg.ShepardRenderFunction.apply(positions_n * canvas_width, colors, q, canvas_width, canvas_height)
pydiffvg.imwrite(final.clamp(0, 1).cpu(), 'results/shepard_rendering/final.png', gamma=2.2)

# Convert the intermediate renderings to a video.
from subprocess import call
call(["ffmpeg", "-framerate", "24", "-i",
    "results/shepard_rendering/iter_%d.png", "-vb", "20M",
    "results/shepard_rendering/out.mp4"])