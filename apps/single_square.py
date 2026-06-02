# single_square.py
# Based on single_rect.py with the following modifications:
# Instead of using pydiffvg.Rect, we use pydiffvg.Polygon with 4 corners
# derived from a center (cx, cy) and a single size parameter to enforce squareness

# required_grad boolean tells PyTorch to track a variable for gradient computation

import pydiffvg
import torch
import skimage
import numpy as np

# Use GPU if available
pydiffvg.set_use_gpu(torch.cuda.is_available())

canvas_width, canvas_height = 256 ,256

# Attributes of the square
cx = torch.tensor(100.0)
cy = torch.tensor(100.0)
size = torch.tensor(120.0)
half = size / 2.0

# Creates a square
points = torch.stack([
    torch.stack([cx - half, cy - half]),
    torch.stack([cx + half, cy - half]),
    torch.stack([cx + half, cy + half]),
    torch.stack([cx - half, cy + half]),
])

square = pydiffvg.Polygon(points = points, is_closed = True)

shapes = [square]

square_group = pydiffvg.ShapeGroup(shape_ids = torch.tensor([0]),
                                 fill_color = torch.tensor([0.3, 0.6, 0.3, 1.0]))
shape_groups = [square_group]
scene_args = pydiffvg.RenderFunction.serialize_scene(\
    canvas_width, canvas_height, shapes, shape_groups)

render = pydiffvg.RenderFunction.apply
img = render(256, # width
             256, # height
             2,   # num_samples_x
             2,   # num_samples_y
             0,   # seed
             None, # background_image
             *scene_args)
# The output image is in linear RGB space. Do Gamma correction before saving the image.
pydiffvg.imwrite(img.cpu(), 'results/single_square/target.png', gamma=2.2)
target = img.clone()

# Move the square to produce initial guess
# normalize cx, cy, size to [0, 1] range for easier learning rate
cx = torch.tensor(130.0 / 256.0, requires_grad=True)
cy = torch.tensor(130.0 / 256.0, requires_grad=True)
size = torch.tensor(60.0 / 256.0, requires_grad=True)
color = torch.tensor([0.3, 0.2, 0.5, 1.0], requires_grad=True)
cx_px = cx * 256
cy_px = cy * 256
half = (size * 256) / 2.0
square.points = torch.stack([
    torch.stack([cx_px - half, cy_px - half]),
    torch.stack([cx_px + half, cy_px - half]),
    torch.stack([cx_px + half, cy_px + half]),
    torch.stack([cx_px - half, cy_px + half]),
])
square_group.fill_color = color
scene_args = pydiffvg.RenderFunction.serialize_scene(\
    canvas_width, canvas_height, shapes, shape_groups)
img = render(256, # width
             256, # height
             2,   # num_samples_x
             2,   # num_samples_y
             1,   # seed
             None, # background_image
             *scene_args)
pydiffvg.imwrite(img.cpu(), 'results/single_square/init.png', gamma=2.2)

# Optimize for center position and size
optimizer = torch.optim.Adam([cx, cy, size, color], lr=1e-2)
# Run 100 Adam iterations.
for t in range(100):
    print('iteration:', t)
    optimizer.zero_grad()
    # Forward pass: render the image.
    cx_px = cx * 256
    cy_px = cy * 256
    half = (size * 256) / 2.0
    square.points = torch.stack([
        torch.stack([cx_px - half, cy_px - half]),
        torch.stack([cx_px + half, cy_px - half]),
        torch.stack([cx_px + half, cy_px + half]),
        torch.stack([cx_px - half, cy_px + half]),
    ])
    square_group.fill_color = color
    scene_args = pydiffvg.RenderFunction.serialize_scene(\
        canvas_width, canvas_height, shapes, shape_groups)
    img = render(256,   # width
                 256,   # height
                 2,     # num_samples_x
                 2,     # num_samples_y
                 t+1,   # seed
                 None, # background_image
                 *scene_args)
    # Save the intermediate render.
    pydiffvg.imwrite(img.cpu(), 'results/single_square/iter_{}.png'.format(t), gamma=2.2)
    # Compute the loss function. Here it is L2.
    loss = (img - target).pow(2).sum()
    print('loss:', loss.item())

    # Backpropagate the gradients.
    loss.backward()
    # Print the gradients
    print('cx.grad:', cx.grad)
    print('cy.grad:', cy.grad)
    print('size.grad:', size.grad)
    print('color.grad:', color.grad)

    # Take a gradient descent step.
    optimizer.step()
    # Print the current params.
    print('cx:', cx)
    print('cy:', cy)
    print('size:', size)
    print('color:', square_group.fill_color)

# Render the final result.
scene_args = pydiffvg.RenderFunction.serialize_scene(\
    canvas_width, canvas_height, shapes, shape_groups)
img = render(256,   # width
             256,   # height
             2,     # num_samples_x
             2,     # num_samples_y
             102,    # seed
             None, # background_image
             *scene_args)
# Save the images and differences.
pydiffvg.imwrite(img.cpu(), 'results/single_square/final.png')

# Convert the intermediate renderings to a video.
from subprocess import call
call(["ffmpeg", "-framerate", "24", "-i",
    "results/single_square/iter_%d.png", "-vb", "20M",
    "results/single_square/out.mp4"])