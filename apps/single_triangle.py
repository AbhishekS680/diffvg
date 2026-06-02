# single_triangle.py
# Based on single_rect.py with the following modifications:
# Instead of 4 corners, we use 3 hardcoded corners to form a triangle

# required_grad boolean tells PyTorch to track a variable for gradient computation

import pydiffvg
import torch
import skimage
import numpy as np

# Use GPU if available
pydiffvg.set_use_gpu(torch.cuda.is_available())

canvas_width, canvas_height = 256 ,256

# Creates a triangle
points = torch.stack([
    torch.tensor([100.0, 40.0]),   # top
    torch.tensor([40.0, 160.0]),   # bottom-left
    torch.tensor([160.0, 160.0]),  # bottom-right
])
triangle = pydiffvg.Polygon(points = points, is_closed = True)

shapes = [triangle]

triangle_group = pydiffvg.ShapeGroup(shape_ids = torch.tensor([0]),
                                 fill_color = torch.tensor([0.3, 0.6, 0.3, 1.0]))
shape_groups = [triangle_group]
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
pydiffvg.imwrite(img.cpu(), 'results/single_triangle/target.png', gamma=2.2)
target = img.clone()

# Move the triangle to produce initial guess
points_init = torch.tensor([
    [110.0, 50.0],
    [50.0, 170.0],
    [170.0, 170.0],
], requires_grad=True)
color = torch.tensor([0.3, 0.2, 0.5, 1.0], requires_grad=True)
triangle.points = points_init
triangle_group.fill_color = color

scene_args = pydiffvg.RenderFunction.serialize_scene(\
    canvas_width, canvas_height, shapes, shape_groups)
img = render(256, # width
             256, # height
             2,   # num_samples_x
             2,   # num_samples_y
             1,   # seed
             None, # background_image
             *scene_args)
pydiffvg.imwrite(img.cpu(), 'results/single_triangle/init.png', gamma=2.2)

# Optimize for vertex positions and color
optimizer = torch.optim.Adam([points_init, color], lr=1.0)
# Run 100 Adam iterations.
for t in range(100):
    print('iteration:', t)
    optimizer.zero_grad()
    # Forward pass: render the image.
    triangle.points = points_init
    triangle_group.fill_color = color
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
    pydiffvg.imwrite(img.cpu(), 'results/single_triangle/iter_{}.png'.format(t), gamma=2.2)
    # Compute the loss function. Here it is L2.
    loss = (img - target).pow(2).sum()
    print('loss:', loss.item())

    # Backpropagate the gradients.
    loss.backward()
    # Print the gradients
    print('points.grad:', points_init.grad)
    print('color.grad:', color.grad)

    # Take a gradient descent step.
    optimizer.step()
    # Print the current params.
    print('points:', points_init)
    print('color:', triangle_group.fill_color)

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
pydiffvg.imwrite(img.cpu(), 'results/single_triangle/final.png')

# Convert the intermediate renderings to a video.
from subprocess import call
call(["ffmpeg", "-framerate", "24", "-i",
    "results/single_triangle/iter_%d.png", "-vb", "20M",
    "results/single_triangle/out.mp4"])