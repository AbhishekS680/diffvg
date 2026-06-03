"""
Square primitive rendering using pydiffvg.Polygon.
Each square is defined by a center (cx, cy) and a size parameter.

Scream: python square_rendering.py imgs/scream.jpg --num_squares 2048
Baboon: python square_rendering.py imgs/baboon.png --num_squares 1024 --num_iter 250
Baboon Lpips: python square_rendering.py imgs/baboon.png --num_squares 1024 --num_iter 500 --use_lpips_loss
"""
import pydiffvg
import torch
import skimage
import skimage.io
import random
import ttools.modules
import argparse
import math

pydiffvg.set_print_timing(True)

gamma = 1.0

def main(args):
    # Use GPU if available
    pydiffvg.set_use_gpu(torch.cuda.is_available())
    
    perception_loss = ttools.modules.LPIPS().to(pydiffvg.get_device())
    
    #target = torch.from_numpy(skimage.io.imread('imgs/lena.png')).to(torch.float32) / 255.0
    target = torch.from_numpy(skimage.io.imread(args.target)).to(torch.float32) / 255.0
    target = target.pow(gamma)
    target = target.to(pydiffvg.get_device())
    target = target.unsqueeze(0)
    target = target.permute(0, 3, 1, 2) # NHWC -> NCHW
    #target = torch.nn.functional.interpolate(target, size = [256, 256], mode = 'area')
    canvas_width, canvas_height = target.shape[3], target.shape[2]
    num_squares = args.num_squares
    
    random.seed(1234)
    torch.manual_seed(1234)
    
    shapes = []
    shape_groups = []
    
    square_params = []
    for i in range(num_squares):
        cx = torch.tensor(random.random() * canvas_width)
        cy = torch.tensor(random.random() * canvas_height)
        size = torch.tensor(random.random() * 0.1 * min(canvas_width, canvas_height) + 5.0)
        square_params.append((cx, cy, size))
        half = size / 2.0
        points = torch.stack([
        torch.stack([cx - half, cy - half]),
        torch.stack([cx + half, cy - half]),
        torch.stack([cx + half, cy + half]),
        torch.stack([cx - half, cy + half]),
        ])
        square = pydiffvg.Polygon(points = points, is_closed = True)
        shapes.append(square)
        group = pydiffvg.ShapeGroup(
            shape_ids=torch.tensor([len(shapes) - 1]),
            fill_color=torch.tensor([random.random(), random.random(),
                                    random.random(), random.random()])
        )
        shape_groups.append(group)
    
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
    pydiffvg.imwrite(img.cpu(), 'results/square_rendering/init.png', gamma=gamma)

    cx_vars, cy_vars, size_vars, color_vars = [], [], [], []
    for (cx, cy, size) in square_params:
        cx.requires_grad = True
        cy.requires_grad = True
        size.requires_grad = True
        cx_vars.append(cx)
        cy_vars.append(cy)
        size_vars.append(size)
    for group in shape_groups:
        group.fill_color.requires_grad = True
        color_vars.append(group.fill_color)
    
    # Optimize
    pos_optim = torch.optim.Adam(cx_vars + cy_vars, lr=1.0)
    size_optim = torch.optim.Adam(size_vars, lr=0.5)
    color_optim = torch.optim.Adam(color_vars, lr=0.01)

    # Adam iterations.
    for t in range(args.num_iter):
        print('iteration:', t)
        pos_optim.zero_grad()
        size_optim.zero_grad()
        color_optim.zero_grad()

        for i, (cx, cy, size) in enumerate(zip(cx_vars, cy_vars, size_vars)):
            half = size / 2.0
            shapes[i].points = torch.stack([
                torch.stack([cx - half, cy - half]),
                torch.stack([cx + half, cy - half]),
                torch.stack([cx + half, cy + half]),
                torch.stack([cx - half, cy + half]),
            ])

        # Forward pass: render the image.
        scene_args = pydiffvg.RenderFunction.serialize_scene(\
            canvas_width, canvas_height, shapes, shape_groups)
        img = render(canvas_width, # width
                     canvas_height, # height
                     2,   # num_samples_x
                     2,   # num_samples_y
                     t,   # seed
                     None,
                     *scene_args)
        # Compose img with white background
        img = img[:, :, 3:4] * img[:, :, :3] + torch.ones(img.shape[0], img.shape[1], 3, device = pydiffvg.get_device()) * (1 - img[:, :, 3:4])
        # Save the intermediate render.
        pydiffvg.imwrite(img.cpu(), 'results/square_rendering/iter_{}.png'.format(t), gamma=gamma)
        img = img[:, :, :3]
        # Convert img from HWC to NCHW
        img = img.unsqueeze(0)
        img = img.permute(0, 3, 1, 2) # NHWC -> NCHW
        if args.use_lpips_loss:
            loss = perception_loss(img, target) + (img.mean() - target.mean()).pow(2)
        else:
            loss = (img - target).pow(2).mean()
        print('render loss:', loss.item())
    
        # Backpropagate the gradients.
        loss.backward()

        # Take a gradient descent step.
        pos_optim.step()
        size_optim.step()
        color_optim.step()
        for cx, cy, size in zip(cx_vars, cy_vars, size_vars):
            cx.data.clamp_(0, canvas_width)
            cy.data.clamp_(0, canvas_height)
            size.data.clamp_(1.0, 0.5 * min(canvas_width, canvas_height))
        for group in shape_groups:
            group.fill_color.data.clamp_(0.0, 1.0)

        if t % 10 == 0 or t == args.num_iter - 1:
            pydiffvg.save_svg('results/square_rendering/iter_{}.svg'.format(t),
                              canvas_width, canvas_height, shapes, shape_groups)
    
    # Render the final result.
    img = render(target.shape[1], # width
                 target.shape[0], # height
                 2,   # num_samples_x
                 2,   # num_samples_y
                 0,   # seed
                 None,
                 *scene_args)
    # Save the intermediate render.
    pydiffvg.imwrite(img.cpu(), 'results/square_rendering/final.png'.format(t), gamma=gamma)
    # Convert the intermediate renderings to a video.
    from subprocess import call
    call(["ffmpeg", "-framerate", "24", "-i",
        "results/square_rendering/iter_%d.png", "-vb", "20M",
        "results/square_rendering/out.mp4"])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("target", help="target image path")
    parser.add_argument("--num_squares", type=int, default=512)
    parser.add_argument("--use_lpips_loss", dest='use_lpips_loss', action='store_true')
    parser.add_argument("--num_iter", type=int, default=500)
    args = parser.parse_args()
    main(args)