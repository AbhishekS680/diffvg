import torch
import pydiffvg

pydiffvg.set_use_gpu(False)

N = 10
width, height = 64, 64
q = 3.0

positions = torch.rand(N, 2) * 64
colours   = torch.rand(N, 3)

img = pydiffvg.ShepardRenderFunction.apply(positions, colours, q, width, height)

print('output shape:', img.shape)          # expect (64, 64, 3)
print('min/max:', img.min().item(), img.max().item())  # expect ~[0, 1]
print('any NaN:', torch.isnan(img).any().item())       # expect False

pydiffvg.imwrite(img.cpu(), 'results/test_shepard_diffvg/test.png', gamma=2.2)
print('saved test_shepard_diffvg.png')

# Backward pass check
positions_g = positions.clone().requires_grad_(True)
colours_g   = colours.clone().requires_grad_(True)

img2 = pydiffvg.ShepardRenderFunction.apply(positions_g, colours_g, q, width, height)
target = torch.rand(height, width, 3)
loss = (img2 - target).pow(2).sum()
loss.backward()

print('positions grad norm:', positions_g.grad.norm().item())  # expect > 0
print('colours grad norm:',   colours_g.grad.norm().item())    # expect > 0
print('any NaN in grads:', torch.isnan(positions_g.grad).any().item() or
                           torch.isnan(colours_g.grad).any().item())  # expect False