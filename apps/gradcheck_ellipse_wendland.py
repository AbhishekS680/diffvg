# gradcheck_ellipse_wendland.py
#
# Finite-difference gradient check for EllipseWendlandRenderFunction.
# Confirms the hand-derived backward pass (dt/da, dt/db, dt/dtheta,
# dt/dposition) matches a numerical estimate of the same gradients.
#
# NOT part of the core four-primitive comparison used in the report --
# this is a correctness test for the C++ backward pass, run once when
# the kernel math changes, not as part of any reconstruction pipeline.
#
# Usage:
#   python gradcheck_ellipse_wendland.py
#
# No CLI args -- canvas size, ellipse parameters, and the random seed
# are all fixed by design, so the check is fully reproducible.
import torch
import pydiffvg

torch.manual_seed(0)

canvas_width, canvas_height = 64, 64  # small canvas keeps this fast
N = 1  # single ellipse is enough to check the math
pydiffvg.set_use_gpu(False)  # CPU only, for reproducibility

# Random target image -- doesn't need to mean anything, just needs to exist
target = torch.rand(canvas_height, canvas_width, 3)

# --- Parameters to check ---
position = torch.tensor([[32.0, 32.0]], requires_grad=True)
a = torch.tensor([15.0], requires_grad=True)
b = torch.tensor([10.0], requires_grad=True)
theta = torch.tensor([0.4], requires_grad=True)
colour = torch.tensor([[0.5, 0.3, 0.7]], requires_grad=True)

def compute_loss(position, a, b, theta, colour):
    with torch.no_grad():
        img = pydiffvg.EllipseWendlandRenderFunction.apply(
            position, colour, a, b, theta, None, canvas_width, canvas_height)
        return (img - target).pow(2).sum()

# --------------------------------------
# Step 1: analytical gradient, from the backward pass.
# --------------------------------------
loss = pydiffvg.EllipseWendlandRenderFunction.apply(
    position, colour, a, b, theta, None, canvas_width, canvas_height)
loss = (loss - target).pow(2).sum()
loss.backward()

analytical_grads = {
    'position': position.grad.clone(),
    'a': a.grad.clone(),
    'b': b.grad.clone(),
    'theta': theta.grad.clone(),
    'colour': colour.grad.clone(),
}
print('Analytical gradients (from backward pass):')
for name, grad in analytical_grads.items():
    print(f'  {name}: {grad.flatten().tolist()}')

# --------------------------------------
# Step 2: numerical gradient, via central-difference finite differences.
# --------------------------------------
eps = 1e-2
base_position = position.detach().clone()
base_a = a.detach().clone()
base_b = b.detach().clone()
base_theta = theta.detach().clone()
base_colour = colour.detach().clone()

def numerical_grad(param_name):
    """Perturbs the tensor named param_name (one entry at a time), reruns
    the forward pass with all other params held at their base values, and
    estimates the gradient via central difference."""
    params = {
        'position': base_position.clone(),
        'a': base_a.clone(),
        'b': base_b.clone(),
        'theta': base_theta.clone(),
        'colour': base_colour.clone(),
    }
    target_param = params[param_name]
    grad_estimate = torch.zeros_like(target_param)
    flat_param = target_param.view(-1)
    flat_grad = grad_estimate.view(-1)
    for i in range(flat_param.numel()):
        original_value = flat_param[i].item()
        flat_param[i] = original_value + eps
        loss_plus = compute_loss(params['position'], params['a'], params['b'],
                                  params['theta'], params['colour']).item()
        flat_param[i] = original_value - eps
        loss_minus = compute_loss(params['position'], params['a'], params['b'],
                                   params['theta'], params['colour']).item()
        flat_param[i] = original_value  # restore before moving to next entry
        flat_grad[i] = (loss_plus - loss_minus) / (2 * eps)
    return grad_estimate

print('\nComputing numerical gradients (this re-renders many times, may take a moment)...')
numerical_grads = {
    'position': numerical_grad('position'),
    'a': numerical_grad('a'),
    'b': numerical_grad('b'),
    'theta': numerical_grad('theta'),
    'colour': numerical_grad('colour'),
}
print('\nNumerical gradients (finite differences):')
for name, grad in numerical_grads.items():
    print(f'  {name}: {grad.flatten().tolist()}')

# --------------------------------------
# Step 3: compare.
# --------------------------------------
print('\n--- Comparison (analytical vs numerical) ---')
all_ok = True
for name in analytical_grads:
    a_grad = analytical_grads[name].flatten()
    n_grad = numerical_grads[name].flatten()
    diff = (a_grad - n_grad).abs()
    rel_error = diff / (n_grad.abs() + 1e-8)  # avoid divide-by-zero
    print(f'{name}:')
    print(f'  analytical: {a_grad.tolist()}')
    print(f'  numerical:  {n_grad.tolist()}')
    print(f'  abs diff:   {diff.tolist()}')
    print(f'  rel error:  {rel_error.tolist()}')
    if (rel_error > 0.05).any():  # 5% relative error threshold
        print(f'  *** WARNING: {name} gradient mismatch exceeds 5% ***')
        all_ok = False

print('\nAll gradients match within tolerance!' if all_ok else '\nSome gradients did NOT match -- check the flagged parameters above.')