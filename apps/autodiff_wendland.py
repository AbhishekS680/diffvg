import torch

# 1. Define the input and set requires_grad=True to track operations
x = torch.tensor([2.0], requires_grad=True)

# 2. Define the mathematical function
# In this case: y = 2x^2 + 3
y = max((1-x)**4, 0) * (1+4*x)

# 3. Perform the backward pass to compute derivatives
y.backward()


# 4. Access the computed derivative
# The analytical derivative of 2x^2 + 3 is 4x.
# At x = 2, the derivative is 4 * 2 = 8.
print(f"Derivative at x = 2: {x.grad.item()}")
print(d)
