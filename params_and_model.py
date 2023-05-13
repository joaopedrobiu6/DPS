import jax.numpy as jnp
import torch

# Parameters and initial conditions
initial_conditions = [0., 1.]
a = [0.1, 0.2, 0.3]
tmin = 0
tmax = 3
nt = 50
delta = 1e-5

# Define the system of equations
def B(a, x, y):
    return a[0] + a[1] * x + y * a[2] ** 2

def system(w, t, a):
    x, y = w
    B_val = B(a, x, y)
    return [x + y * B_val, x - y * 2 * B_val]

class ODEFunc(torch.nn.Module):
    def __init__(self, a):
        super(ODEFunc, self).__init__()
        self.a = a

    def forward(self, t, w):
        x, y = w[..., 0], w[..., 1]
        B_val = self.a[0] + self.a[1] * x + y * self.a[2] ** 2
        dxdt = x + y * B_val
        dydt = x - y * 2 * B_val
        return torch.stack([dxdt, dydt], dim=-1)
