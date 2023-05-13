import torch
from torchdiffeq import odeint
import time

# Define the system of ODEs
def B(a, x, y):
    return a[0] + a[1]*x + y*a[2]**2

class ODEFunc(torch.nn.Module):
    def __init__(self, a):
        super(ODEFunc, self).__init__()
        self.a = a

    def forward(self, t, w):
        x, y = w[..., 0], w[..., 1]
        B_val = B(self.a, x, y)
        dxdt = x + y * B_val
        dydt = x - y * 2 * B_val
        return torch.stack([dxdt, dydt], dim=-1)

# Initial conditions x(0) = 0, y(0) = 1
initial_conditions = torch.tensor([0., 1.], requires_grad=True)

# Parameter 'a'
a = torch.tensor([0.5, 0.5, 0.5], requires_grad=True) 

# Time points from t=0 to t=0.1
t = torch.linspace(0., 0.1, 5)

# Use odeint to solve the system
solution = odeint(ODEFunc(a), initial_conditions, t)

# Compute the Jacobian of the final state w.r.t. 'a'
final_state = solution[-1]
jacobian = torch.autograd.functional.jacobian(lambda a: odeint(ODEFunc(a), initial_conditions, t)[-1], a)

print(f"Jacobian: \n{jacobian}")
