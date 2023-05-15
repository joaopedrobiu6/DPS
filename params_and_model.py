import torch
import numpy as np

# Parameters and initial conditions
initial_conditions = [5., 5., 5.]
a_initial = [10., 20., 8./3.]  # sigma, rho, beta
tmin = 0
tmax = 9
nt_per_time_unit = 50
n_steps_to_compute_loss = 100
x_target = 2.5
x_to_optimize = 0 # optimize x0
delta_jacobian_scipy = 1e-6
tol_optimization = 1e-3
max_nfev_optimization = 50
learning_rate_torch = 0.05
learning_rate_jax = 0.5

nt = int(nt_per_time_unit * (tmax - tmin))
n_steps_to_compute_loss = np.min([n_steps_to_compute_loss, nt])

# Define the system of equations
def system(w, t, a):
    x, y, z = w
    dxdt = a[0] * (y - x)
    dydt = x * (a[1] - z) - y
    dzdt = x * y - a[2] * z
    return [dxdt, dydt, dzdt]

class ODEFunc(torch.nn.Module):
    def __init__(self, a):
        super(ODEFunc, self).__init__()
        self.a = torch.nn.Parameter(a.clone().detach().requires_grad_(True))

    def forward(self, t, w):
        x, y, z = w[..., 0], w[..., 1], w[..., 2]
        dxdt = self.a[0] * (y - x)
        dydt = x * (self.a[1] - z) - y
        dzdt = x * y - self.a[2] * z
        return torch.stack([dxdt, dydt, dzdt], dim=-1)