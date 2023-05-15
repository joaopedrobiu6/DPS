import torch
import numpy as np

model = 'lorenz' # 'lorenz' or 'guiding-center'

# Parameters and initial conditions
if model == 'lorenz':
    initial_conditions = [5., 5., 5.]
    a_initial = [10., 20., 8./3.]  # sigma, rho, beta
    tmin = 0
    tmax = 13
    nt_per_time_unit = 80
    n_steps_to_compute_loss = 200
    x_target = 3.5
    x_to_optimize = 0 # optimize x0
    max_nfev_optimization = 20
    learning_rate_torch = 0.4
    learning_rate_jax = 0.2
elif model == 'guiding-center':
    initial_conditions = [5., 5., 5.]
    a_initial = [10., 20., 8./3.]  # sigma, rho, beta
    tmin = 0
    tmax = 0.1
    nt_per_time_unit = 80
    n_steps_to_compute_loss = 200
    x_target = 3.5
    x_to_optimize = 0 # optimize x0
    max_nfev_optimization = 20
    learning_rate_torch = 0.4
    learning_rate_jax = 0.2

delta_jacobian_scipy = 1e-5
tol_optimization = 1e-3
use_scipy_torch = True
use_scipy_jax = False
step_optimization_verbose = False

nt = int(nt_per_time_unit * (tmax - tmin))
n_steps_to_compute_loss = np.min([n_steps_to_compute_loss, nt])

# Define the system of equations
if model == 'lorenz':
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
elif model == 'guiding-center':
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