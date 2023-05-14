import torch

# Parameters and initial conditions
initial_conditions = [5., 5., 5.]
a_initial = [10., 20., 8./3.]  # sigma, rho, beta
tmin = 0
tmax = 0.9
nt = int(40 * (tmax - tmin))
delta_jacobian_scipy = 1e-5
tol_optimization = 1e-2
max_nfev_optimization = 100

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