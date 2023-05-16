import torch
import numpy as np
import jax.numpy as jnp

model = 'guiding-center' # 'lorenz' or 'guiding-center'

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
    learning_rate_torch = 1.1
    learning_rate_jax = 0.2
elif model == 'guiding-center':
    initial_conditions = [0.5, 0.1, 0.1]
    a_initial = [1.0, 0.1, 0.01]  # B0, B1c, alpha
    tmin = 0
    tmax = 200
    nt_per_time_unit = 10
    n_steps_to_compute_loss = 30
    x_target = initial_conditions[0]
    x_to_optimize = 0 # optimize x0
    max_nfev_optimization = 20
    learning_rate_torch = 0.1
    learning_rate_jax = 0.2

delta_jacobian_scipy = 1e-5
tol_optimization = 1e-3
use_scipy_torch = True
use_scipy_jax = False
step_optimization_verbose = True

nt = int(nt_per_time_unit * (tmax - tmin))
n_steps_to_compute_loss = np.min([n_steps_to_compute_loss, nt])

# Define the system of equations
if model == 'guiding-center':
    ## Guiding-center equations for x=psi, y=theta, z=phi
    # Define the system of equations
    Lambda = 0.1
    iota = 0.41
    G = 2*np.pi
    def B(a, x, y, z):
        return a[0] + a[1] * x * jnp.cos(y) + a[2] * jnp.sin(z)
    
    def dBdx(a, x, y, z):
        return a[1] * jnp.cos(y)
    
    def dBdy(a, x, y, z):
        return -a[1] * x * jnp.sin(y)

    def system(w, t, a):
        x, y, z = w

        vpar_sign = 1
        B_val = B(a, x, y, z)
        dBdx_val = dBdx(a,x,y,z)
        dBdy_val = dBdy(a,x,y,z)

        v_parallel = vpar_sign*jnp.sqrt(1-Lambda*B_val)
        dxdt =-(2*Lambda-B_val)/(2*Lambda*B_val)*dBdy_val
        dydt = (B_val-2*Lambda)/(2*Lambda*B_val)*dBdx_val+iota*v_parallel/G*B_val
        dzdt = v_parallel*B_val/G
        return [dxdt, dydt, dzdt]

    class ODEFunc(torch.nn.Module):
        def __init__(self, a):
            super(ODEFunc, self).__init__()
            self.a = torch.nn.Parameter(a.clone().detach().requires_grad_(True))

        def forward(self, t, w):
            x, y, z = w[..., 0], w[..., 1], w[..., 2]
            
            vpar_sign = 1
            B_val = self.a[0] + self.a[1] * x * torch.cos(y) + self.a[2] * torch.sin(z)
            dBdx_val = self.a[1] * torch.cos(y)
            dBdy_val = -self.a[1] * x * torch.sin(y)

            v_parallel = vpar_sign*torch.sqrt(1-Lambda*B_val)
            dxdt =-(2*Lambda-B_val)/(2*Lambda*B_val)*dBdy_val
            dydt = (B_val-2*Lambda)/(2*Lambda*B_val)*dBdx_val+iota*v_parallel/G*B_val
            dzdt = v_parallel*B_val/G
            return torch.stack([dxdt, dydt, dzdt], dim=-1)
elif model == 'lorenz':
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