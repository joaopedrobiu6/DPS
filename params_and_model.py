import torch
import numpy as np
import jax.numpy as jnp

model = 'guiding-center' # 'lorenz' or 'guiding-center'
solver_models = ["Scipy", "JAX"]#, "PyTorch"] # change here the solvers to compare
label_styles = [['k-','k*'], ['r--','rx'], ['b-.','b+']]

# Parameters and initial conditions
if model == 'lorenz':
    variables = ['x', 'y', 'z']
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
    variables = ['psi', 'theta', 'phi', 'vparallel']
    initial_conditions = [0.4, 1.5, 0.0, -0.2]
    a_initial = [1.0, 0.9, 0.000]  # B0, B1c, B01s
    iota = 0.418
    G = 0.01
    tmin = 0
    tmax = 5
    nt_per_time_unit = 60
    n_steps_to_compute_loss = 30
    x_target = initial_conditions[0]
    x_to_optimize = 0 # optimize x0
    max_nfev_optimization = 20
    learning_rate_torch = 0.1
    learning_rate_jax = 0.2

delta_jacobian_scipy = 1e-5
tol_optimization = 1e-3
use_scipy_jax = False
use_scipy_torch = False
step_optimization_verbose = True

nt = int(nt_per_time_unit * (tmax - tmin))
n_steps_to_compute_loss = np.min([n_steps_to_compute_loss, nt])

# Define the system of equations
if model == 'guiding-center':
    ## Guiding-center equations for x=psi, y=theta, z=phi
    def B(a, x, y, z):
        return a[0] + a[1] * np.sqrt(x) * np.cos(y) + a[2] * np.sin(z)
    def dBdx(a, x, y, z):
        return a[1] * np.cos(y) / (2 * np.sqrt(x))
    def dBdy(a, x, y, z):
        return -a[1] * np.sqrt(x) * np.sin(y)
    def dBdz(a, x, y, z):
        return a[2] * np.cos(z)
    
    def B_jax(a, x, y, z):
        return a[0] + a[1] * jnp.sqrt(x) * jnp.cos(y) + a[2] * jnp.sin(z)
    def dBdx_jax(a, x, y, z):
        return a[1] * jnp.cos(y) / (2 * jnp.sqrt(x))
    def dBdy_jax(a, x, y, z):
        return -a[1] * jnp.sqrt(x) * jnp.sin(y)
    def dBdz_jax(a, x, y, z):
        return a[2] * jnp.cos(z)
    
    def B_torch(a, x, y, z):
        return a[0] + a[1] * torch.sqrt(x) * torch.cos(y) + a[2] * torch.sin(z)
    def dBdx_torch(a, x, y, z):
        return a[1] * torch.cos(y) / (2 * torch.sqrt(x))
    def dBdy_torch(a, x, y, z):
        return -a[1] * torch.sqrt(x) * torch.sin(y)
    def dBdz_torch(a, x, y, z):
        return a[2] * torch.cos(z)
    
    # Lambda = mu*B0/Energy
    Lambda = (1-initial_conditions[3]**2)/B(a_initial, initial_conditions[0], initial_conditions[1], initial_conditions[2]) # 0.8
    
    # Define the system of equations
    def system(w, t, a):
        # x, y, z, v = w
        # v_parallel = vpar_sign*jnp.sqrt(1-Lambda*B_val)
        x, y, z, v_parallel = w
        B_val = B(a, x, y, z)
        dBdx_val = dBdx(a,x,y,z)
        dBdy_val = dBdy(a,x,y,z)
        dBdz_val = dBdz(a,x,y,z)
        dxdt = -1/B_val*dBdy_val*(2/Lambda-B_val)
        dydt =  1/B_val*dBdx_val*(2/Lambda-B_val)+iota*v_parallel*B_val/G
        dzdt = v_parallel*B_val/G
        dvdt = -(iota*dBdy_val + dBdz_val)*B_val/G*Lambda/2
        # return [dxdt, dydt, dzdt]
        return [dxdt, dydt, dzdt, dvdt]

    def system_jax(w, t, a):
        # x, y, z, v = w
        # v_parallel = vpar_sign*jnp.sqrt(1-Lambda*B_val)
        x, y, z, v_parallel = w
        B_val = B_jax(a, x, y, z)
        dBdx_val = dBdx_jax(a,x,y,z)
        dBdy_val = dBdy_jax(a,x,y,z)
        dBdz_val = dBdz_jax(a,x,y,z)
        dxdt = -1/B_val*dBdy_val*(2/Lambda-B_val)
        dydt =  1/B_val*dBdx_val*(2/Lambda-B_val)+iota*v_parallel*B_val/G
        dzdt = v_parallel*B_val/G
        dvdt = -(iota*dBdy_val + dBdz_val)*B_val/G*Lambda/2
        # return [dxdt, dydt, dzdt]
        return [dxdt, dydt, dzdt, dvdt]

    class ODEFunc(torch.nn.Module):
        def __init__(self, a):
            super(ODEFunc, self).__init__()
            self.a = torch.nn.Parameter(a.clone().detach().requires_grad_(True))

        def forward(self, t, w):
            # x, y, z = w[..., 0], w[..., 1], w[..., 2]
            # v_parallel = vpar_sign*torch.sqrt(1-Lambda*B_val)
            x, y, z, v_parallel = w[..., 0], w[..., 1], w[..., 2], w[..., 3]
            
            # B_val    = self.a[0] + self.a[1] * torch.sqrt(x) * torch.cos(y) + self.a[2] * torch.sin(z)
            # dBdx_val = self.a[1] * torch.cos(y) / (2 * torch.sqrt(x))
            # dBdy_val =-self.a[1] * torch.sqrt(x) * torch.sin(y)
            # dBdz_val = self.a[2] * torch.cos(y)
            B_val = B_torch(self.a, x, y, z)
            dBdx_val = dBdx_torch(self.a,x,y,z)
            dBdy_val = dBdy_torch(self.a,x,y,z)
            dBdz_val = dBdz_torch(self.a,x,y,z)

            dxdt = -1/B_val*dBdy_val*(2/Lambda-B_val)
            dydt =  1/B_val*dBdx_val*(2/Lambda-B_val)+iota*v_parallel*B_val/G
            dzdt = v_parallel*B_val/G
            dvdt = -(iota*dBdy_val + dBdz_val)*B_val/G*Lambda/2
            # return torch.stack([dxdt, dydt, dzdt], dim=-1)
            return torch.stack([dxdt, dydt, dzdt, dvdt], dim=-1)
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