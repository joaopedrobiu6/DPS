import torch
import numpy as np
import jax.numpy as jnp

from desc.compute.utils import get_transforms, get_profiles, get_params, dot, cross
from desc.compute import compute as compute_fun
from desc.backend import jnp
from desc.grid import Grid
import desc.io
import desc.examples

model = 'DESC' # 'lorenz' or 'guiding-center' or 'pendulum' or 'DESC'
solver_models = ["JAX"] #, "PyTorch"] # change here the solvers to compare
label_styles = [['k-','k*'], ['r--','rx'], ['b-.','b+']]

# Parameters and initial conditions
if model == 'lorenz':
    variables = ['x', 'y', 'z']
    initial_conditions = [5., 5., 5.]
    a_initial = [10., 20., 8./3.]  # sigma, rho, beta
    tmin = 0
    tmax = 12
    nt_per_time_unit = 120
    n_steps_to_compute_loss = 70
    x_target = 0.5
    x_to_optimize = 0 # optimize x0
    max_nfev_optimization = 20
    learning_rate_torch = 1.1
    learning_rate_jax = 0.2
elif model == 'guiding-center':
    variables = ['psi', 'theta', 'phi', 'vparallel']
    initial_conditions = [0.4, 1.5, 0.1, -0.2]
    a_initial = [0.9, 0.03, 0.05]  # B1c, B1s, B01s
    # a_initial = [ 0.73958918, -0.11144603, -0.08769668]
    iota = 0.418
    G = 0.01
    tmin = 0
    tmax = 8
    nt_per_time_unit = 100
    n_steps_to_compute_loss = 40
    x_target = initial_conditions[0]
    x_to_optimize = 0 # optimize x0
    max_nfev_optimization = 20
    learning_rate_torch = 0.1
    learning_rate_jax = 0.2
elif model == 'pendulum':
    variables = ['x', 'v']
    initial_conditions = [0.1, 0]
    a_initial = [0.4, 0.36]
    tmin = 0
    tmax = 160
    nt_per_time_unit = 120
    n_steps_to_compute_loss = 70
    x_target = 0.8
    x_to_optimize = 0
    max_nfev_optimization = 20
    learning_rate_torch = 1.1
    learning_rate_jax = 0.2

elif model == 'DESC':

    variables = ['psi', 'theta', 'zeta', 'vpar']
    initial_conditions = [0.4, 1.5, 0.1, -0.2]
    a_initial = [0.9, 0.03, 0.05, 1]  # mu, coeff1, coeff2, coeff3
    
    tmin = 0
    tmax = 2
    nt_per_time_unit = 100
    n_steps_to_compute_loss = 40
    x_target = initial_conditions[0]
    x_to_optimize = 0 # optimize x0
    max_nfev_optimization = 20
    learning_rate_torch = 0.1
    learning_rate_jax = 0.2



delta_jacobian_scipy = 1e-7
tol_optimization = 1e-2
use_scipy_jax = False
use_scipy_torch = False
step_optimization_verbose = True

nt = int(nt_per_time_unit * (tmax - tmin))
n_steps_to_compute_loss = np.min([n_steps_to_compute_loss, nt])

# Define the system of equations
if model == 'guiding-center':
    ## Guiding-center equations for x=psi, y=theta, z=phi
    def B(a, x, y, z):
        # return a[0] + a[1] * np.sqrt(x) * np.cos(y) + a[2] * np.sin(z)
        return 1 + a[0] * np.sqrt(x) * np.cos(y-a[1]*z) + a[2] * np.sin(z)
    def dBdx(a, x, y, z):
        # return a[1] * np.cos(y) / (2 * np.sqrt(x))
        return a[0] * np.cos(y-a[1]*z) / (2 * np.sqrt(x))
    def dBdy(a, x, y, z):
        # return -a[1] * np.sqrt(x) * np.sin(y)
        return -a[0] * np.sqrt(x) * np.sin(y-a[1]*z)
    def dBdz(a, x, y, z):
        # return a[2] * np.cos(z)
        return a[0] * a[1] * np.sqrt(x) * np.sin(y-a[1]*z) + a[2] * np.cos(z)
    
    def B_jax(a, x, y, z):
        # return a[0] + a[1] * jnp.sqrt(x) * jnp.cos(y) + a[2] * jnp.sin(z)
        return 1 + a[0] * jnp.sqrt(x) * jnp.cos(y-a[1]*z) + a[2] * jnp.sin(z)
    def dBdx_jax(a, x, y, z):
        # return a[1] * jnp.cos(y) / (2 * jnp.sqrt(x))
        return a[0] * jnp.cos(y-a[1]*z) / (2 * jnp.sqrt(x))
    def dBdy_jax(a, x, y, z):
        # return -a[1] * jnp.sqrt(x) * jnp.sin(y)
        return -a[0] * jnp.sqrt(x) * jnp.sin(y-a[1]*z)
    def dBdz_jax(a, x, y, z):
        # return a[2] * jnp.cos(z)
        return a[0] * a[1] * jnp.sqrt(x) * jnp.sin(y-a[1]*z) + a[2] * jnp.cos(z)
    
    def B_torch(a, x, y, z):
        # return a[0] + a[1] * torch.sqrt(x) * torch.cos(y) + a[2] * torch.sin(z)
        return 1 + a[0] * torch.sqrt(x) * torch.cos(y-a[1]*z) + a[2] * torch.sin(z)
    def dBdx_torch(a, x, y, z):
        # return a[1] * torch.cos(y) / (2 * torch.sqrt(x))
        return a[0] * torch.cos(y-a[1]*z) / (2 * torch.sqrt(x))
    def dBdy_torch(a, x, y, z):
        # return -a[1] * torch.sqrt(x) * torch.sin(y)
        return -a[0] * torch.sqrt(x) * torch.sin(y-a[1]*z)
    def dBdz_torch(a, x, y, z):
        # return a[2] * torch.cos(z)
        return a[0] * a[1] * torch.sqrt(x) * torch.sin(y-a[1]*z) + a[2] * torch.cos(z)
    
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
    
    def system_jax(w, t, a):
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
elif model == 'pendulum':
    def system(w, t, a):
        x, v = w
        dxdt = v
        dvdt = -(a[0]+a[1]*np.cos(t))*np.sin(x)
        return [dxdt, dvdt]
    
    def system_jax(w, t, a):
        x, v = w
        dxdt = v
        dvdt = -(a[0]+a[1]*jnp.cos(t))*jnp.sin(x)
        return [dxdt, dvdt]

    class ODEFunc(torch.nn.Module):
        def __init__(self, a):
            super(ODEFunc, self).__init__()
            self.a = torch.nn.Parameter(a.clone().detach().requires_grad_(True))

        def forward(self, t, w):
            x, v = w[..., 0], w[..., 1]
            dxdt = v
            dvdt = -(self.a[0] + self.a[1]*jnp.cos(t))*jnp.sin(x)
            return torch.stack([dxdt, dvdt], dim=-1)

elif model == 'DESC':
    eq = desc.io.load("/home/joaobiu/DESC/desc/examples/ct32NFP4_init.h5")
    eq._iota = eq.get_profile("iota")
    eq._current = None 

    def system_jax(w, t, a):
    
        #initial conditions
        psi, theta, zeta, vpar = w
        
        mu = a[0]

        #obtaining data from DESC   
        keys = ["B", "|B|", "grad(|B|)", "grad(psi)", "e^theta", "e^zeta", "G"] # etc etc, whatever terms you need
        grid = Grid(jnp.array([psi, theta, zeta]).T, jitable=True, sort=False)
        transforms = get_transforms(keys, eq, grid, jitable=True)
        profiles = get_profiles(keys, eq, grid, jitable=True)
        params = get_params(keys, eq)
        data = compute_fun(eq, keys, params, transforms, profiles)
        
        
        psidot = a[1]*(1/data["|B|"]**3)*(mu*data["|B|"] + vpar**2)*jnp.sum(jnp.cross(data["B"], data["grad(|B|)"], axis=-1) * data["grad(psi)"]) # etc etc
        
        
        thetadot = a[2]*vpar/data["|B|"] * jnp.sum(data["B"] * data["e^theta"]) + (1/data["|B|"]**3)*(mu*data["|B|"] + vpar**2)*jnp.sum(jnp.cross(data["B"], data["grad(|B|)"], axis=-1) * data["e^theta"])
        
        
        zetadot = a[3]*(vpar/data["|B|"]) * dot(data["B"], data["e^zeta"]) 
        
        b = data["B"]/data["|B|"]

        teste1 = (b + (1/(vpar*data["|B|"]**3)) * (mu*data["|B|"] + vpar**2) * jnp.cross(data["B"], data["grad(|B|)"], axis=-1))
        teste2 = data["grad(|B|)"]
        vpardot = -mu*dot(teste1,teste2)
        #vpardot = -mu*jnp.sum(((data["B"]/data["|B|"])+ (1/vpar*data["|B|"]**3)*(mu*data["|B|"] + vpar**2)*jnp.cross(data["B"], data["grad(|B|)"], axis=-1)) * data["grad(|B|)"])
        
        return jnp.array([psidot, thetadot, zetadot, vpardot]) #, zetadot, vpardot])
    
    def system(w, t, a):
        system_jax(w, t, a)

    class ODEFunc(torch.nn.Module):
        def __init__(self, a):
            super(ODEFunc, self).__init__()
            self.a = torch.nn.Parameter(a.clone().detach().requires_grad_(True))

        def forward(self, t, w):
            x, v = w[..., 0], w[..., 1]
            dxdt = v
            dvdt = -(self.a[0] + self.a[1]*jnp.cos(t))*jnp.sin(x)
            return torch.stack([dxdt, dvdt], dim=-1)



