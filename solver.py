import torch
import jax.numpy as jnp
import numpy as np
from jax import jit
from scipy.integrate import solve_ivp
from torchdiffeq import odeint_adjoint as torch_odeint
from jax.experimental.ode import odeint as jax_odeint
from params_and_model import system, ODEFunc, initial_conditions, a, tmin, tmax, nt

# Solve the ODE using Scipy
def solve_with_scipy():
    t = np.linspace(tmin, tmax, nt)
    sol = solve_ivp(lambda t, w: system(w, t, a), [tmin, tmax], initial_conditions, t_eval=t, method='RK45')
    return sol.y

# Solve the ODE using PyTorch
def solve_with_pytorch():
    initial_conditions_torch = torch.tensor(initial_conditions, requires_grad=True)
    a_torch = torch.tensor(a, requires_grad=True)
    t_torch = torch.linspace(tmin, tmax, nt)
    solution_torch = torch_odeint(ODEFunc(a_torch), initial_conditions_torch, t_torch).detach().numpy()
    return solution_torch

# Solve the ODE using JAX
def solve_with_jax():
    initial_conditions_jax = jnp.array(initial_conditions, dtype=jnp.float64)
    a_jax = jnp.array(a, dtype=jnp.float64)
    t_jax = jnp.linspace(tmin, tmax, nt)
    system_jit = jit(system)
    solution_jax = jax_odeint(system_jit, initial_conditions_jax, t_jax, a_jax)
    return solution_jax
