import torch
import jax.numpy as jnp
import numpy as np
from jax import jit, jacobian as jax_jacobian
from scipy.integrate import solve_ivp, odeint
from torchdiffeq import odeint as torch_odeint
from jax.experimental.ode import odeint as jax_odeint
from params_and_model import system, ODEFunc, initial_conditions, a, tmin, tmax, nt, delta_jacobian_scipy

num_functions = len(initial_conditions)  # assuming number of functions is the same as the length of initial conditions

# Compute Scipy Jacobians
def compute_jacobian_scipy():
    t = np.linspace(tmin, tmax, nt)
    Jacobian_scipy = np.empty((num_functions, len(a)))
    for i in range(len(a)):
        a_plus_delta = a.copy()
        a_plus_delta[i] += delta_jacobian_scipy
        a_minus_delta = a.copy()
        a_minus_delta[i] -= delta_jacobian_scipy

        # sol_plus_delta = solve_ivp(lambda t, w: system(w, t, a_plus_delta), [tmin, tmax], initial_conditions, t_eval=t)
        # sol_minus_delta = solve_ivp(lambda t, w: system(w, t, a_minus_delta), [tmin, tmax], initial_conditions, t_eval=t)
        # Jacobian_scipy[:, i] = (sol_plus_delta.y[:, -1] - sol_minus_delta.y[:, -1]) / (2 * delta_jacobian_scipy)

        sol_plus_delta  = odeint(lambda w, t: system(w, t, a_plus_delta), initial_conditions, t)
        sol_minus_delta = odeint(lambda w, t: system(w, t, a_minus_delta), initial_conditions, t)
        Jacobian_scipy[:, i] = (sol_plus_delta[-1, :] - sol_minus_delta[-1, :]) / (2 * delta_jacobian_scipy)

    return Jacobian_scipy

# Compute PyTorch Jacobians
def compute_jacobian_torch():
    initial_conditions_torch = torch.tensor(initial_conditions, requires_grad=True)
    a_torch = torch.tensor(a, requires_grad=True)
    t_torch = torch.linspace(tmin, tmax, nt)
    Jacobian_torch_fn = torch.autograd.functional.jacobian(
        lambda a: torch_odeint(ODEFunc(a), initial_conditions_torch, t_torch)[-1],
        a_torch
    )
    Jacobian_torch = Jacobian_torch_fn.detach().numpy()
    return Jacobian_torch

# Compute JAX Jacobians
def compute_jacobian_jax():
    initial_conditions_jax = jnp.array(initial_conditions, dtype=jnp.float64)
    a_jax = jnp.array(a, dtype=jnp.float64)
    t_jax = jnp.linspace(tmin, tmax, nt)
    system_jit = jit(system)
    Jacobian_jax_fn = jax_jacobian(lambda a: jax_odeint(system_jit, initial_conditions_jax, t_jax, a)[-1])
    Jacobian_jax = Jacobian_jax_fn(a_jax)
    return Jacobian_jax
