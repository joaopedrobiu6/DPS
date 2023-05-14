import torch
import jax.numpy as jnp
import numpy as np
from jax import jit
from scipy.integrate import solve_ivp, odeint
from torchdiffeq import odeint as torch_odeint
import torch
import torchode as to
from jax.experimental.ode import odeint as jax_odeint
from params_and_model import system, ODEFunc, initial_conditions, a, tmin, tmax, nt

# Solve the ODE using Scipy
def solve_with_scipy():
    t = np.linspace(tmin, tmax, nt)
    # sol = solve_ivp(lambda t, w: system(w, t, a), [tmin, tmax], initial_conditions, t_eval=t, method='RK45')
    # return sol.y
    sol = odeint(lambda w, t: system(w, t, a), initial_conditions, t)
    return sol.T  # note that odeint returns the transpose of the solution compared to solve_ivp

# Solve the ODE using PyTorch
def solve_with_pytorch():
    initial_conditions_torch = torch.tensor(initial_conditions, requires_grad=True)
    a_torch = torch.tensor(a, requires_grad=True)
    t_torch = torch.linspace(tmin, tmax, nt)
    solution_torch = torch_odeint(ODEFunc(a_torch), initial_conditions_torch, t_torch, method='rk4')#.detach().numpy()
    return solution_torch
# torchode solves ODEs in parallel but it is not working yet for python3.11
# def solve_with_pytorch():
#     initial_conditions_torch = torch.tensor(initial_conditions, requires_grad=True)
#     a_torch = torch.tensor(a, requires_grad=True)
#     t_torch = torch.linspace(tmin, tmax, nt)
#     # Define the system of equations
#     def f(t, y):
#         return ODEFunc(a_torch)(t, y)
#     # Create the ODE term, step method, and step size controller
#     term = to.ODETerm(f)
#     step_method = to.Dopri5(term=term)
#     step_size_controller = to.IntegralController(atol=1e-6, rtol=1e-3, term=term)
#     # Create the solver and compile it
#     solver = to.AutoDiffAdjoint(step_method, step_size_controller)
#     jit_solver = torch.compile(solver)
#     # Solve the ODE
#     sol = jit_solver.solve(to.InitialValueProblem(y0=initial_conditions_torch, t_eval=t_torch))
#     return sol.ys.detach().numpy()

# Solve the ODE using JAX
def solve_with_jax():
    initial_conditions_jax = jnp.array(initial_conditions, dtype=jnp.float64)
    a_jax = jnp.array(a, dtype=jnp.float64)
    t_jax = jnp.linspace(tmin, tmax, nt)
    system_jit = jit(system)
    solution_jax = jax_odeint(system_jit, initial_conditions_jax, t_jax, a_jax) # already uses a RK4 method
    return solution_jax
