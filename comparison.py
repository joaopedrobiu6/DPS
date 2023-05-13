# comparison.py

import time
import numpy as np
import jax.numpy as jnp
from jax.experimental.ode import odeint
import torch
from torchdiffeq import odeint as torch_odeint
import matplotlib.pyplot as plt
from ode_system import initial_conditions, a, system, tmin, tmax, nt

# Solve the ODE using scipy's solve_ivp
print("Solving ODE using scipy...")
start_time = time.time()
sol_scipy = solve_ivp(lambda t, w: system(w, t, a), [tmin, tmax], initial_conditions, t_eval=np.linspace(tmin, tmax, nt))
scipy_solve_time = time.time() - start_time
print(f"Solving ODE using scipy took {scipy_solve_time} seconds")

# Solve the ODE using JAX's odeint
print("Solving ODE using JAX...")
start_time = time.time()
sol_jax = odeint(lambda w, t: jnp.array(system(w, t, a)), jnp.array(initial_conditions), jnp.linspace(tmin, tmax, nt))
jax_solve_time = time.time() - start_time
print(f"Solving ODE using JAX took {jax_solve_time} seconds")

# Solve the ODE using PyTorch's odeint
print("Solving ODE using PyTorch...")
start_time = time.time()
class ODEFunc(torch.nn.Module):
    def __init__(self, a):
        super(ODEFunc, self).__init__()
        self.a = torch.tensor(a, dtype=torch.float64)

    def forward(self, t, w):
        x, y = w[..., 0], w[..., 1]
        B_val = self.a[0] + self.a[1]*x + y*self.a[2]**2
        dxdt = x + y * B_val
        dydt = x - y * 2 * B_val
        return torch.stack([dxdt, dydt], dim=-1)

initial_conditions_torch = torch.tensor(initial_conditions, dtype=torch.float64)
sol_torch = torch_odeint(ODEFunc(a), initial_conditions_torch, torch.linspace(tmin, tmax, nt))
torch_solve_time = time.time() - start_time
print(f"Solving ODE using PyTorch took {torch_solve_time} seconds")

# Extract the solution
t = sol_scipy.t
x_scipy = sol_scipy.y[0]
y_scipy = sol_scipy.y[1]
x_jax = sol_jax[:, 0]
y_jax = sol_jax[:, 1]
x_torch = sol_torch.detach().numpy()[:, 0]
y_torch = sol_torch.detach().numpy()[:, 1]

# Plot the results
plt.figure()
plt.plot(t, x_scipy, label='x_scipy(t)')
plt.plot(t, y_scipy, label='y_scipy(t)')
plt.plot(t, x_jax, label='x_jax(t)')
plt.plot(t, y_jax, label='y_jax(t)')
plt.plot(t, x_torch, label='x_torch(t)')
plt.plot(t, y_torch, label='y_torch(t)')
plt.legend(loc='best')
plt.xlabel('t')
plt.ylabel('Solution')
plt.title('Comparison of ODE Solvers')
plt.grid(True)
plt.show()

# Print solve times
print(f"Solving ODE using scipy took {scipy_solve_time} seconds")
# print(f"Solving ODE using JAX took {jax_solve
