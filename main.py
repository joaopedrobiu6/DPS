import numpy as np
import torch
import time
from scipy.integrate import solve_ivp
from torchdiffeq import odeint
from jax.experimental.ode import odeint as jax_odeint
from jax import jit, jacobian as jax_jacobian
import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax.config import config

config.update("jax_enable_x64", True)

# Define parameters and initial conditions
initial_conditions = [0., 1.]
a = [0.1, 0.2, 0.3]
tmin = 0
tmax = 0.1
nt = 10
delta = 1e-5

# Define the system of equations
def B(a, x, y):
    return a[0] + a[1] * x + y * a[2] ** 2

@jit
def system(w, t, a):
    x, y = w
    B_val = B(a, x, y)
    return [x + y * B_val, x - y * 2 * B_val]

# Solve the ODE using Scipy
t = np.linspace(tmin, tmax, nt)
start_time = time.time()
sol = solve_ivp(lambda t, w: system(w, t, a), [tmin, tmax], initial_conditions, t_eval=t)
elapsed_time_ode_scipy = time.time() - start_time
w_scipy = sol.y

# Define the ODE function for PyTorch
class ODEFunc(torch.nn.Module):
    def __init__(self, a):
        super(ODEFunc, self).__init__()
        self.a = a

    def forward(self, t, w):
        x, y = w[..., 0], w[..., 1]
        B_val = self.a[0] + self.a[1] * x + y * self.a[2] ** 2
        dxdt = x + y * B_val
        dydt = x - y * 2 * B_val
        return torch.stack([dxdt, dydt], dim=-1)

# Solve the ODE using PyTorch
initial_conditions_torch = torch.tensor(initial_conditions, requires_grad=True)
a_torch = torch.tensor(a, requires_grad=True) 
t_torch = torch.linspace(tmin, tmax, nt)

start_time = time.time()
solution_torch = odeint(ODEFunc(a_torch), initial_conditions_torch, t_torch).detach().numpy()
elapsed_time_ode_torch = time.time() - start_time
w_torch = solution_torch

# Solve the ODE using JAX
initial_conditions_jax = jnp.array(initial_conditions, dtype=jnp.float64)
a_jax = jnp.array(a, dtype=jnp.float64)
t_jax = jnp.linspace(tmin, tmax, nt)

start_time = time.time()
system_jit = jit(system)
solution_jax = jax_odeint(system_jit, initial_conditions_jax, t_jax, a_jax)
elapsed_time_ode_jax = time.time() - start_time
w_jax = solution_jax

# Print elapsed times for ODE solvers
print(f"Scipy ODE solver time: {elapsed_time_ode_scipy:.6f} seconds")
print(f"PyTorch ODE solver time: {elapsed_time_ode_torch:.6f} seconds")
print(f"JAX ODE solver time: {elapsed_time_ode_jax:.6f} seconds")

# Compute differences between Scipy, PyTorch, and JAX solutions
diff_w_scipy_torch = np.linalg.norm(w_scipy.T - w_torch)
diff_w_scipy_jax = np.linalg.norm(w_scipy.T - w_jax)
diff_w_torch_jax = np.linalg.norm(w_torch - w_jax)

print(f"Difference between Scipy and PyTorch solutions: {diff_w_scipy_torch:.6f}")
print(f"Difference between Scipy and JAX solutions: {diff_w_scipy_jax:.6f}")
print(f"Difference between PyTorch and JAX solutions: {diff_w_torch_jax:.6f}")

# Plot trajectories
plt.figure(figsize=(10, 6))
plt.plot(t, w_scipy[0], label='Scipy x')
plt.plot(t, w_scipy[1], label='Scipy y')
plt.plot(t, w_torch[:, 0], label='PyTorch x')
plt.plot(t, w_torch[:, 1], label='PyTorch y')
plt.plot(t, w_jax[:, 0], label='JAX x')
plt.plot(t, w_jax[:, 1], label='JAX y')
plt.xlabel('Time')
plt.ylabel('Values')
plt.title('Trajectories of x and y')
plt.legend()
plt.show()

# Compute and plot Jacobians
Jacobian_scipy = np.empty((2, len(a)))
Jacobian_torch = np.empty((2, len(a)))
Jacobian_jax = np.empty((2, len(a)))

# Compute Scipy Jacobians
start_time = time.time()
for i in range(len(a)):
    a_plus_delta = a.copy()
    a_plus_delta[i] += delta
    a_minus_delta = a.copy()
    a_minus_delta[i] -= delta

    sol_plus_delta = solve_ivp(lambda t, w: system(w, t, a_plus_delta), [tmin, tmax], initial_conditions, t_eval=t)
    sol_minus_delta = solve_ivp(lambda t, w: system(w, t, a_minus_delta), [tmin, tmax], initial_conditions, t_eval=t)

    Jacobian_scipy[:, i] = (sol_plus_delta.y[:, -1] - sol_minus_delta.y[:, -1]) / (2 * delta)
elapsed_time_jacobian_scipy = time.time() - start_time

# Compute PyTorch Jacobians
start_time = time.time()
Jacobian_torch_fn = torch.autograd.functional.jacobian(
    lambda a: odeint(ODEFunc(a), initial_conditions_torch, t_torch)[-1],
    a_torch
)
Jacobian_torch = Jacobian_torch_fn.detach().numpy()
elapsed_time_jacobian_torch = time.time() - start_time

# Compute JAX Jacobians
start_time = time.time()
Jacobian_jax_fn = jax_jacobian(lambda a: jax_odeint(system_jit, initial_conditions_jax, t_jax, a)[-1])
Jacobian_jax = Jacobian_jax_fn(a_jax)
elapsed_time_jacobian_jax = time.time() - start_time

# Print elapsed times for Jacobian calculations
print(f"Scipy Jacobian computation time: {elapsed_time_jacobian_scipy:.6f} seconds")
print(f"PyTorch Jacobian computation time: {elapsed_time_jacobian_torch:.6f} seconds")
print(f"JAX Jacobian computation time: {elapsed_time_jacobian_jax:.6f} seconds")

# Compute differences between Scipy, PyTorch, and JAX Jacobians
diff_jacobian_scipy_torch = np.linalg.norm(Jacobian_scipy - Jacobian_torch)
diff_jacobian_scipy_jax = np.linalg.norm(Jacobian_scipy - Jacobian_jax)
diff_jacobian_torch_jax = np.linalg.norm(Jacobian_torch - Jacobian_jax)

print(f"Difference between Scipy and PyTorch Jacobians: {diff_jacobian_scipy_torch:.6f}")
print(f"Difference between Scipy and JAX Jacobians: {diff_jacobian_scipy_jax:.6f}")
print(f"Difference between PyTorch and JAX Jacobians: {diff_jacobian_torch_jax:.6f}")

# Plot Jacobians
plt.figure(figsize=(10, 6))
plt.plot(a, Jacobian_scipy[0, :], label='Scipy Jacobian x')
plt.plot(a, Jacobian_scipy[1, :], label='Scipy Jacobian y')
plt.plot(a, Jacobian_torch[0, :], label='PyTorch Jacobian x')
plt.plot(a, Jacobian_torch[1, :], label='PyTorch Jacobian y')
plt.plot(a, Jacobian_jax[0, :], label='JAX Jacobian x')
plt.plot(a, Jacobian_jax[1, :], label='JAX Jacobian y')
plt.xlabel('Parameter a')
plt.ylabel('Jacobian Value')
plt.title('Jacobians with respect to a')
plt.legend()

# Compute computation times
times_solve = [elapsed_time_ode_scipy, elapsed_time_ode_torch, elapsed_time_ode_jax]
times_jacobian = [elapsed_time_jacobian_scipy, elapsed_time_jacobian_torch, elapsed_time_jacobian_jax]

# Plot computation times
methods = ['Scipy', 'PyTorch', 'JAX']

fig, ax1 = plt.subplots(figsize=(10, 6))
ax2 = ax1.twinx()
ax1.bar(methods, times_solve, label='ODE Solve Time')
ax2.bar(methods, times_jacobian, label='Jacobian Calculation Time', alpha=0.5)
ax1.set_ylabel('Time (seconds)')
ax2.set_ylabel('Time (seconds)')
ax1.set_title('Computation Times for ODE Solvers')
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')
plt.show()
