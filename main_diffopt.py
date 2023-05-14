import os
import time
import optax
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.optimize import minimize
import torch
import jax
from jax import jit, value_and_grad
import jax.numpy as jnp
from jax.experimental.ode import odeint as jax_odeint
from torchdiffeq import odeint as torch_odeint
from jax.config import config
config.update("jax_enable_x64", True)

# Import solvers, parameters and models from separate files
from params_and_model import system, ODEFunc, initial_conditions, a, tmin, tmax, nt
from solver import solve_with_scipy, solve_with_pytorch, solve_with_jax

# Perform the optimization using SciPy
def optimize_with_scipy():
    def objective(a):
        solution = solve_with_scipy(a)
        final_state = solution[:, -1]
        return np.sum(final_state ** 2)
    
    losses = []
    parameters = []

    def callback(xk):
        losses.append(objective(xk))
        parameters.append(xk)

    result = minimize(objective, a, method='BFGS', callback=callback)
    optimized_a = result.x
    return optimized_a, losses, parameters

# Perform the optimization using PyTorch
def optimize_with_pytorch():
    initial_conditions_torch = torch.tensor(initial_conditions, requires_grad=True)
    a_torch = torch.tensor(a, requires_grad=True)
    t_torch = torch.linspace(tmin, tmax, nt)

    class Loss(torch.nn.Module):
        def __init__(self):
            super(Loss, self).__init__()

        def forward(self, a):
            solution_torch = torch_odeint(ODEFunc(a), initial_conditions_torch, t_torch, method='rk4')
            final_state = solution_torch[-1]
            loss = torch.sum(final_state ** 2)
            return loss

    loss_func = Loss()
    optimizer = torch.optim.LBFGS([a_torch], lr=0.1)
    
    def closure():
        optimizer.zero_grad()
        loss = loss_func(a_torch)
        loss.backward()
        return loss

    losses = []
    parameters = []
    loss_old = optimizer.step(closure)
    for step in range(1000):
        loss = optimizer.step(closure)
        if loss_old - loss < 1e-5:
            break
        else:
            loss_old = loss
        params = a_torch.detach().numpy().copy()
        losses.append(loss.item())
        parameters.append(params)
        print(f'Step {step}, Loss {loss}, Parameters {params}')

    optimized_a = a_torch.detach().numpy().copy()
    return optimized_a, losses, parameters

# Perform the optimization using JAX
def optimize_with_jax():
    def loss_fn(a):
        solution_jax = solve_with_jax(a)
        final_state = solution_jax[-1]
        loss = jnp.sum(final_state ** 2)
        return loss

    a_initial = jnp.array(a, dtype=jnp.float64)
    optimizer = optax.adam(1e-3)
    optimizer_state = optimizer.init(a_initial)
    
    loss_grad_fn = jit(value_and_grad(loss_fn))

    params = a_initial
    loss_old, _ = loss_grad_fn(params)
    losses = []
    parameters = []
    for i in range(1000):
        loss, grads = loss_grad_fn(params)
        updates, optimizer_state = optimizer.update(grads, optimizer_state)
        params = optax.apply_updates(params, updates)

        # Save optimization results
        losses.append(loss)
        parameters.append(params)
        # if loss_old - loss < 1e-5:
        #     break
        # else:
        loss_old = loss
        print(f'Step {i}, Loss {loss}, Parameters {params}')

    optimized_a = params

    return optimized_a

# Main function
def main(results_path='results'):
    # # Solve the ODE using Scipy
    # scipy_solution = solve_with_scipy()

    # # Solve the ODE using PyTorch
    # torch_solution = solve_with_pytorch()

    # # Solve the ODE using JAX
    # jax_solution = solve_with_jax()

    # Optimize using SciPy
    scipy_optimized_a, losses_scipy, parameters_scipy = optimize_with_scipy()
    print("Optimization Progress:")
    for i in range(len(losses_scipy)):
        print(f"Iteration {i+1}, Loss: {losses_scipy[i]}, Parameters: {parameters_scipy[i]}")

    # Optimize using PyTorch
    torch_optimized_a = optimize_with_pytorch()

    # Optimize using JAX
    jax_optimized_a = optimize_with_jax()

    # Convert solutions to numpy arrays
    scipy_solution = np.array(scipy_solution)
    torch_solution = torch_solution.detach().numpy()
    jax_solution = jax_solution.block_until_ready()

    # Plot solutions
    plt.figure()
    t = np.linspace(tmin, tmax, nt)
    plt.plot(t, scipy_solution[0], label='Scipy')
    plt.plot(t, torch_solution[:, 0], label='PyTorch')
    plt.plot(t, jax_solution[:, 0], label='JAX')
    plt.xlabel('t')
    plt.ylabel('x')
    plt.title('Solution Comparison: x')
    plt.legend()
    plt.savefig(os.path.join(results_path, 'solution_comparison.png'))

    # Print optimized parameters
    print('Optimized Parameters:')
    print('SciPy:', scipy_optimized_a)
    print('PyTorch:', torch_optimized_a)
    print('JAX:', jax_optimized_a)

if __name__ == "__main__":
    this_path = str(Path(__file__).parent.resolve())
    results_path = os.path.join(this_path, 'results')
    os.makedirs(results_path, exist_ok=True)
    main(results_path)
