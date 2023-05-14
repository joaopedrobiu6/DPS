import os
import time
import optax
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, least_squares
import torch
from jax import jit
import jax.numpy as jnp
from jax import grad

# Import solvers, parameters and models from separate files
from params_and_model import ODEFunc, a_initial, tol_optimization, max_nfev_optimization
from solver import solve_with_scipy, solve_with_pytorch, solve_with_jax

def save_results(losses, parameters, optimized_a, optimizer_name, results_path, time):
    # Plot losses
    plt.figure()
    plt.plot(losses)
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.title(f'{optimizer_name.capitalize()} Optimization: Loss over steps')
    plt.savefig(os.path.join(results_path, f'{optimizer_name}_losses.png'))

    parameters = np.array(parameters)

    # Only plot parameters if there is more than one step
    if parameters.ndim > 1:
        # Plot parameters
        plt.figure()
        for i in range(parameters.shape[1]):
            plt.plot(parameters[:, i], label=f'Param {i+1}')
        plt.xlabel('Step')
        plt.ylabel('Parameter value')
        plt.title(f'{optimizer_name.capitalize()} Optimization: Parameters over steps')
        plt.legend()
        plt.savefig(os.path.join(results_path, f'{optimizer_name}_parameters.png'))

    print(f'Final results with {optimizer_name} in {len(losses)} steps and {time:.4e} seconds:')
    print(f'  Parameters: {", ".join([f"{val:.4e}" for val in optimized_a])}')
    print(f'  Loss: {losses[-1]:.4e}')



# Perform the optimization using SciPy
def optimize_with_scipy(results_path):
    def objective(a):
        solution = solve_with_scipy(a)
        final_state = solution[:, -1]
        return np.sum(final_state ** 2)
    
    losses = []
    parameters = []
    def residuals(a):
        solution = solve_with_scipy(a)
        final_state = solution[:, -1]
        loss = np.sum(final_state ** 2)
        losses.append(loss)
        parameters.append(a)
        return final_state

    def callback_minimize(xk):
        losses.append(objective(xk))
        parameters.append(xk)

    start_time = time.time()
    result = minimize(objective, a_initial, method='BFGS', callback=callback_minimize)
    # result = least_squares(residuals, a_initial, method='trf', ftol=tol_optimization, max_nfev = max_nfev_optimization, verbose=0)
    optimized_a = result.x
    save_results(losses, parameters, optimized_a, 'scipy', results_path, time.time() - start_time)
    return optimized_a


# Perform the optimization using PyTorch
def optimize_with_pytorch(results_path):

    a_torch = torch.tensor(a_initial, requires_grad=True) 
    ode_system = ODEFunc(a_torch)

    optimizer = torch.optim.LBFGS(ode_system.parameters(), lr=0.1)

    losses = []
    parameters = []

    start_time = time.time()
    for step in range(max_nfev_optimization):
        def closure():
            optimizer.zero_grad()
            solution_torch = solve_with_pytorch(a_torch.detach().numpy(), ode_system)
            loss = torch.sum(solution_torch[-1] ** 2)
            loss.backward()
            return loss

        loss = optimizer.step(closure)
        params = ode_system.a.detach().numpy().copy()
        losses.append(loss.item())
        parameters.append(params)
        if step > 3 and (losses[-2] - losses[-1])/losses[-2] < tol_optimization:
            break

    optimized_a = ode_system.a.detach().numpy().copy()
    save_results(losses, parameters, optimized_a, 'pytorch', results_path, time.time() - start_time)
    return optimized_a


# Perform the optimization using JAX
def optimize_with_jax(results_path):
    def loss_fn(a):
        solution_jax = solve_with_jax(a)
        loss = jnp.sum(solution_jax[-1] ** 2)
        return loss

    a_initial_jax = jnp.array(a_initial, dtype=jnp.float64)

    loss_fn_jit = jit(loss_fn)
    grad_loss_fn_jit = jit(grad(loss_fn))
    optimizer = optax.optimistic_gradient_descent(3e-1)
    optimizer_state = optimizer.init(a_initial_jax)

    # loss_grad_fn = jit(value_and_grad(loss_fn))

    params = a_initial_jax
    losses = []
    parameters = []

    start_time = time.time()
    for i in range(max_nfev_optimization):
        # loss, grads = loss_grad_fn(params)
        grads = grad_loss_fn_jit(params)
        updates, optimizer_state = optimizer.update(grads, optimizer_state)
        params = optax.apply_updates(params, updates)

        loss = loss_fn_jit(params)
        losses.append(loss)
        parameters.append(params)

        if i > 3 and (losses[-2] - losses[-1])/losses[-2] < tol_optimization:
            break

    optimized_a = params
    save_results(losses, parameters, optimized_a, 'jax', results_path, time.time() - start_time)
    return optimized_a