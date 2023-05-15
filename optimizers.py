import os
import time
import optax
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, least_squares
from torchdiffeq import odeint as torch_odeint
import torch
from jax import jit, value_and_grad
import jax.numpy as jnp
from jax import grad
import jaxopt

# Import solvers, parameters and models from separate files
from params_and_model import (ODEFunc, a_initial, tol_optimization, max_nfev_optimization, tmin, tmax, nt,
                              delta_jacobian_scipy, initial_conditions, x_target, x_to_optimize,
                              learning_rate_torch, learning_rate_jax, n_steps_to_compute_loss,
                              use_scipy_torch, use_scipy_jax, step_optimization_verbose)
from solver import solve_with_scipy, solve_with_pytorch, solve_with_jax
num_functions = len(initial_conditions)  # assuming number of functions is the same as the length of initial conditions
t_torch = torch.linspace(tmin, tmax, nt)

def optimization_function_scipy(objective, method='L-BFGS-B'):
    return minimize(objective, a_initial, jac=True, method=method, options={'maxiter': max_nfev_optimization, 'gtol':tol_optimization}, tol=tol_optimization)

def save_results(losses, parameters, optimized_a, optimizer_name, results_path, time):

    plt.figure()
    for i in range(parameters.shape[1]):
        plt.plot(parameters[:, i]-parameters[0, i], label=f'Param {i+1} - {parameters[0, i]:.2e}')
    plt.xlabel('Step')
    plt.ylabel('Parameter - Initial value')
    plt.title(f'{optimizer_name.capitalize()} Optimization: Parameters over steps')
    plt.legend()
    plt.savefig(os.path.join(results_path, f'{optimizer_name}_parameters.png'))

    print(f'  Results with {optimizer_name} in {len(losses)} steps and {time:.4e} seconds:')
    print(f'    Parameters: {", ".join([f"{val:.4e}" for val in optimized_a])}')
    print(f'    Loss: {losses[-1]:.4e}')


# Perform the optimization using SciPy
def optimize_with_scipy(results_path):
    print('SciPy optimization')
    losses = []
    parameters = []
    def objective(a, info={'Nfeval':0}):
        initial_time = time.time()
        solution = solve_with_scipy(a)
        final_state = solution[-n_steps_to_compute_loss:, x_to_optimize]-x_target
        loss = np.sum(np.square(final_state))
        grad = np.empty((len(a_initial,)))
        for i in range(len(a_initial)):
            a_plus_delta = a.copy()
            a_plus_delta[i] += delta_jacobian_scipy
            sol_plus_delta = solve_with_scipy(a_plus_delta)
            loss_new = np.sum(np.square(sol_plus_delta[-n_steps_to_compute_loss:, x_to_optimize]-x_target))
            grad[i] = (loss_new - loss) / delta_jacobian_scipy
        losses.append(loss)
        parameters.append(a)
        if step_optimization_verbose:
            print('  step =', info['Nfeval'], f'loss = {loss.item():.3e} a =', a, f'took {(time.time()-initial_time):.3e} seconds')
        info['Nfeval'] += 1
        return loss, grad
    
    # Compute the Jacobian using finite differences
    def jacobian_central_differences(a):
        grad = np.empty((len(a_initial,)))
        for i in range(len(a_initial)):
            a_plus_delta = a.copy()
            a_plus_delta[i] += delta_jacobian_scipy
            a_minus_delta = a.copy()
            a_minus_delta[i] -= delta_jacobian_scipy
            sol_plus_delta = solve_with_scipy(a_plus_delta)
            sol_minus_delta = solve_with_scipy(a_minus_delta)
            loss_plus_delta = np.sum((sol_plus_delta[-n_steps_to_compute_loss:, x_to_optimize]-x_target) ** 2)
            loss_minus_delta = np.sum((sol_minus_delta[-n_steps_to_compute_loss:, x_to_optimize]-x_target) ** 2)
            grad[i] = (loss_plus_delta - loss_minus_delta) / (2 * delta_jacobian_scipy)
        return grad
    
    def residuals(a):
        solution = solve_with_scipy(a)
        final_state = solution[-n_steps_to_compute_loss:, x_to_optimize]-x_target
        loss = np.sum(final_state ** 2)
        losses.append(loss)
        parameters.append(a)
        return loss

    start_time = time.time()
    result = optimization_function_scipy(objective)
    # result = least_squares(residuals, a_initial, gtol=tol_optimization, max_nfev=max_nfev_optimization, verbose=0, jac=jacobian_central_differences)
    optimized_a = result.x
    elapsed_time = time.time() - start_time
    save_results(losses, np.array(parameters), optimized_a, 'scipy', results_path, elapsed_time)
    return optimized_a, elapsed_time, losses


# Perform the optimization using PyTorch
def optimize_with_pytorch(results_path):
    print('PyTorch optimization')
    a_torch = torch.tensor(a_initial, requires_grad=True) 
    ode_system = ODEFunc(a_torch)
    losses = []
    parameters = []
    start_time = time.time()

    if use_scipy_torch:
        ### USING SCIPY ###
        def objective(a, info={'Nfeval':0}):
            initial_time = time.time()
            ode_system.a.data = torch.tensor(a, requires_grad=True)
            solution_torch = solve_with_pytorch(a, ode_system)
            loss = torch.sum(torch.square(solution_torch[-n_steps_to_compute_loss:, x_to_optimize]-x_target))
            grads = torch.autograd.grad(loss, ode_system.a)
            loss_numpy = float(loss.detach().numpy())
            grads_numpy = np.array(grads[0].detach().numpy(), dtype=np.float64)
            losses.append(loss_numpy)
            parameters.append(a)
            if step_optimization_verbose:
                print('  step =', info['Nfeval'], f'loss = {loss.item():.3e} a =', a, f'took {(time.time()-initial_time):.3e} seconds')
            info['Nfeval'] += 1
            return loss_numpy, grads_numpy
        result = optimization_function_scipy(objective)
    else:
        ### USING PYTORCH OPTIMIZER ###
        # optimizer = torch.optim.RMSprop(ode_system.parameters(), lr=learning_rate_torch)
        # optimizer = torch.optim.SGD(ode_system.parameters(), lr=learning_rate_torch)
        # optimizer = torch.optim.Adam(ode_system.parameters(), lr=learning_rate_torch)
        # optimizer = torch.optim.LBFGS(ode_system.parameters(), lr=learning_rate_torch, max_iter=2, max_eval=3, tolerance_grad=1e-1, tolerance_change=1e-1, history_size=15, line_search_fn='strong_wolfe')
        # optimizer = torch.optim.RMSprop(ode_system.parameters(), lr=learning_rate_torch)
        optimizer = torch.optim.Adam(ode_system.parameters(), lr=learning_rate_torch)
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=1, factor=0.5)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

        def closure():
            optimizer.zero_grad()
            solution_torch = torch_odeint(ode_system, ode_system.a, t_torch)
            loss = torch.sum(torch.square(solution_torch[-n_steps_to_compute_loss:, x_to_optimize]-x_target))
            loss.backward()
            # print('a =',ode_system.a.detach().numpy(), 'loss =', loss.item())
            return loss
        for step in range(max_nfev_optimization):
            initial_time = time.time()
            loss = optimizer.step(closure)
            # scheduler.step(loss)
            scheduler.step()
            params = ode_system.a.detach().numpy().copy()
            losses.append(loss.item())
            parameters.append(params)
            if step_optimization_verbose:
                print('  step =', step, f'loss = {loss.item():.3e} a =', params, f'took {(time.time()-initial_time):.3e} seconds')
            if step > 4 and np.abs(losses[-2] - losses[-1])/losses[-2] < tol_optimization:
                break

    ### SAVE RESULTS ###
    optimized_a = ode_system.a.detach().numpy().copy()
    elapsed_time = time.time() - start_time
    save_results(losses, np.array(parameters), optimized_a, 'pytorch', results_path, elapsed_time)
    return optimized_a, elapsed_time, losses


# Perform the optimization using JAX
def optimize_with_jax(results_path):
    print('JAX optimization')
    def loss_fn(a):
        solution_jax = solve_with_jax(a)
        loss = jnp.sum(jnp.square(solution_jax[-n_steps_to_compute_loss:,x_to_optimize]-x_target))
        return loss
    a_initial_jax = jnp.array(a_initial)#, dtype=jnp.float64)
    loss_fn_jit = jit(loss_fn)
    losses=[loss_fn_jit(a_initial)]
    parameters=[a_initial_jax]
    start_time = time.time()

    if use_scipy_jax:
        ### USING SCIPY ###
        losses = []
        parameters = []
        loss_grad_fn = jit(value_and_grad(loss_fn))
        def objective(a):
            loss, grads = loss_grad_fn(a)
            losses.append(loss)
            parameters.append(a)
            return loss, grads
        result = optimization_function_scipy(objective)
    else:
        ### USING OPTAX ###
        # grad_loss_fn_jit = jit(grad(loss_fn))
        # # optimizer = optax.adabelief(learning_rate_jax)
        # warmup_cosine_decay_scheduler = optax.warmup_cosine_decay_schedule(init_value=learning_rate_jax, peak_value=learning_rate_jax*3,
        #                                                                 warmup_steps=int(max_nfev_optimization*0.1),
        #                                                                 decay_steps=max_nfev_optimization, end_value=learning_rate_jax/5)
        # optimizer = optax.rmsprop(learning_rate=warmup_cosine_decay_scheduler)#learning_rate_jax)
        # optimizer_state = optimizer.init(a_initial_jax)
        # params = a_initial_jax
        # for step in range(max_nfev_optimization):
        #     # loss, grads = loss_grad_fn(params)
        #     grads = grad_loss_fn_jit(params)
        #     updates, optimizer_state = optimizer.update(grads, optimizer_state)
        #     params = optax.apply_updates(params, updates)
        #     loss = loss_fn_jit(params)
        #     losses.append(loss)
        #     parameters.append(params)
        #     if step > 3 and np.abs(losses[-2] - losses[-1])/losses[-2] < tol_optimization:
        #         break

        ### USING JAXOPT ###
        loss_grad_fn = jit(value_and_grad(loss_fn))
        # optimizer = jaxopt.LBFGS(loss_grad_fn, value_and_grad=True, maxiter=max_nfev_optimization, tol=tol_optimization)
        # optimizer = jaxopt.GradientDescent(loss_grad_fn, value_and_grad=True, maxiter=max_nfev_optimization, tol=tol_optimization)
        # optimizer = jaxopt.NonlinearCG(loss_grad_fn, value_and_grad=True, maxiter=max_nfev_optimization)#, maxiter=max_nfev_optimization, tol=tol_optimization)
        # optimizer = jaxopt.ScipyMinimize(fun=loss_fn_jit, method='Newton-CG', tol=tol_optimization,maxiter=max_nfev_optimization, jit=True)#, options={'jac':True})
        optimizer = jaxopt.ScipyMinimize(fun=loss_fn_jit, method='L-BFGS-B', tol=tol_optimization,maxiter=max_nfev_optimization, jit=True)#, options={'jac':True})
        params, state = optimizer.run(a_initial_jax)
        losses = [loss_fn_jit(a_initial),loss_fn_jit(params)]
        parameters = [a_initial, params]

    ### SAVE RESULTS ###
    optimized_a = parameters[-1]
    elapsed_time = time.time() - start_time
    save_results(losses, np.array(parameters), optimized_a, 'jax', results_path, elapsed_time)
    return optimized_a, elapsed_time, losses