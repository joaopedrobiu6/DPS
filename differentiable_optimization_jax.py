import os
import time
import jax
import jax.numpy as jnp
import numpy as np
from jax import grad, jit, vmap, value_and_grad
import optax
from jax.experimental.ode import odeint
from pathlib import Path
import matplotlib.pyplot as plt

# Define a class for the system of ODEs
def ODEFunc(a, t, state):
    x, y = state[..., 0], state[..., 1]
    dx_dt = -x * y * a[0]
    dy_dt = (-x * a[1] - y) * a[0]
    return jnp.stack([dx_dt, dy_dt], axis=-1)

def main(results_path='results'):
    # Initial conditions
    initial_conditions = jnp.array([1.0, 1.0])

    # Times for which the solution is to be computed
    t = jnp.linspace(0.0, 1, 100).astype(jnp.float32)

    # Initial parameters
    params = jnp.array([0.1, 0.2])

    # Optimizer
    optimizer = optax.adam(1e-3)
    optimizer_state = optimizer.init(params)

    # Define loss function and its gradient
    @jit
    def loss_fn(params):
        state = odeint(lambda y, t: ODEFunc(params, t, y), initial_conditions, t)
        loss = jnp.sum(state[-1, :] ** 2)
        return loss

    loss_grad_fn = jit(value_and_grad(loss_fn))

    # Arrays to store optimization results
    losses = []
    parameters = []

    # Optimization loop
    start_time = time.time()
    for i in range(1000):
        loss, grads = loss_grad_fn(params)
        updates, optimizer_state = optimizer.update(grads, optimizer_state)
        params = optax.apply_updates(params, updates)

        # Save optimization results
        losses.append(loss)
        parameters.append(params)

        if i % 100 == 0:
            print(f'Step {i}, Loss {loss}, Parameters {params}')

    elapsed_time = time.time() - start_time

    # Convert results to numpy arrays
    losses = np.array(losses)
    parameters = np.array(parameters)

    # Plot convergence
    plt.figure()
    plt.plot(losses)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title(f'Initial Parameters: {params}, Final Parameters: {params}')
    plt.savefig(os.path.join(results_path, 'differentiable_opt_convergence.png'))

    # Print optimization time
    print(f"Optimization Time: {elapsed_time} seconds")

    # Final solution x and y as a function of t
    plt.figure()
    final_state = odeint(lambda y, t: ODEFunc(params, t, y), initial_conditions, t)
    plt.plot(t, final_state[:, 0], label='x (Final)')
    plt.plot(t, final_state[:, 1], label='y (Final)')

    # Compute the initial state using the initial `a` values
    initial_state = odeint(lambda y, t: ODEFunc(params, t, y), initial_conditions, t)
    plt.plot(t, initial_state[:, 0], '--', label='x (Initial)')
    plt.plot(t, initial_state[:, 1], '--', label='y (Initial)')

    plt.xlabel('t')
    plt.ylabel('State')
    plt.title('Final and Initial Solutions: x and y')
    plt.legend()
    plt.savefig(os.path.join(results_path, 'differentiable_opt_solutions_timetrace.png'))

    # Plot x as a function of y
    plt.figure()
    plt.plot(final_state[:, 1], final_state[:, 0], label='Final')

    # Initial state x as a function of y
    plt.plot(initial_state[:, 1], initial_state[:, 0], '--', label='Initial')

    plt.xlabel('y')
    plt.ylabel('x')
    plt.title('Final and Initial x as a function of y')
    plt.legend()
    plt.savefig(os.path.join(results_path, 'differentiable_opt_solution_2D.png'))
    plt.show()


if __name__ == "__main__":
    this_path = str(Path(__file__).parent.resolve())
    results_path = os.path.join(this_path, 'results')
    os.makedirs(results_path, exist_ok=True)
    main(results_path)
