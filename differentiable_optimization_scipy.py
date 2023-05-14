import os
import time
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.optimize import minimize

# Define the system of ODEs
def ode_system(state, t, a):
    x, y = state
    dx_dt = -x * y * a[0]
    dy_dt = (-x * a[1] - y) * a[0]
    return [dx_dt, dy_dt]

def main(results_path='results'):
    # Initial conditions
    initial_conditions = [1.0, 1.0]

    # Times for which the solution is to be computed
    t = np.linspace(0.0, 1.0, 100)

    # Initial parameters
    a_initial = [0.1, 0.2]

    # Define the objective function for optimization
    def objective(a):
        solution = odeint(ode_system, initial_conditions, t, args=(a,))
        final_state = solution[-1]
        return np.sum(final_state ** 2)

    # Perform optimization
    losses = []
    parameters = []

    def callback(xk):
        losses.append(objective(xk))
        parameters.append(xk)

    start_time = time.time()
    result = minimize(objective, a_initial, method='BFGS', callback=callback)
    optimized_a = result.x
    elapsed_time = time.time() - start_time

    # Compute the final solution with optimized parameters
    final_solution = odeint(ode_system, initial_conditions, t, args=(optimized_a,))
    final_state = final_solution.T

    # Compute the initial solution with initial parameters
    initial_solution = odeint(ode_system, initial_conditions, t, args=(a_initial,))
    initial_state = initial_solution.T

    # Plot convergence
    plt.figure()
    iterations = np.arange(len(losses))
    plt.plot(iterations, losses)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title(f'Initial Parameters: {a_initial}, Final Parameters: {optimized_a}')
    plt.savefig(os.path.join(results_path, 'simple_opt_convergence.png'))

    # Print optimization progress
    print("Optimization Progress:")
    for i in range(len(losses)):
        print(f"Iteration {i+1}, Loss: {losses[i]}, Parameters: {parameters[i]}")

    # Print optimization time
    print(f"Optimization Time: {elapsed_time} seconds")

    # Final solution x and y as a function of t
    plt.figure()
    plt.plot(t, final_state[0], label='x (Final)')
    plt.plot(t, final_state[1], label='y (Final)')
    plt.plot(t, initial_state[0], '--', label='x (Initial)')
    plt.plot(t, initial_state[1], '--', label='y (Initial)')
    plt.xlabel('t')
    plt.ylabel('State')
    plt.title('Final and Initial Solutions: x and y')
    plt.legend()
    plt.savefig(os.path.join(results_path, 'simple_opt_solutions_timetrace.png'))

    # Plot x as a function of y
    plt.figure()
    plt.plot(final_state[1], final_state[0], label='Final')
    plt.plot(initial_state[1], initial_state[0], '--', label='Initial')
    plt.xlabel('y')
    plt.ylabel('x')
    plt.title('Final and Initial x as a function of y')
    plt.legend()
    plt.savefig(os.path.join(results_path, 'simple_opt_solution_2D.png'))
    plt.show()


if __name__ == "__main__":
    this_path = str(Path(__file__).parent.resolve())
    results_path = os.path.join(this_path, 'results')
    os.makedirs(results_path, exist_ok=True)
    main(results_path)
