import os
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

from jax.config import config
config.update("jax_enable_x64", True)

# Import solvers, parameters and models from separate files
from params_and_model import tmin, tmax, nt
from solver import solve_with_scipy, solve_with_pytorch, solve_with_jax
from optimizers import optimize_with_scipy, optimize_with_pytorch, optimize_with_jax

labels = ["Scipy", "PyTorch", "JAX"]

# Main function
def main(results_path='results'):
    scipy_optimized_a = optimize_with_scipy(results_path)
    torch_optimized_a = optimize_with_pytorch(results_path)
    jax_optimized_a = optimize_with_jax(results_path)

    # Solve the ODE using PyTorch
    torch_solution = solve_with_pytorch(torch_optimized_a)

    # Solve the ODE using Scipy
    scipy_solution = solve_with_scipy(scipy_optimized_a)
    
    # Solve the ODE using JAX with optimized parameters
    jax_solution = solve_with_jax(jax_optimized_a)

    # Convert solutions to numpy arrays
    scipy_solution = np.array(scipy_solution)
    torch_solution = torch_solution.detach().numpy()
    jax_solution = jax_solution.block_until_ready()

    # Plot solutions
    plt.figure()
    t = np.linspace(tmin, tmax, nt)
    plt.plot(t, scipy_solution[:, 0], label='Scipy x')
    plt.plot(t, torch_solution[:, 0], label='PyTorch x')
    plt.plot(t, jax_solution[:, 0], label='JAX x')
    plt.plot(t, scipy_solution[:, 1], label='Scipy y')
    plt.plot(t, torch_solution[:, 1], label='PyTorch y')
    plt.plot(t, jax_solution[:, 1], label='JAX z')
    plt.plot(t, scipy_solution[:, 2], label='Scipy z')
    plt.plot(t, torch_solution[:, 2], label='PyTorch z')
    plt.plot(t, jax_solution[:, 2], label='JAX z')
    plt.xlabel('Time')
    plt.ylabel('Solution Value')
    plt.title('ODE Solutions')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_path, 'optimized_solution_timetrace.png'))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(scipy_solution[:, 0], scipy_solution[:, 1], scipy_solution[:, 2], label='Scipy')
    ax.plot(torch_solution[:, 0], torch_solution[:, 1], torch_solution[:, 2], label='PyTorch')
    ax.plot(jax_solution[:, 0], jax_solution[:, 1], jax_solution[:, 2], label='JAX')
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('x3')
    plt.title('x1 vs x2 vs x3')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_path, 'optimized_solution_3D.png'))

if __name__ == "__main__":
    this_path = str(Path(__file__).parent.resolve())
    results_path = os.path.join(this_path, 'results')
    os.makedirs(results_path, exist_ok=True)
    main(results_path)