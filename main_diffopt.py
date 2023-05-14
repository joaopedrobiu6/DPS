import os
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

from jax.config import config
config.update("jax_enable_x64", True)

# Import solvers, parameters and models from separate files
from params_and_model import tmin, tmax, nt, a_initial
from solver import solve_with_scipy, solve_with_pytorch, solve_with_jax
from optimizers import optimize_with_scipy, optimize_with_pytorch, optimize_with_jax

labels = ["Scipy", "PyTorch", "JAX"]

# Main function
def main(results_path='results'):
    scipy_optimized_a, time_scipy, loss_scipy = optimize_with_scipy(results_path)
    torch_optimized_a, time_pytorch, loss_pytorch = optimize_with_pytorch(results_path)
    jax_optimized_a, time_jax, loss_jax = optimize_with_jax(results_path)

    # Solve the ODE using PyTorch
    torch_initial = solve_with_pytorch(a_initial).detach().numpy()
    torch_solution = solve_with_pytorch(torch_optimized_a).detach().numpy()

    # Solve the ODE using Scipy
    scipy_initial = np.array(solve_with_scipy(a_initial))
    scipy_solution = np.array(solve_with_scipy(scipy_optimized_a))
    
    # Solve the ODE using JAX with optimized parameters
    jax_initial = solve_with_jax(a_initial).block_until_ready()
    jax_solution = solve_with_jax(jax_optimized_a).block_until_ready()

    # Plot initial figures
    plt.figure()
    t = np.linspace(tmin, tmax, nt)
    plt.plot(t, scipy_initial[:, 0], label='Scipy x')
    plt.plot(t, torch_initial[:, 0], label='PyTorch x')
    plt.plot(t, jax_initial[:, 0], label='JAX x')
    plt.plot(t, scipy_initial[:, 1], label='Scipy y')
    plt.plot(t, torch_initial[:, 1], label='PyTorch y')
    plt.plot(t, jax_initial[:, 1], label='JAX z')
    plt.plot(t, scipy_initial[:, 2], label='Scipy z')
    plt.plot(t, torch_initial[:, 2], label='PyTorch z')
    plt.plot(t, jax_initial[:, 2], label='JAX z')
    plt.xlabel('Time')
    plt.ylabel('Solution Value')
    plt.title('ODE Solutions')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_path, 'initial_solution_timetrace.png'))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(scipy_initial[:, 0], scipy_initial[:, 1], scipy_initial[:, 2], label='Scipy')
    ax.plot(torch_initial[:, 0], torch_initial[:, 1], torch_initial[:, 2], label='PyTorch')
    ax.plot(jax_initial[:, 0], jax_initial[:, 1], jax_initial[:, 2], label='JAX')
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('x3')
    plt.title('x1 vs x2 vs x3')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_path, 'initial_solution_3D.png'))

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

    plt.figure()
    p1 = plt.bar('Scipy', time_scipy, label=f'Scipy loss={loss_scipy:.4e}')
    p2 = plt.bar('PyTorch', time_pytorch, label=f'PyTorch loss={loss_pytorch:.4e}')
    p3 = plt.bar('JAX', time_jax, label=f'JAX loss={loss_jax:.4e}')
    plt.xlabel('Library')
    plt.ylabel('Time (s)')
    plt.tight_layout()
    plt.legend()
    plt.savefig(os.path.join(results_path, 'optimization_time.png'))

    print('All figures saved to results folder.')

if __name__ == "__main__":
    this_path = str(Path(__file__).parent.resolve())
    results_path = os.path.join(this_path, 'results')
    os.makedirs(results_path, exist_ok=True)
    main(results_path)