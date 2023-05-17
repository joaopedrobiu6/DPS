import os
import time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from solver import solve_with_scipy, solve_with_pytorch, solve_with_jax
from jacobi import compute_jacobian_scipy, compute_jacobian_torch, compute_jacobian_jax
from differences import compute_diff
from params_and_model import (
    initial_conditions, tmin, tmax, nt, a_initial, model, iota, Lambda,
    solver_models, variables, label_styles
)
from jax.config import config
config.update("jax_enable_x64", True)

def compute_and_time(func):
    start = time.time()
    result = func()
    return result, time.time() - start

def main(results_path='results'):
    solver_functions = {
        "Scipy": (solve_with_scipy, compute_jacobian_scipy),
        "PyTorch": (solve_with_pytorch, compute_jacobian_torch),
        "JAX": (solve_with_jax, compute_jacobian_jax)
    }

    solvers = [solver_functions.get(solver, (None, None))[0] for solver in solver_models]
    jacobi_computers = [solver_functions.get(solver, (None, None))[1] for solver in solver_models]

    w_solvers, times_solve = zip(*[compute_and_time(solver) for solver in solvers])
    Jacobian_solvers, times_jacobi = zip(*[compute_and_time(jacobian) for jacobian in jacobi_computers])

    diffs = compute_diff(solvers, jacobi_computers)

    t = np.linspace(tmin, tmax, nt)

    for i, func in enumerate(variables):
        plt.figure()
        for w_i, label, ls in zip(w_solvers, solver_models, label_styles):
            w_i_val = w_i[:, i].detach().numpy() if label == 'PyTorch' else w_i[:, i]
            plt.plot(t, w_i_val, ls[0], label=f'{label} {func}')
        plt.xlabel('Time')
        plt.ylabel(f'{func}')
        plt.title('ODE Solutions')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(results_path, f'initial_solution_timetrace_{model}_{func}.png'))

    plt.figure()
    for Jacobian_i, label, ls in zip(Jacobian_solvers, solver_models, label_styles):
        for i, func in enumerate(variables):
            plt.plot(a_initial, Jacobian_i[i, :], ls[1], label=f'{label} Jacobian {func}', markersize=10)
    plt.xlabel('Parameter a')
    plt.ylabel('Jacobian Value')
    plt.title('Jacobians with respect to a')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_path, f'jacobians_a_{model}.png'))

    fig = plt.figure()
    if model == 'lorenz':
        ax = fig.add_subplot(111, projection='3d')
        for w_i, label, ls in zip(w_solvers, solver_models, label_styles):
            w_i_val = w_i.detach().numpy() if label == 'PyTorch' else w_i
            ax.plot(w_i_val[:, 0], w_i_val[:, 1], w_i_val[:, 2], ls[0], label=f'{label}')
        ax.set_xlabel(variables[0])
        ax.set_ylabel(variables[1])
        ax.set_zlabel(variables[2])
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(results_path, f'initial_solution_3D_{model}.png'))
    elif model == 'guiding-center':
        plt.figure()
        for w_i, label, ls in zip(w_solvers, solver_models, label_styles):
            w_i_val = w_i.detach().numpy() if label == 'PyTorch' else w_i
            alpha = w_i_val[:, 1] - iota * w_i_val[:, 2]
            x = w_i_val[:, 0]
            plt.plot(np.sqrt(x) * np.cos(alpha), np.sqrt(x) * np.sin(alpha), ls[0], label=f'{label} {func}')
        plt.xlabel(f'sqrt(psi)*cos(alpha)')
        plt.ylabel(f'sqrt(psi)*sin(alpha)')
        plt.title('ODE Solutions')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(results_path, f'initial_solution_2D_{model}.png'))

        plt.figure()
        for w_i, label, ls in zip(w_solvers, solver_models, label_styles):
            w_i_val = w_i.detach().numpy() if label == 'PyTorch' else w_i
            x = w_i_val[:, 0]
            y = w_i_val[:, 1]
            z = w_i_val[:, 2]
            if w_i_val.shape[1] == 4:
                v_parallel = w_i_val[:, 3]
                plt.plot(t, v_parallel, ls[0], label=f'{label} {func}')
            B_val = a_initial[0] + a_initial[1] * np.sqrt(x) * np.cos(y) + a_initial[2] * np.sin(z)
            v_parallel_analytical = np.sqrt(1 - Lambda * B_val)
            plt.plot(t, v_parallel_analytical, ls[1], label=f'{label} {func} analytical')
        plt.xlabel('Time')
        plt.ylabel(f'v_parallel')
        plt.title('ODE Solutions')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(results_path, f'initial_solution_vparallel_{model}.png'))

        plt.figure()
        for w_i, label, ls in zip(w_solvers, solver_models, label_styles):
            w_i_val = w_i.detach().numpy() if label == 'PyTorch' else w_i
            x = w_i_val[:, 0]
            y = w_i_val[:, 1]
            z = w_i_val[:, 2]
            B_val = a_initial[0] + a_initial[1] * np.sqrt(x) * np.cos(y) + a_initial[2] * np.sin(z)
            plt.plot(t, B_val, ls[0], label=f'{label} {func}')
        plt.xlabel('Time')
        plt.ylabel(f'B')
        plt.title('ODE Solutions')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(results_path, f'initial_solution_B_{model}.png'))


    print(f'All plots saved to results folder {results_path}.')

    print('Analyzing differences between solutions and Jacobians:')
    for diff_label, diff_values in diffs.items():
        print(f"  Relative difference between {diff_label} solutions: {diff_values[0]:.4e}")
        print(f"  Relative difference between {diff_label} Jacobians: {diff_values[1]:.4e}")
        print(f"  -----------------------------")

    print('Analyzing time taken for solving ODE and computing Jacobian:')
    for time_solve, time_jacobian, label in zip(times_solve, times_jacobi, solver_models):
        print(f"  Time taken by {label} solve: {time_solve:.4f} seconds")
        print(f"  Time taken by {label} jacobian: {time_jacobian:.4f} seconds")
        print(f"  -----------------------------")

if __name__ == "__main__":
    this_path = str(Path(__file__).parent.resolve())
    results_path = os.path.join(this_path, 'results')
    os.makedirs(results_path, exist_ok=True)
    main(results_path)
