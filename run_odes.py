import os
import time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from src.solver import solve_with_scipy, solve_with_pytorch, solve_with_jax
from src.jacobi import compute_jacobian_scipy, compute_jacobian_torch, compute_jacobian_jax
from src.differences import compute_diff
from params_and_model import (tmin, tmax, nt, a_initial, model, solver_models, variables, label_styles)

if model == 'guiding-center':
    from params_and_model import B, Lambda

from jax.config import config
config.update("jax_enable_x64", True)

# This runs the solver and how long does it take to solve the system
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
    
    # Selects the models from solver_models if they are also in solver_functions
    selected_solver_models = [model for model in solver_models if model in solver_functions]
    # Makes solver_functions equal to the selected ones 
    solver_functions = {model: solver_functions[model] for model in selected_solver_models}

    # Separates the ODE solvers from the Jacobian calculation functions
    # Gets the solver functions - solve_with_scipy, solve_with_pytorch, solve_with_jax 
    solvers = [solver_functions.get(solver, (None, None))[0] for solver in solver_models] 
    # Gets the jacobian functions - compute_jacobian_scipy, compute_jacobian_pytorch, compute_jacobian_jax
    jacobi_computers = [solver_functions.get(solver, (None, None))[1] for solver in solver_models]
    
    # Runs the solvers and jacobian calculators and unzips them
    w_solvers, times_solve = zip(*[compute_and_time(solver) for solver in solvers])
    Jacobian_solvers, times_jacobi = zip(*[compute_and_time(jacobian) for jacobian in jacobi_computers])

    # ? não percebi muito bem
    diffs = compute_diff(w_solvers, Jacobian_solvers, list(solver_functions.keys()))

    t = np.linspace(tmin, tmax, nt)

    # Plot each component of the solution in time
    for i, func in enumerate(variables):
        plt.figure()
        for w_i, label, ls in zip(w_solvers, solver_models, label_styles):
            w_i_val = w_i.detach().numpy() if label == 'PyTorch' else w_i
            plt.plot(t, w_i_val[:, i], ls[0], label=f'{label} {func}')
            if model=='guiding-center' and i==3:
                B_val = B(a_initial, w_i_val[:, 0], w_i_val[:, 1], w_i_val[:, 2])
                v_parallel_analytical = np.sign(w_i_val[:, 3])*np.sqrt(1 - Lambda * B_val)
                plt.plot(t, v_parallel_analytical, ls[1], label=f'{label} {func} analytical')
        plt.xlabel('Time')
        plt.ylabel(f'{func}')
        plt.title('ODE Solutions')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(results_path, f'initial_solution_timetrace_{model}_{func}.png'))
    plt.figure()
    
    # Plot the  jacobians calculated with the different packages
    for Jacobian_i, label, ls in zip(Jacobian_solvers, solver_models, label_styles):
        for i, func in enumerate(variables):
            plt.plot(a_initial, Jacobian_i[i, :], ls[1], label=f'{label} Jacobian {func}', markersize=10)
    plt.xlabel('Parameter a')
    plt.ylabel('Jacobian Value')
    plt.title('Jacobians with respect to a')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_path, f'jacobians_a_{model}.png'))
    plt.figure()
    
    bar_width = 0.35
    index = np.arange(len(solver_models))
    bars1 = plt.bar(index, times_solve, bar_width, label='Solve Time')
    bars2 = plt.bar(index + bar_width, times_jacobi, bar_width, label='Jacobian Time')
    plt.xlabel('Library')
    plt.ylabel('Time (s)')
    plt.title('Time Taken for Solving ODE and Computing Jacobian')
    plt.xticks(index + bar_width / 2, solver_models)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_path, f'time_solve_Jacobian_{model}.png'))

    if model == 'lorenz':
        fig = plt.figure()
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
            plt.plot(np.sqrt(w_i_val[:, 0]) * np.cos(w_i_val[:, 1]), np.sqrt(w_i_val[:, 0]) * np.sin(w_i_val[:, 1]), ls[0], label=f'{label}')
        plt.xlabel(f'sqrt(psi)*cos(theta)')
        plt.ylabel(f'sqrt(psi)*sin(theta)')
        plt.title('ODE Solutions')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(results_path, f'initial_solution_2D_{model}.png'))

        plt.figure()
        for w_i, label, ls in zip(w_solvers, solver_models, label_styles):
            w_i_val = w_i.detach().numpy() if label == 'PyTorch' else w_i
            B_val = B(a_initial, w_i_val[:, 0], w_i_val[:, 1], w_i_val[:, 2])
            plt.plot(t, B_val, ls[0], label=f'{label}')
        plt.xlabel('Time')
        plt.ylabel(f'B')
        plt.title('ODE Solutions')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(results_path, f'initial_solution_B_{model}.png'))

        plt.figure()
        relative_energy_threshold = 1e-12
        for w_i, label, ls in zip(w_solvers, solver_models, label_styles):
            w_i_val = w_i.detach().numpy() if label == 'PyTorch' else w_i
            B_val = B(a_initial, w_i_val[:, 0], w_i_val[:, 1], w_i_val[:, 2])
            energy = w_i_val[:, 3] * w_i_val[:, 3] + Lambda * B_val
            energy_diff = np.abs((energy - energy[0]) / energy[0])

            filtered_indices = np.where(energy_diff > relative_energy_threshold)[0]
            t_filtered, energy_diff_filtered = t[filtered_indices], energy_diff[filtered_indices]

            plt.plot(t_filtered, energy_diff_filtered, ls[0], label=f'{label}')
            print(f'Max relative energy difference for {label}:', np.max(energy_diff))

        plt.xlabel('Time')
        plt.ylabel('Relative Energy Error')
        plt.yscale('log')
        plt.title('ODE Solutions')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(results_path, f'initial_solution_energy_{model}.png'))



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
