import os
import time
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from solver import solve_with_scipy, solve_with_pytorch, solve_with_jax
from jacobi import compute_jacobian_scipy, compute_jacobian_torch, compute_jacobian_jax
from differences import compute_diff
from params_and_model import initial_conditions, tmin, tmax, nt, a_initial, model, iota, vpar_sign, Lambda
# import torch
# torch.set_default_dtype(torch.float64)
from jax.config import config
config.update("jax_enable_x64", True)

solvers = [solve_with_scipy, solve_with_pytorch, solve_with_jax]
jacobi_computers = [compute_jacobian_scipy, compute_jacobian_torch, compute_jacobian_jax]
labels = ["Scipy", "PyTorch", "JAX"]
label_styles = [['k-','k*'], ['r--','rx'], ['b-.','b+']]
variables = ['x', 'y', 'z']
num_functions = len(initial_conditions)
functions = [f"x{i+1}" for i in range(num_functions)]

def compute_and_time(func):
    start = time.time()
    result = func()
    return result, time.time() - start

def main(results_path='results'):
    w, times_solve = zip(*[compute_and_time(solver) for solver in solvers])
    Jacobian, times_jacobi = zip(*[compute_and_time(jacobian) for jacobian in jacobi_computers])
    diffs = compute_diff()

    t = np.linspace(tmin, tmax, nt)

    for i, func in enumerate(functions):
        plt.figure()
        for w_i, label, ls in zip(w, labels, label_styles):
            if label == 'PyTorch':
                plt.plot(t, w_i[:, i].detach().numpy(), ls[0], label=f'{label} {func}')
            else:
                plt.plot(t, w_i[:, i], ls[0], label=f'{label} {func}')
        plt.xlabel('Time')
        plt.ylabel(f'{func}')
        plt.title('ODE Solutions')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(results_path, f'initial_solution_timetrace_{model}_{func}.png'))

    plt.figure()
    for Jacobian_i, label, ls in zip(Jacobian, labels, label_styles):
        for i, func in enumerate(functions):
            plt.plot(a_initial, Jacobian_i[i, :], ls[1], label=f'{label} Jacobian {func}', markersize=10)
    plt.xlabel('Parameter a')
    plt.ylabel('Jacobian Value')
    plt.title('Jacobians with respect to a')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_path, f'jacobians_a_{model}.png'))

    plt.figure()
    p1 = plt.bar(labels, times_solve, label='Solving ODE')
    p2 = plt.bar(labels, times_jacobi, bottom=times_solve, label='Computing Jacobian')
    plt.xlabel('Library')
    plt.ylabel('Time (s)')
    plt.tight_layout()
    plt.legend()
    plt.savefig(os.path.join(results_path, f'time_solveJacobian_{model}.png'))

    fig = plt.figure()
    if model == 'lorenz':
        ax = fig.add_subplot(111, projection='3d')
        for w_i, label, ls in zip(w, labels, label_styles):
            if label == 'PyTorch':
                ax.plot(w_i[:, 0].detach().numpy(), w_i[:, 1].detach().numpy(), w_i[:, 2].detach().numpy(), ls[0], label=f'{label}')
            else:
                ax.plot(w_i[:, 0], w_i[:, 1], w_i[:, 2], ls[0], label=f'{label}')
        ax.set_xlabel(variables[0])
        ax.set_ylabel(variables[1])
        ax.set_zlabel(variables[2])
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(results_path, f'initial_solution_3D_{model}.png'))
    elif model == 'guiding-center':
        plt.figure()
        for w_i, label, ls in zip(w, labels, label_styles):
            if label == 'PyTorch':
                alpha = w_i[:, 1].detach().numpy()-iota*w_i[:, 2].detach().numpy()
                x = w_i[:, 0].detach().numpy()
            else:
                alpha = w_i[:, 1]-iota*w_i[:, 2]
                x = w_i[:, 0]
            plt.plot(np.sqrt(x)*np.cos(alpha), np.sqrt(x)*np.sin(alpha), ls[0], label=f'{label} {func}')
        plt.xlabel(f'sqrt(psi)*cos(alpha)')
        plt.ylabel(f'sqrt(psi)*sin(alpha)')
        plt.title('ODE Solutions')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(results_path, f'initial_solution_2D_{model}.png'))

        plt.figure()
        for w_i, label, ls in zip(w, labels, label_styles):
            if label == 'PyTorch':
                x = w_i[:, 0].detach().numpy()
                y = w_i[:, 1].detach().numpy()
                z = w_i[:, 2].detach().numpy()
            else:
                x = w_i[:, 0]
                y = w_i[:, 1]
                z = w_i[:, 2]
            B_val = a_initial[0] + a_initial[1] * np.sqrt(x) * np.cos(y) + a_initial[2] * np.sin(z)
            v_parallel = vpar_sign*np.sqrt(1-Lambda*B_val)
            plt.plot(t, v_parallel, ls[0], label=f'{label} {func}')
        plt.xlabel('Time')
        plt.ylabel(f'v_parallel')
        plt.title('ODE Solutions')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(results_path, f'initial_solution_vparallel_{model}.png'))

        plt.figure()
        for w_i, label, ls in zip(w, labels, label_styles):
            if label == 'PyTorch':
                x = w_i[:, 0].detach().numpy()
                y = w_i[:, 1].detach().numpy()
                z = w_i[:, 2].detach().numpy()
            else:
                x = w_i[:, 0]
                y = w_i[:, 1]
                z = w_i[:, 2]
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

    print('')

    print('Analyzing time taken for solving ODE and computing Jacobian:')
    for time_solve, time_jacobian, label in zip(times_solve, times_jacobi, labels):
        print(f"  Time taken by {label} solve: {time_solve:.4f} seconds")
        print(f"  Time taken by {label} jacobian: {time_jacobian:.4f} seconds")
        print(f"  -----------------------------")

if __name__ == "__main__":
    this_path = str(Path(__file__).parent.resolve())
    results_path = os.path.join(this_path, 'results')
    os.makedirs(results_path, exist_ok=True)
    main(results_path)
