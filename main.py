import time
import matplotlib.pyplot as plt
import numpy as np
from solver import solve_with_scipy, solve_with_pytorch, solve_with_jax
from jacobi import compute_jacobian_scipy, compute_jacobian_torch, compute_jacobian_jax
from differences import compute_diff
from params_and_model import tmin, tmax, nt, a
from jax.config import config

config.update("jax_enable_x64", True)

solvers = [solve_with_scipy, solve_with_pytorch, solve_with_jax]
jacobi_computers = [compute_jacobian_scipy, compute_jacobian_torch, compute_jacobian_jax]
labels = ["Scipy", "PyTorch", "JAX"]

def compute_and_time(func):
    start = time.time()
    result = func()
    return result, time.time() - start

def main():
    w, times_solve = zip(*[compute_and_time(solver) for solver in solvers])
    Jacobian, times_jacobi = zip(*[compute_and_time(jacobian) for jacobian in jacobi_computers])
    times = [time_solve + time_jacobi for time_solve, time_jacobi in zip(times_solve, times_jacobi)]
    diff_w_scipy_torch, diff_w_scipy_jax, diff_w_torch_jax, diff_jacobian_scipy_torch, diff_jacobian_scipy_jax, diff_jacobian_torch_jax = compute_diff()

    diffs = {
        "Scipy-PyTorch": [diff_w_scipy_torch, diff_jacobian_scipy_torch],
        "Scipy-JAX": [diff_w_scipy_jax, diff_jacobian_scipy_jax],
        "PyTorch-JAX": [diff_w_torch_jax, diff_jacobian_torch_jax]
    }

    t = np.linspace(tmin, tmax, nt)

    plt.figure(figsize=(10, 6))
    for w_i, label in zip(w, labels):
        if label == 'Scipy':
            plt.plot(t, w_i[0], label=f'{label} x')
            plt.plot(t, w_i[1], label=f'{label} y')
        else:
            plt.plot(t, w_i[:, 0], label=f'{label} x')
            plt.plot(t, w_i[:, 1], label=f'{label} y')
    plt.xlabel('Time')
    plt.ylabel('Solution Value')
    plt.title('ODE Solutions')
    plt.legend()

    plt.figure(figsize=(10, 6))
    for Jacobian_i, label in zip(Jacobian, labels):
        plt.plot(a, Jacobian_i[0, :], label=f'{label} Jacobian x')
        plt.plot(a, Jacobian_i[1, :], label=f'{label} Jacobian y')
    plt.xlabel('Parameter a')
    plt.ylabel('Jacobian Value')
    plt.title('Jacobians with respect to a')
    plt.legend()

    plt.figure(figsize=(10, 6))
    plt.bar(labels, times)
    plt.xlabel('Library')
    plt.ylabel('Time (s)')
    plt.title('Time taken for solving ODE and computing Jacobian')
    plt.show()

    for diff_label, diff_values in diffs.items():
        print(f"Difference between {diff_label} solutions: {diff_values[0]:.4e}")
        print(f"Difference between {diff_label} Jacobians: {diff_values[1]:.4e}")

    print('')

    for time_solve, time_jacobian, label in zip(times_solve, times_jacobi, labels):
        print(f"Time taken by {label} solve: {time_solve:.4f} seconds")
        print(f"Time taken by {label} jacobian: {time_jacobian:.4f} seconds")

if __name__ == "__main__":
    main()
