# differences.py
import numpy as np
from solver import solve_with_scipy, solve_with_pytorch, solve_with_jax
from jacobi import compute_jacobian_scipy, compute_jacobian_torch, compute_jacobian_jax

def compute_diff():
    w_scipy = solve_with_scipy()
    w_torch = solve_with_pytorch().detach().numpy()
    w_jax = solve_with_jax()

    Jacobian_scipy = compute_jacobian_scipy()
    Jacobian_torch = compute_jacobian_torch()
    Jacobian_jax = compute_jacobian_jax()

    diff_w_scipy_torch = np.linalg.norm(w_scipy.T - w_torch) / np.linalg.norm(w_scipy)
    diff_w_scipy_jax = np.linalg.norm(w_scipy.T - w_jax) / np.linalg.norm(w_scipy)
    diff_w_torch_jax = np.linalg.norm(w_torch - w_jax) / np.linalg.norm(w_torch)

    diff_jacobian_scipy_torch = np.linalg.norm(Jacobian_scipy - Jacobian_torch) / np.linalg.norm(Jacobian_scipy)
    diff_jacobian_scipy_jax = np.linalg.norm(Jacobian_scipy - Jacobian_jax) / np.linalg.norm(Jacobian_scipy)
    diff_jacobian_torch_jax = np.linalg.norm(Jacobian_torch - Jacobian_jax) / np.linalg.norm(Jacobian_torch)

    diffs = {
        "scipy-torch": (diff_w_scipy_torch, diff_jacobian_scipy_torch),
        "scipy-jax": (diff_w_scipy_jax, diff_jacobian_scipy_jax),
        "torch-jax": (diff_w_torch_jax, diff_jacobian_torch_jax),
    }

    return diffs
