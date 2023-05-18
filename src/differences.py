# differences.py
import numpy as np
from itertools import combinations

def compute_diff(w_solvers, Jacobian_solvers, solvers):
    diffs = {}
    for (w_i, Jacobian_i, solver_i), (w_j, Jacobian_j, solver_j) in combinations(zip(w_solvers, Jacobian_solvers, solvers), 2):
        w_i_val = w_i.detach().numpy() if solver_i == 'PyTorch' else w_i
        w_j_val = w_j.detach().numpy() if solver_j == 'PyTorch' else w_j
        diff_w = np.linalg.norm(w_i_val - w_j_val) / np.linalg.norm(w_i_val)
        diff_jacobian = np.linalg.norm(Jacobian_i - Jacobian_j) / np.linalg.norm(Jacobian_i)
        solver_pair = f"{solver_i}-{solver_j}"
        diffs[solver_pair] = (diff_w, diff_jacobian)

    return diffs