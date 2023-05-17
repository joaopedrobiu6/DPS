# differences.py
import numpy as np

def compute_diff(solvers, jacobi_computers):
    w_solvers = [solver() for solver in solvers]
    Jacobian_solvers = [jacobi_computer() for jacobi_computer in jacobi_computers]

    diffs = {}
    for i in range(len(w_solvers)):
        for j in range(i + 1, len(w_solvers)):
            diff_w = np.linalg.norm(w_solvers[i] - w_solvers[j]) / np.linalg.norm(w_solvers[i])
            diff_jacobian = np.linalg.norm(Jacobian_solvers[i] - Jacobian_solvers[j]) / np.linalg.norm(Jacobian_solvers[i])
            solver_pair = f"{solvers[i].__name__}-{solvers[j].__name__}"
            diffs[solver_pair] = (diff_w, diff_jacobian)

    return diffs