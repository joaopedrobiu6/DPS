import os
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from jax.config import config

config.update("jax_enable_x64", True)

from params_and_model import tmin, tmax, nt, a_initial, initial_conditions, model as model_equations, solver_models, variables, label_styles
from src.solver import solve_with_scipy, solve_with_pytorch, solve_with_jax
from src.optimizers import optimize_with_scipy, optimize_with_pytorch, optimize_with_jax

num_functions = len(initial_conditions)


def main(results_path='results'):
    solve_functions = {
        "Scipy": solve_with_scipy,
        "PyTorch": solve_with_pytorch,
        "JAX": solve_with_jax
    }

    selected_solver_models = [model for model in solver_models if model in solve_functions]
    selected_solve_functions = {model: solve_functions[model] for model in selected_solver_models}

    optimized_a = []
    time_data = []
    loss_data = []
    whole_loss_data = []

    for model in selected_solver_models:
        if model == "Scipy":
            optimized_a_scipy, time_scipy, loss_scipy = optimize_with_scipy(results_path)
            optimized_a.append(optimized_a_scipy)
            time_data.append(time_scipy)
            loss_data.append(loss_scipy[-1])
            whole_loss_data.append(loss_scipy)
        elif model == "PyTorch":
            optimized_a_pytorch, time_pytorch, loss_pytorch = optimize_with_pytorch(results_path)
            optimized_a.append(optimized_a_pytorch)
            time_data.append(time_pytorch)
            loss_data.append(loss_pytorch[-1])
            whole_loss_data.append(loss_pytorch)
        elif model == "JAX":
            optimized_a_jax, time_jax, loss_jax = optimize_with_jax(results_path)
            optimized_a.append(optimized_a_jax)
            time_data.append(time_jax)
            loss_data.append(loss_jax[-1])
            whole_loss_data.append(loss_jax)

    initial_solutions = [selected_solve_functions[model](a_initial) for model in selected_solver_models]
    optimized_solutions = [selected_solve_functions[model](optimized_a[i]) for i, model in enumerate(selected_solver_models)]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    t = np.linspace(tmin, tmax, nt)

    for i, ax in enumerate(axes.flat):
        for j, model in enumerate(selected_solver_models):
            initial_solutions_val = initial_solutions[j].detach().numpy() if model == 'PyTorch' else initial_solutions[j]
            optimized_solutions_val = optimized_solutions[j].detach().numpy() if model == 'PyTorch' else optimized_solutions[j]
            if model_equations == 'lorenz':
                i_max = 3
            elif model_equations == 'guiding-center':
                i_max = 4
            elif model_equations == 'pendulum':
                i_max = 2
            if i < i_max:
                ax.plot(t, initial_solutions_val[:, i], label_styles[j][0], label=f'{model} Initial')
                ax.plot(t, optimized_solutions_val[:, i], label_styles[j][1], label=f'{model} Optimized')
                ax.set_xlabel('Time')
                ax.set_ylabel(f'{variables[i]}')
                ax.legend()
            else:
                if j==0:
                    ax_3d = fig.add_subplot(2, 2, 4, projection='3d')
                ax_3d.plot(initial_solutions_val[:, 0], initial_solutions_val[:, 1], initial_solutions_val[:, 2],
                        label_styles[j][0], label=f'{model} Initial')
                ax_3d.plot(optimized_solutions_val[:, 0], optimized_solutions_val[:, 1], optimized_solutions_val[:, 2],
                        label_styles[j][1], label=f'{model} Optimized')
                ax_3d.set_xlabel(f'{variables[0]}')
                ax_3d.set_ylabel(f'{variables[1]}')
                ax_3d.set_zlabel(f'{variables[2]}')
                ax_3d.legend()
    for ax in axes.ravel():
        if not ax.lines:
            ax.set_visible(False)
        ax.margins(0)
        ax.autoscale(enable=True, axis='x', tight=True)
        ax.autoscale(enable=True, axis='y', tight=True)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(results_path, f'solution_comparison_{model_equations}.png'))

    plt.figure(figsize=(8, 6))
    bars = plt.bar(solver_models, time_data)
    for bar, loss in zip(bars, loss_data):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height,
                 f'Loss: {loss:.4e}', ha='center', va='bottom')
    plt.xlabel('Library')
    plt.ylabel('Time (s)')
    plt.title('Optimization Time')
    plt.tight_layout()
    plt.savefig(os.path.join(results_path, f'optimization_time_{model_equations}.png'))

    plt.figure(figsize=(8, 6))
    [plt.plot(np.log(whole_loss_data[i]), label_styles[i][1], label=solver_models[i]) for i in range(len(solver_models))]
    plt.xlabel('Iterations')
    plt.ylabel('Ln(Loss)')
    plt.tight_layout()
    plt.legend()
    plt.savefig(os.path.join(results_path, f'optimization_loss_{model_equations}.png'))

    if model_equations == 'guiding-center':
        plt.figure(figsize=(8, 6))
        for initial_sols, optimized_sols, label, ls in zip(initial_solutions, optimized_solutions, solver_models, label_styles):
            initial_sols_val = initial_sols.detach().numpy() if label == 'PyTorch' else initial_sols
            optimized_sols_val = optimized_sols.detach().numpy() if label == 'PyTorch' else optimized_sols
            plt.plot(np.sqrt(initial_sols_val[:, 0]) * np.cos(initial_sols_val[:, 1]), np.sqrt(initial_sols_val[:, 0]) * np.sin(initial_sols_val[:, 1]), ls[0], label=f'{label} Initial')
            plt.plot(np.sqrt(optimized_sols_val[:, 0]) * np.cos(optimized_sols_val[:, 1]), np.sqrt(optimized_sols_val[:, 0]) * np.sin(optimized_sols_val[:, 1]), ls[1], label=f'{label} Optimized')
        plt.xlabel(f'sqrt(psi)*cos(theta)')
        plt.ylabel(f'sqrt(psi)*sin(theta)')
        plt.title('ODE Solutions')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(results_path, f'initial_vs_optimized_solution_2D_{model}.png'))

    print('All figures saved to the results folder.')

if __name__ == "__main__":
    this_path = str(Path(__file__).parent.resolve())
    results_path = os.path.join(this_path, 'results')
    os.makedirs(results_path, exist_ok=True)
    main(results_path)
