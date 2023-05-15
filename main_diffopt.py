import os
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from jax.config import config
config.update("jax_enable_x64", True)

from params_and_model import tmin, tmax, nt, a_initial, initial_conditions
from solver import solve_with_scipy, solve_with_pytorch, solve_with_jax
from optimizers import optimize_with_scipy, optimize_with_pytorch, optimize_with_jax

labels = ["Scipy", "PyTorch", "JAX"]
label_styles = [['k-', 'k*'], ['r--', 'rx'], ['b-.', 'b+']]
num_functions = len(initial_conditions)
variables = ['x', 'y', 'z']

def main(results_path='results'):
    scipy_optimized_a, time_scipy, loss_scipy = optimize_with_scipy(results_path)
    torch_optimized_a, time_pytorch, loss_pytorch = optimize_with_pytorch(results_path)
    jax_optimized_a, time_jax, loss_jax = optimize_with_jax(results_path)

    solve_functions = [
        solve_with_scipy,
        lambda a: solve_with_pytorch(a).detach().numpy(),
        lambda a: solve_with_jax(a).block_until_ready()
    ]
    optimized_a = [scipy_optimized_a, torch_optimized_a, jax_optimized_a]

    initial_solutions = [solve_functions[i](a_initial) for i in range(3)]
    optimized_solutions = [solve_functions[i](optimized_a[i]) for i in range(3)]
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    t = np.linspace(tmin, tmax, nt)

    for i in range(4):
        row = i // 2
        col = i % 2

        if i < 3:
            for j in range(len(labels)):
                axes[row, col].plot(t, initial_solutions[j][:, i], label_styles[j][0], label=f'{labels[j]} Initial')
                axes[row, col].plot(t, optimized_solutions[j][:, i], label_styles[j][1], label=f'{labels[j]} Optimized')

            axes[row, col].set_xlabel('Time')
            axes[row, col].set_ylabel(f'{variables[i]}')
            axes[row, col].legend()
        else:
            ax_3d = fig.add_subplot(2, 2, 4, projection='3d')
            for j in range(len(labels)):
                ax_3d.plot(initial_solutions[j][:, 0],initial_solutions[j][:, 1],initial_solutions[j][:, 2], label_styles[j][0],label=f'{labels[j]} Initial')
                ax_3d.plot(optimized_solutions[j][:, 0],optimized_solutions[j][:, 1],optimized_solutions[j][:, 2], label_styles[j][1],label=f'{labels[j]} Optimized')
            ax_3d.set_xlabel(f'{variables[0]}')
            ax_3d.set_ylabel(f'{variables[1]}')
            ax_3d.set_zlabel(f'{variables[2]}')
            ax_3d.legend()

    # Remove unnecessary axes and spacing
    for ax in axes.ravel():
        if not ax.lines:
            ax.set_visible(False)
        ax.margins(0)
        ax.autoscale(enable=True, axis='x', tight=True)
        ax.autoscale(enable=True, axis='y', tight=True)

    # Adjust spacing and layout
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    # Save the figure
    plt.savefig(os.path.join(results_path, 'solution_comparison.png'))

    plt.figure(figsize=(8, 6))
    time_data = [time_scipy, time_pytorch, time_jax]
    loss_data = [loss_scipy[-1], loss_pytorch[-1], loss_jax[-1]]
    bars = plt.bar(labels, time_data)
    
    for bar, loss in zip(bars, loss_data):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height,
                 f'Loss: {loss:.4e}', ha='center', va='bottom')
    plt.xlabel('Library')
    plt.ylabel('Time (s)')
    plt.title('Optimization Time')
    plt.tight_layout()
    plt.savefig(os.path.join(results_path, 'optimization_time.png'))

    plt.figure(figsize=(8, 6))
    loss_data = [loss_scipy, loss_pytorch, loss_jax]
    [plt.plot(np.log(loss_data[i]), label_styles[i][1], label=labels[i]) for i in range(len(labels))]
    plt.xlabel('Iterations')
    plt.ylabel('Ln(Loss)')
    plt.tight_layout()
    plt.legend()
    plt.savefig(os.path.join(results_path, 'optimization_loss.png'))

    print('All figures saved to the results folder.')

if __name__ == "__main__":
    this_path = str(Path(__file__).parent.resolve())
    results_path = os.path.join(this_path, 'results')
    os.makedirs(results_path, exist_ok=True)
    main(results_path)
