import os
import time
import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from torchdiffeq import odeint
from torch.optim import Adam

# Define a class for the system of ODEs
class ODEFunc(torch.nn.Module):
    def __init__(self, a):
        super(ODEFunc, self).__init__()
        self.a = torch.nn.Parameter(a)

    def forward(self, t, state):
        x, y = state[..., 0], state[..., 1]
        dx_dt = -x * y * self.a[0]
        dy_dt = (-x * self.a[1] - y) * self.a[0]
        return torch.stack([dx_dt, dy_dt], dim=-1)

def main(results_path='results'):
    # Initial conditions
    initial_conditions = [1.0, 1.0]
    initial_state = torch.tensor(initial_conditions, requires_grad=True)

    # Times for which the solution is to be computed
    t = torch.linspace(0.0, 1, 100)

    # Initial parameters
    a = [0.1, 0.2]
    a_torch = torch.tensor(a, requires_grad=True)

    # Define ODE system
    ode_system = ODEFunc(a_torch)

    # Optimizer
    optimizer = Adam(ode_system.parameters(), lr=1e-2)

    # Arrays to store optimization results
    losses = []
    parameters = []

    # Optimization loop
    start_time = time.time()
    for step in range(1000):
        optimizer.zero_grad()
        state = odeint(ode_system, initial_state, t)

        # Loss is the sum of squares of x and y at the final time step
        loss = torch.sum(state[-1, :] ** 2)

        loss.backward()
        optimizer.step()

        # Save optimization results
        losses.append(loss.item())
        parameters.append(ode_system.a.detach().numpy().copy())

        if step % 100 == 0:
            print(f'Step {step}, Loss {loss.item()}, Parameters {ode_system.a.data.numpy()}')

    elapsed_time = time.time() - start_time

    # Convert results to numpy arrays
    losses = np.array(losses)
    parameters = np.array(parameters)

    # Plot convergence
    plt.figure()
    plt.plot(losses)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title(f'Initial Parameters: {a}, Final Parameters: {ode_system.a.data.numpy()}')
    plt.savefig(os.path.join(results_path, 'differentiable_opt_convergence.png'))

    # Print optimization time
    print(f"Optimization Time: {elapsed_time} seconds")

    # Final solution x and y as a function of t
    plt.figure()
    final_state = state.detach().numpy()
    plt.plot(t.numpy(), final_state[:, 0], label='x (Final)')
    plt.plot(t.numpy(), final_state[:, 1], label='y (Final)')

    # Compute the initial state using the initial `a` values
    initial_a = np.array(a)
    initial_system = ODEFunc(torch.tensor(initial_a))
    initial_state = odeint(initial_system, initial_state, t, method='euler')
    initial_state = initial_state.detach().numpy()
    plt.plot(t.numpy(), initial_state[:, 0], '--', label='x (Initial)')
    plt.plot(t.numpy(), initial_state[:, 1], '--', label='y (Initial)')

    plt.xlabel('t')
    plt.ylabel('State')
    plt.title('Final and Initial Solutions: x and y')
    plt.legend()
    plt.savefig(os.path.join(results_path, 'differentiable_opt_solutions_timetrace.png'))

    # Plot x as a function of y
    plt.figure()
    plt.plot(final_state[:, 1], final_state[:, 0], label='Final')

    # Initial state x as a function of y
    plt.plot(initial_state[:, 1], initial_state[:, 0], '--', label='Initial')

    plt.xlabel('y')
    plt.ylabel('x')
    plt.title('Final and Initial x as a function of y')
    plt.legend()
    plt.savefig(os.path.join(results_path, 'differentiable_opt_solution_2D.png'))
    plt.show()


if __name__ == "__main__":
    this_path = str(Path(__file__).parent.resolve())
    results_path = os.path.join(this_path, 'results')
    os.makedirs(results_path, exist_ok=True)
    main(results_path)
