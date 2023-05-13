# simple_ode_solver.py

from scipy.integrate import solve_ivp
import numpy as np
import matplotlib.pyplot as plt
from ode_system import initial_conditions, a, system, tmin, tmax, nt

# Solve the ODE using scipy's solve_ivp
sol = solve_ivp(lambda t, w: system(w, t, a), [tmin, tmax], initial_conditions, t_eval=np.linspace(tmin, tmax, nt))

# Extract the solution
t = sol.t
x = sol.y[0]
y = sol.y[1]

# Plot the results
plt.figure()
plt.plot(t, x, label='x(t)')
plt.plot(t, y, label='y(t)')
plt.legend(loc='best')
plt.xlabel('t')
plt.ylabel('Solution')
plt.title('Simple ODE Solver')
plt.grid(True)
plt.show()
