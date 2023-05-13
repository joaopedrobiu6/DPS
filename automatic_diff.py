from jax.experimental.ode import odeint
from jax import jit, grad, vmap, random, jacobian
import jax.numpy as jnp
import time
from ode_system import initial_conditions, a, system, tmin, tmax, nt

initial_conditions = jnp.array(initial_conditions, dtype=jnp.float64)
a = jnp.array(a, dtype=jnp.float64)

t = jnp.linspace(tmin, tmax, nt)

start_time = time.time()
system_jit = jit(system)
solution = odeint(system_jit, initial_conditions, t, a)
print(f"ODE solving time: {time.time() - start_time} seconds")

start_time = time.time()
jacob_fn = jit(jacobian(lambda a: odeint(system_jit, initial_conditions, t, a)[-1]))
jacob = jacob_fn(a)
print(f"Jacobian computation time: {time.time() - start_time} seconds")

print(f"Jacobian: \n{jacob}")
