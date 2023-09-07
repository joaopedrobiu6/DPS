from desc.compute.utils import get_transforms, get_profiles, get_params, dot
from desc.compute import compute as compute_fun
from desc.backend import jnp
from desc.grid import Grid, LinearGrid
import desc.io
import desc.examples
from functools import partial
from jax import jit
from jax.experimental.ode import odeint as jax_odeint
import matplotlib.pyplot as plt
import numpy as np
import time
from desc.plotting import plot_surfaces, plot_section, plot_3d

# eq = desc.examples.get("DSHAPE")
eq = desc.io.load("../equilibria/test_run.h5")
eq._iota = eq.get_profile("iota").to_powerseries(order=eq.L, sym=True)
eq._current = None
eq.solve()

def plot_trajectory(solution, name):
    plt.plot(np.sqrt(solution[:, 0]) * np.cos(solution[:, 1]), np.sqrt(solution[:, 0]) * np.sin(solution[:, 1]))
    plt.xlabel(r'sqrt($\psi$)*cos($\theta$)')
    plt.ylabel(r'sqrt($\psi$)*sin($\theta$)')
    plt.title(rf'E = {E_}, q/m = {q_}/{m_}, $\Delta$t = [{t_i}, {t_f}], nt = {nt_}, x$_i$ = [{psi_i:.2f}, {theta_i:.2f}, {zeta_i:.2f}], v$_\%$ = {vpar_i_ratio:.2f}')
    plt.savefig("../results/DESC_tracing_" + name + ".png", dpi = 300)
    print("plot trajectory: saved as " + "results/DESC_tracing_" + name + ".png")

def plot_quantities(t_i, t_f, nt_, solution, name):
    t = np.linspace(t_i, t_f, nt_)
    fig, axs = plt.subplots(2, 2)
    axs[0, 1].plot(t, solution[:, 0], 'tab:orange')
    axs[0, 1].set_title(r'$\psi$ (t)')
    axs[1, 0].plot(t, solution[:, 1], 'tab:green')
    axs[1, 0].set_title(r'$\theta$ (t)')
    axs[1, 1].plot(t, solution[:, 2], 'tab:red')
    axs[1, 1].set_title(r'$\zeta$ (t)')
    axs[0, 0].plot(t, solution[:, 3], 'tab:blue')
    axs[0, 0].set_title(r"$v_{\parallel}$ (t)")

    fig = plt.gcf()
    fig.set_size_inches(10.5, 10.5)

    fig.savefig("../results/DESC_quantities_" + name + ".png", dpi = 300)
    print("plot quantities: saved as " + "results/DESC_quantities_" + name + ".png")

def plot_energy(t_i, t_f, nt_, solution, name):
    t = jnp.linspace(t_i, t_f, nt_)
    grid = Grid(np.vstack((np.sqrt(solution[:, 0]), solution[:, 1], solution[:, 2])).T,sort=False)
    B_field = eq.compute("|B|", grid=grid)
    Energy = 0.5*(sol[:, 3]**2 + 2*B_field["|B|"]*mu)*m
    plt.plot(t, Energy)
    plt.xlabel(r'time')
    plt.ylabel(r'Energy')
    plt.title("Energy Variation in Time")
    plt.savefig("../results/DESC_energy_" + name + ".png", dpi = 300)
    print("plot energy: saved as " + "results/DESC_energy_" + name + ".png")

def B_for_f_ratio_surface(psi_i):
    grid = LinearGrid(rho = np.sqrt(psi_i), M = 20, N = 20, NFP = eq.NFP)
    output = eq.compute("|B|", grid=grid)
    B = output["|B|"]
    return B

def B_for_f_ratio_fieldline(psi_i):
    coords = jnp.ones((250, 3)) #rho alpha zeta
    coords = coords.at[:, 0].set(coords[:, 0] * jnp.sqrt(psi_i))
    coords = coords.at[:, 2].set(jnp.linspace(0, 6 * jnp.pi, 250))

    start_time = time.time()
    print("starting map coords")
    print("--- %s seconds ---" % (time.time() - start_time))

    coords1 = eq.map_coordinates(
        coords=coords,
        inbasis=["rho", "alpha", "zeta"],
        outbasis=["rho", "theta", "zeta"],
        period=[jnp.inf, 2 * jnp.pi, jnp.inf],
        guess=None,
    )  # (2 * jnp.pi / eq.NFP)],

    grid = Grid(coords1, jitable=False, sort=False)
    output = eq.compute("|B|", grid=grid)

    B = output["|B|"]
    return B, coords1

def f_ratio(B):
    return np.sqrt(1-np.nanmin(B)/np.nanmax(B))

def check(quantity, rho_i, theta_i, zeta_i):
    grid = Grid(jnp.array([rho_i, theta_i, zeta_i]).T, jitable=True, sort=False)
    output = eq.compute(quantity, grid=grid)
    return output[quantity]

def psi_at_boundary():
    bound_psi = check("psi", 1, 0, 0)[0]
    return bound_psi
    
def system(w, t, a):
    #initial conditions
    psi, theta, zeta, vpar = w
    
    keys = ["psidot", "thetadot", "zetadot", "vpardot"] # etc etc, whatever terms you need
    grid = Grid(jnp.array([jnp.sqrt(psi), theta, zeta]).T, jitable=True, sort=False)
    transforms = get_transforms(keys, eq, grid, jitable=True)
    profiles = get_profiles(keys, eq, grid, jitable=True)
    params = get_params(keys, eq)
    data = compute_fun(eq, keys, params, transforms, profiles, mu = a[0], vpar = vpar, m_q = a[1])
    
    psidot = data["psidot"]
    thetadot = data["thetadot"]
    zetadot = data["zetadot"]
    vpardot = data["vpardot"]

    return jnp.array([psidot, thetadot, zetadot, vpardot])

def solve(E, charge, m, t_i, t_f, nt_ ,psi_i, theta_i, zeta_i, vpar_i_ratio):
    tmin = t_i
    tmax = t_f
    nt = nt_
    m_q = m/charge

    v_parallel = vpar_i_ratio*jnp.sqrt(2*E/m)
    
    grid = Grid(jnp.array([jnp.sqrt(psi_i), theta_i, zeta_i]).T, jitable=True, sort=False)
    data = eq.compute("|B|", grid=grid)

    mu = E/(m*data["|B|"]) - (v_parallel**2)/(2*data["|B|"])
    a_initial = [float(mu), m_q]
    initial_conditions = [psi_i, theta_i, zeta_i, v_parallel]

    def solve_with_jax(a=None):
        initial_conditions_jax = jnp.array(initial_conditions, dtype=jnp.float64)
        a_jax = jnp.array(a_initial, dtype=jnp.float64)
        t_jax = jnp.linspace(tmin, tmax, nt)
        system_jit = jit(system)
        solution_jax = jax_odeint(partial(system_jit, a=a_jax), initial_conditions_jax, t_jax)
        return solution_jax
    
    print("\n\nSOLVING...\n\n")

    sol = solve_with_jax()
    return sol, mu

E_ = 1
q_= 1
m_ = 1
t_i = 0 
t_f = 0.00007
nt_ = 50

charge = q_*1.6e-19
m = m_*1.673e-27
E = E_*3.52e6*charge
m_q = m/charge

psi_i = 0.7
theta_i = 0.2
zeta_i = 0.2
bound_psi = check("psi", 1, theta_i, zeta_i)[0]

f = f_ratio(B_for_f_ratio_surface(psi_i=psi_i))

print(psi_i, theta_i, zeta_i)

vpar_i_ratio = 0.7*f
print(vpar_i_ratio)
sol, mu = solve(E, charge, m, t_i, t_f, nt_, psi_i, theta_i, zeta_i, vpar_i_ratio)

plot_trajectory(solution=sol, name="trapped_particle")
plot_quantities(t_i=t_i, t_f=t_f, nt_=nt_, solution=sol, name="trapped")
plot_energy(t_i, t_f, nt_, sol, "energyplot")

print(sol.shape)
