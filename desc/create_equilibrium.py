from desc import set_device
set_device("gpu")
from desc.grid import Grid
from desc.plotting import plot_surfaces, plot_3d
from desc.geometry import FourierRZToroidalSurface
from desc.equilibrium import Equilibrium
from desc.continuation import solve_continuation_automatic

surf = FourierRZToroidalSurface(
    R_lmn=[1, 0.125, 0.1],
    Z_lmn=[-0.125, -0.1],
    modes_R=[[0, 0], [1, 0], [0, 1]],
    modes_Z=[[-1, 0], [0, -1]],
    NFP=4,
)
eq = Equilibrium(M=4, N=4, Psi=1, surface=surf)
eq = solve_continuation_automatic(eq, objective="force", bdry_step=0.5, verbose=3)[-1]
eq.save("new_equilibrium.h5")

fig1, ax1 = plot_surfaces(eq)
fig1.savefig("plt1.png")
fig2, ax2 = plot_surfaces(eq)
fig2.savefig("plt2.png")