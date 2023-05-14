import torch

# Parameters and initial conditions
initial_conditions = [5., 5., 5.]
a = [10, 20, 8/3] # sigma, rho, beta
tmin = 0
tmax = 10
nt = int(50*(tmax-tmin))
delta_jacobian_scipy = 1e-5

# Define the system of equations
def B(a, x, y, z):
    return a[0] + a[1] * x + y * a[2] ** 2 + z

def system(w, t, a):
    x, y, z = w
    # B_val = B(a, x, y, z)
    dxdt = a[0]*(y-x)
    dydt = x*(a[1]-z)-y
    dzdt = x*y-a[2]*z
    return [dxdt, dydt, dzdt]

class ODEFunc(torch.nn.Module):
    def __init__(self, a):
        super(ODEFunc, self).__init__()
        self.a = a

    def forward(self, t, w):
        x, y, z = w[..., 0], w[..., 1], w[..., 2]
        # B_val = self.a[0] + self.a[1] * x + y * self.a[2] ** 2 + z
        dxdt = self.a[0]*(y-x)
        dydt = x*(self.a[1]-z)-y
        dzdt = x*y-self.a[2]*z
        return torch.stack([dxdt, dydt, dzdt], dim=-1)
