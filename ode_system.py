# ode_system.py

# Initial conditions x(0) = 0, y(0) = 1
initial_conditions = [0., 1.]

# Parameter 'a'
a = [0.5, 0.5, 0.5]

# Time points from tmin to tmax
tmin = 0
tmax = 0.1
nt = 5

# Function B
def B(a, x, y):
    return a[0] + a[1]*x + y*a[2]**2

# System of ODEs
def system(w, t, a):
    x, y = w
    B_val = B(a, x, y)
    return [x + y * B_val, x - y * 2 * B_val]
