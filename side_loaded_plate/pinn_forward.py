"""Backend supported: pytorch, jax, paddle

Implementation of the linear elasticity 2D example in paper https://doi.org/10.1016/j.cma.2021.113741.
References:
    https://github.com/sciann/sciann-applications/blob/master/SciANN-Elasticity/Elasticity-Forward.ipynb.
"""
import deepxde as dde
import numpy as np

x_max = 3.0
E = 210e3  # Young's modulus
nu = 0.3  # Poisson's ratio

lmbd = E * nu / ((1 + nu) * (1 - 2 * nu))  # Lame's first parameter
mu = E / (2 * (1 + nu))  # Lame's second parameter

# Load
m = 10
b = 50
def side_load(y):
    return m * y + b

# Define functions
sin = dde.backend.sin
cos = dde.backend.cos
stack = dde.backend.stack

geom = dde.geometry.Rectangle([0, 0], [x_max, x_max])
BC_type = ["hard", "soft"][0]


def boundary_left(x, on_boundary):
    return on_boundary and dde.utils.isclose(x[0], 0.0)


def boundary_right(x, on_boundary):
    return on_boundary and dde.utils.isclose(x[0], x_max)


def boundary_bottom(x, on_boundary):
    return on_boundary and dde.utils.isclose(x[1], 0.0)

# Soft Boundary Conditions
ux_left_bc = dde.icbc.DirichletBC(geom, lambda x: 0, boundary_left, component=0)
uy_bottom_bc = dde.icbc.DirichletBC(geom, lambda x: 0, boundary_bottom, component=1)
sxx_right_bc = dde.icbc.DirichletBC(geom, lambda x: side_load(x[:, 1]), boundary_right, component=2)


# Hard Boundary Conditions
def hard_BC(x, f, x_max=3.0):
    Ux = f[:, 0] * x[:, 0] 
    Uy = f[:, 1] * x[:, 0]

    Sxx = f[:, 2] * (x_max - x[:, 0]) + side_load(x[:, 1])
    Syy = f[:, 3] 
    Sxy = f[:, 4]
    return stack((Ux, Uy, Sxx, Syy, Sxy), axis=1)


# Load FEM reference solution
from scipy.interpolate import RegularGridInterpolator
data = np.loadtxt("fem_solution_200_points.dat")
X_val = data[:, :2]
u_val = data[:, 2:4]
stress_val = data[:, 7:10]

solution_val = np.hstack((u_val, stress_val))

n_mesh_points = int(np.sqrt(X_val.shape[0]))

# Interpolate solution
x_grid = np.linspace(0, 3, n_mesh_points)
y_grid = np.linspace(0, 3, n_mesh_points)

interpolators = []
for i in range(solution_val.shape[1]):
    interp = RegularGridInterpolator((x_grid, y_grid), solution_val[:, i].reshape(n_mesh_points, n_mesh_points).T)
    interpolators.append(interp)

solution_fn = lambda x, y: np.array([interp((x, y)) for interp in interpolators]).T


def jacobian(f, x, i, j):
    if dde.backend.backend_name == "jax":
        return dde.grad.jacobian(f, x, i=i, j=j)[0]
    else:
        return dde.grad.jacobian(f, x, i=i, j=j)


def pde(x, f):
    E_xx = jacobian(f, x, i=0, j=0)
    E_yy = jacobian(f, x, i=1, j=1)
    E_xy = 0.5 * (jacobian(f, x, i=0, j=1) + jacobian(f, x, i=1, j=0))

    S_xx = E_xx * (2 * mu + lmbd) + E_yy * lmbd
    S_yy = E_yy * (2 * mu + lmbd) + E_xx * lmbd
    S_xy = E_xy * 2 * mu

    Sxx_x = jacobian(f, x, i=2, j=0)
    Syy_y = jacobian(f, x, i=3, j=1)
    Sxy_x = jacobian(f, x, i=4, j=0)
    Sxy_y = jacobian(f, x, i=4, j=1)

    momentum_x = Sxx_x + Sxy_y 
    momentum_y = Sxy_x + Syy_y 

    if dde.backend.backend_name == "jax":
        f = f[0]  # f[1] is the function used by jax to compute the gradients

    stress_x = S_xx - f[:, 2:3]
    stress_y = S_yy - f[:, 3:4]
    stress_xy = S_xy - f[:, 4:5]

    return [momentum_x, momentum_y, stress_x, stress_y, stress_xy]


if BC_type == "hard":
    bcs = []
else:
    bcs = [
        sxx_right_bc,
        uy_bottom_bc,
        ux_left_bc,
    ]

data = dde.data.PDE(
    geom,
    pde,
    bcs,
    num_domain=500,
    num_boundary=500,
    num_test=100,
    solution=solution_fn,
)

layers = [2, [40] * 5, [40] * 5, [40] * 5, [40] * 5, 5]
activation = "tanh"
initializer = "Glorot uniform"
net = dde.nn.PFNN(layers, activation, initializer)
if BC_type == "hard":
    net.apply_output_transform(hard_BC)

model = dde.Model(data, net)
model.compile("adam", lr=0.001
losshistory, train_state = model.train(iterations=5000)

dde.saveplot(losshistory, train_state, issave=True, isplot=True)
