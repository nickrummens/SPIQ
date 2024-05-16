"""Backend supported: pytorch, paddle, jax

Implementation of the linear elasticity 2D example in paper https://doi.org/10.1016/j.cma.2021.113741.
References:
    https://github.com/sciann/sciann-applications/blob/master/SciANN-Elasticity/Elasticity-Forward.ipynb.
"""

import deepxde as dde
import numpy as np
import time
import os
from scipy.interpolate import RegularGridInterpolator
import argparse
import jax
import jax.numpy as jnp

parser = argparse.ArgumentParser(description='Physics Informed Neural Networks for Linear Elastic Plate')

parser.add_argument('--n_iter', type=int, default=10000000, help='Number of iterations')
parser.add_argument('--log_every', type=int, default=1000, help='Log every n steps')
parser.add_argument('--available_time', type=int, default=20, help='Available time in minutes')
parser.add_argument('--log_output_fields', nargs='+', default=['Ux', 'Uy', 'Sxx', 'Syy', 'Sxy'], help='Fields to log')
parser.add_argument('--net_type', choices=['spinn', 'pfnn'], default='spinn', help='Type of network')
parser.add_argument('--bc_type', choices=['hard', 'soft'], default='hard', help='Type of boundary condition')
parser.add_argument('--mlp', choices=['mlp', 'modified_mlp'], default='mlp', help='Type of MLP for SPINN')
parser.add_argument('--n_DIC', type=int, default=6, help='Number of DIC')
parser.add_argument('--noise_ratio', type=float, default=0, help='Noise ratio')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
parser.add_argument('--u_0', type=float, default=1e-4, help='Displacement scaling factor')
parser.add_argument('--loss_weights', nargs='+', type=float, default=[1,1,1,1,1,1e2], help='Loss weights (more on DIC points)')

args = parser.parse_args()

n_iter = args.n_iter
log_every = args.log_every
available_time = args.available_time
log_output_fields = {i: field for i, field in enumerate(args.log_output_fields)}
net_type = args.net_type
bc_type = args.bc_type
mlp = args.mlp
n_DIC = args.n_DIC
noise_ratio = args.noise_ratio
lr = args.lr
u_0 = args.u_0
loss_weights = args.loss_weights

if net_type == "spinn":
    dde.config.set_default_autodiff("forward")

x_max = 1.0
y_max = 4.0
E = 210e3  # Young's modulus
nu = 0.3  # Poisson's ratio

lmbd = E * nu / ((1 + nu) * (1 - 2 * nu))  # Lame's first parameter
mu = E / (2 * (1 + nu))  # Lame's second parameter

# Load
pstress = 1.0


sin = dde.backend.sin
cos = dde.backend.cos
stack = dde.backend.stack


# Load geometry mapping
nx=25
ny=75
dir_path = os.path.dirname(os.path.realpath(__file__))
Xp = np.loadtxt(os.path.join(dir_path, f"../geometry/deep_notched_{nx}x{ny}.txt"))


# Interpolate mapping
X_map_points = Xp[:, 0].reshape((ny, nx)).T
Y_map_points = Xp[:, 1].reshape((ny, nx)).T

def coordMap(x, padding=1e-6):
    x_pos = x[0]/x_max*(nx-1)*(1-2*padding) + padding
    y_pos = x[1]/y_max*(ny-1)*(1-2*padding) + padding

    x_mapped = jax.scipy.ndimage.map_coordinates(X_map_points, [x_pos, y_pos], order=1, mode='nearest')
    y_mapped = jax.scipy.ndimage.map_coordinates(Y_map_points, [x_pos, y_pos], order=1, mode='nearest')

    return jnp.stack((x_mapped, y_mapped), axis=0)

def tensMap(tens, x):
    J = jax.jacobian(coordMap)(x)
    return J @ tens

geom = dde.geometry.Rectangle([0, 0], [x_max, y_max])


def boundary_left(x, on_boundary):
    return on_boundary and dde.utils.isclose(x[0], 0.0)


def boundary_right(x, on_boundary):
    return on_boundary and dde.utils.isclose(x[0], y_max)


def boundary_bottom(x, on_boundary):
    return on_boundary and dde.utils.isclose(x[1], 0.0)

# Soft Boundary Conditions
ux_left_bc = dde.icbc.DirichletBC(geom, lambda x: 0, boundary_left, component=0)
uy_bottom_bc = dde.icbc.DirichletBC(geom, lambda x: 0, boundary_bottom, component=1)
sxx_right_bc = dde.icbc.DirichletBC(geom, lambda x: pstress, boundary_right, component=2)



def HardBC(x, f):
    if net_type == "spinn" and x.shape[0] != f.shape[0]:
        x_mesh = [x_.ravel() for x_ in jnp.meshgrid(x[:, 0], x[:, 1], indexing="ij")]
        x = stack(x_mesh, axis=-1)

    Ux = f[:, 0] * x[:, 1]*(y_max - x[:, 1])*u_0 
    Uy = f[:, 1] * x[:, 1]*u_0

    U_mapped = jax.vmap(tensMap)(stack((Ux, Uy), axis=1), x)


    Sxx = f[:, 2] * (x_max - x[:, 0])*x[:, 0]
    Syy = f[:, 3] #* (y_max - x[:, 1]) + pstress
    Sxy = f[:, 4] * x[:, 0]*(x_max - x[:, 0])

    S = jnp.stack((Sxx, Sxy, Sxy, Syy), axis=1).reshape(-1, 2, 2)
    S_mapped = jax.vmap(tensMap)(S, x)

    Syy_mapped = S_mapped[:,1,1] * (y_max - x[:, 1]) + pstress

    S_mapped = jnp.stack((S_mapped[:,0,0],Syy_mapped,(S_mapped[:,0,1]+S_mapped[:,1,0])/2), axis=1)
    return jnp.concatenate((U_mapped, S_mapped), axis=1)
    # return stack((Ux, Uy, Sxx, Syy, Sxy), axis=1)




def jacobian(f, x, i, j):
    if dde.backend.backend_name == "jax":
        return dde.grad.jacobian(f, x, i=i, j=j)[
            0
        ]  # second element is the function used by jax to compute the gradients
    else:
        return dde.grad.jacobian(f, x, i=i, j=j)


def pde(x, f):
    # x_mesh = jnp.meshgrid(x[:,0].ravel(), x[:,0].ravel(), indexing='ij')
    if net_type == "spinn":
        x_mesh = [x_.reshape(-1) for x_ in jnp.meshgrid(x[:, 0], x[:, 1], indexing="ij")]
        x = stack(x_mesh, axis=-1)
    x = jax.vmap(coordMap)(x)

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

# X_DIC = geom.uniform_points(1000, boundary=False)
# X_DIC_input = np.stack([np.linspace(0, x_max, n_DIC)] * 2, axis=1)
# X_DIC_mesh = [x_.ravel() for x_ in np.meshgrid(X_DIC_input[:,0],X_DIC_input[:,1],indexing="ij")]
# X_DIC_plot = np.stack(X_DIC_mesh, axis=1)
# if net_type != "spinn":
#     X_DIC_input = X_DIC_plot

# U_DIC = solution_fn(X_DIC_input)[:,:2]
# noise_floor = noise_ratio * np.std(U_DIC)
# U_DIC += np.random.normal(0, noise_floor, U_DIC.shape)

# measure_Ux = dde.PointSetBC(X_DIC_input, U_DIC[:, 0:1], component=0)
# measure_Uy = dde.PointSetBC(X_DIC_input, U_DIC[:, 1:2], component=1)

# Integral stress BC
n_integral = 10
x_integral = np.linspace(0, x_max, n_integral)
y_integral = np.concatenate((np.linspace(0, y_max*0.4, int(n_integral/2)), np.linspace(y_max*0.6, y_max, int(n_integral/2))))
integral_points = np.stack((x_integral, y_integral), axis=1)

def integral_stress(inputs, outputs, X):
    x_inputs = jnp.meshgrid(inputs[:, 0], inputs[:, 1], indexing="ij")[0]
    Syy = outputs[:, 3:4].reshape(x_inputs.shape)
    return jnp.trapezoid(Syy, x_inputs, axis=0)

Integral_BC = dde.PointSetOperatorBC(integral_points, pstress*x_max, integral_stress)

bcs = [Integral_BC]
num_boundary = 0

# if n_DIC:
#     bcs += [measure_Ux, measure_Uy]

if bc_type == "soft":
    bcs += [
        sxx_right_bc,
        uy_bottom_bc,
        ux_left_bc,
    ]
    num_boundary = 64 if net_type == "spinn" else 500


def get_num_params(net, input_shape=None):
    if dde.backend.backend_name == "pytorch":
        return sum(p.numel() for p in net.parameters())
    elif dde.backend.backend_name == "paddle":
        return sum(p.numpy().size for p in net.parameters())
    elif dde.backend.backend_name == "jax":
        if input_shape is None:
            raise ValueError("input_shape must be provided for jax backend")
        import jax
        import jax.numpy as jnp

        rng = jax.random.PRNGKey(0)
        return sum(
            p.size for p in jax.tree.leaves(net.init(rng, jnp.ones(input_shape)))
        )


activation = "tanh"
initializer = "Glorot uniform"
optimizer = "adam"
h_plot = 0.02
if net_type == "spinn":
    layers = [2, 32, 32, 32, 32, 5]
    net = dde.nn.SPINN(layers, activation, initializer, mlp)
    num_point = 100
    total_points = num_point**2 + num_boundary**2
    num_params = get_num_params(net, input_shape=layers[0])
    x_plot = np.linspace(0,x_max,100)
    y_plot = np.linspace(0,y_max,100)
    X_plot = np.stack((x_plot, y_plot), axis=1)

else:
    layers = [2, [40] * 5, [40] * 5, [40] * 5, [40] * 5, 5]
    net = dde.nn.PFNN(layers, activation, initializer)
    num_point = 500
    total_points = num_point + num_boundary
    num_params = get_num_params(net, input_shape=layers[0])
    X_mesh = np.meshgrid(
        np.linspace(0, x_max, int(x_max/h_plot)),
        np.linspace(0, y_max, int(y_max/h_plot)),
        indexing="ij",
    )
    X_plot = np.stack((X_mesh[0].ravel(), X_mesh[1].ravel()), axis=1)

num_test = 100

data = dde.data.PDE(
    geom,
    pde,
    bcs,
    num_domain=num_point,
    num_boundary=num_boundary,
    # solution=solution_fn,
    num_test=num_test,
    is_SPINN=net_type == "spinn",
)

if bc_type == "hard":
    net.apply_output_transform(HardBC)


results_path = [r"./forward",r"/mnt/d/phd/SPIQ/deep_notched/forward"][1]
folder_name = f"{net_type}_{available_time if available_time else n_iter}{'min' if available_time else 'iter'}"

# Check if any folders with the same name exist
existing_folders = [f for f in os.listdir(results_path) if f.startswith(folder_name)]

# If there are existing folders, find the highest number suffix
if existing_folders:
    suffixes = [int(f.split("-")[-1]) for f in existing_folders if f != folder_name]
    if suffixes:
        max_suffix = max(suffixes)
        folder_name = f"{folder_name}-{max_suffix + 1}"
    else:
        folder_name = f"{folder_name}-1"

# Create the new folder
new_folder_path = os.path.join(results_path, folder_name)
if not os.path.exists(new_folder_path):
    os.makedirs(new_folder_path)


callbacks = [dde.callbacks.Timer(available_time)] if available_time else []

for i, field in log_output_fields.items():
    callbacks.append(dde.callbacks.OperatorPredictor(X_plot, lambda x, output, i=i: output[0][:, i], period=log_every, filename=os.path.join(new_folder_path, f"{field}_history.dat")))

model = dde.Model(data, net)
model.compile(optimizer, lr=lr, loss_weights=loss_weights)

start_time = time.time()
losshistory, train_state = model.train(
    iterations=n_iter, callbacks=callbacks, display_every=log_every
)
elapsed = time.time() - start_time


def log_config(fname):
    import json
    import platform
    import psutil

    system_info = {
        "OS": platform.system(),
        "Release": platform.release(),
        "Version": platform.version(),
        "Machine": platform.machine(),
        "Processor": platform.processor(),
        "CPU count": psutil.cpu_count(),
        "RAM": psutil.virtual_memory().total / (1024**3),
    }


    gpu_info = {}

    execution_info = {
        "n_iter": train_state.epoch,
        "elapsed": elapsed,
        "iter_per_sec": train_state.epoch / elapsed,
        "backend": dde.backend.backend_name,
        "batch_size": total_points,
        "num_params": num_params,
        "activation": activation,
        "initializer": initializer,
        "optimizer": optimizer,
        "net_type": net_type,
        "mlp": mlp,
        "bc_type": bc_type,
        "logged_fields": log_output_fields,
        "n_DIC": n_DIC,
        "lr": lr,

    }

    info = {**system_info, **gpu_info, **execution_info}
    info_json = json.dumps(info, indent=4)

    with open(fname, "w") as f:
        f.write(info_json)


log_config(os.path.join(new_folder_path, "config.json"))
dde.utils.save_loss_history(
    losshistory, os.path.join(new_folder_path, "loss_history.dat")
)
