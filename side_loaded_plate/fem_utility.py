# Define boundary condition

from fenics import *
from time import perf_counter


def fem_side_loaded_plate(E=210e3, nu=0.3, Lmax=3, m=10, b=50, mesh_size=100):
    print("Computing FEM solution for side loaded plate...")
    t_start = perf_counter()
    lambda_ = E*nu/((1+nu)*(1-2*nu))
    mu = E/(2*(1+nu))

    mesh = RectangleMesh(Point(0, 0), Point(Lmax, Lmax), mesh_size, mesh_size)
    V = VectorFunctionSpace(mesh, 'P', 1)

    tol = 1E-14

    def left_boundary(x, on_boundary):
        return on_boundary and near(x[0], 0, tol)

    def bottom_boundary(x, on_boundary):
        return on_boundary and near(x[1], 0, tol)

    def right_boundary(x, on_boundary):
        return on_boundary and near(x[0], Lmax, tol)

    bc1 = DirichletBC(V.sub(0), Constant(0), left_boundary)
    bc2 = DirichletBC(V.sub(1), Constant(0), bottom_boundary)
    bc = [bc1, bc2]

    #mark right boundary
    right = AutoSubDomain(right_boundary)
    boundaries = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
    boundaries.set_all(0)
    right.mark(boundaries, 1)
    ds = Measure('ds', domain=mesh, subdomain_data=boundaries)


    # Define strain and stress

    def epsilon(u):
        # return 0.5*(nabla_grad(u) + nabla_grad(u).T)
        return sym(nabla_grad(u))

    def sigma(u):
        return lambda_*div(u)*Identity(d) + 2*mu*epsilon(u)

    # Define variational problem
    x = SpatialCoordinate(mesh)
    u = TrialFunction(V)
    d = u.geometric_dimension()  # space dimension
    v = TestFunction(V)
    f = Constant((0, 0))

    T = Expression(('m*x[1]+b','0'), degree=1, m=m, b=b)
    a = inner(sigma(u), epsilon(v))*dx
    L = dot(f, v)*dx + dot(T, v)*ds(1)

    # Compute solution
    u = Function(V)
    solve(a == L, u, bc)

    tensor_space = TensorFunctionSpace(mesh, 'P', 1)
    strain = project(epsilon(u), tensor_space)
    stress = project(sigma(u), tensor_space)

    print(f"FEM solution computed in {perf_counter()-t_start:.2f} seconds.")

    return u, strain, stress



