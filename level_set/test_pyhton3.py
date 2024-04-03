import matplotlib.pyplot as plt
import pyvista
import ufl
from ufl import SpatialCoordinate, TestFunction, TrialFunction, dot, dx, grad, div, FacetNormal,VectorElement, inner
import numpy as np
from dolfinx import default_scalar_type

from petsc4py import PETSc
from mpi4py import MPI

from dolfinx import fem, mesh, io, plot, cpp
from dolfinx.fem.petsc import assemble_vector, assemble_matrix, create_vector, apply_lifting, set_bc
from dolfinx.io import XDMFFile

def project(e, target_func, bcs=[]):
    """Project UFL expression.
    Note
    ----
    This method solves a linear system (using KSP defaults).
    """

    # Ensure we have a mesh and attach to measure
    V = target_func.function_space
    dx = ufl.dx(V.mesh)

    # Define variational problem for projection
    w = ufl.TestFunction(V)
    v = ufl.TrialFunction(V)
    a = fem.form(ufl.inner(v, w) * dx)
    L = fem.form(ufl.inner(e, w) * dx)

    # Assemble linear system
    A = assemble_matrix(a, bcs)
    A.assemble()
    b = assemble_vector(L)
    apply_lifting(b, [a], [bcs])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    set_bc(b, bcs)

    # Solve linear system
    solver = PETSc.KSP().create(A.getComm())
    solver.setType("bcgs")
    solver.getPC().setType("bjacobi")
    solver.rtol = 1.0e-05
    solver.setOperators(A)
    solver.solve(b, target_func.vector)
    assert solver.reason > 0
    target_func.x.scatter_forward()

    # Destroy PETSc linear algebra objects and solver
    solver.destroy()
    A.destroy()
    b.destroy()

########################################################################################################################
'''
LEVEL SET FUNCTION
'''
########################################################################################################################
# x = ufl.SpatialCoordinate(domain)
# Define temporal parameters
t = 0  # Start time
T = 1.0  # Final time
num_steps = 50
dt = T / num_steps  # time step size

# Define mesh
nx, ny = 50, 50
domain = mesh.create_rectangle(MPI.COMM_WORLD, [np.array([0, 0]), np.array([1, 1])],
                               [nx, ny], mesh.CellType.triangle)
W = fem.FunctionSpace(domain, ("Lagrange", 1))
n = FacetNormal(domain)

phi_n = fem.Function(W)  # Level set function
# Initial condition for phi
def phi_zero(x):
    y = np.ones(len(x[1]))
    for index, x_coord in enumerate(x[0]):
        if x_coord <= 0.34 and x[1][index] < 0.5:
            y[index] = -1
        if x_coord <= 0.34 and x[1][index] == 0.5:
            y[index] = 0
        if 0.34 <= x_coord <= 0.64 and x[1][index] < 0.74:
            y[index] = -1
        if 0.34 <= x_coord <= 0.64 and x[1][index] == 0.74:
            y[index] = 0
        if x_coord >= 0.64 and x[1][index] < 0.5:
            y[index] = -1
        if x_coord >= 0.64 and x[1][index] == 0.5:
            y[index] = 0
    return y

phi_n.interpolate(phi_zero)

# Boundary condition for phi, inflow at the top of the domain = 0, so nothing as to be specified

# Declare variables for weak formulations
# Level set
xdmf_levelset = io.XDMFFile(domain.comm, "level_set.xdmf", "w")
xdmf_levelset.write_mesh(domain)

# Define solution variable, and interpolate initial solution for visualization in Paraview
phi_h = fem.Function(W)
phi_h.name = "uh"
phi_h.interpolate(phi_zero)
xdmf_levelset.write_function(phi_h, t)

phi, w = ufl.TrialFunction(W), ufl.TestFunction(W)

# Potential
D = fem.FunctionSpace(domain, ("DG", 0))
sigma = fem.Function(D)
def anode_conductivity(T):
    return 1. / (5.929e-5 - T * 1.235e-8)
conductivity_anode = anode_conductivity(800)
conductivity_bath = 210
def smoothed_heaviside(var, epsilon):
    # Since phi_n is a 'petsc4py.PETSc.Vec'
    y = np.ones(var.vector.getSize())
    # The method .array is to transform the PETSC vector into a numpy vector
    for index, x in enumerate(var.vector.array):
        if x < -epsilon:
            y[index] = 0
        if -epsilon <= x <= epsilon:
            y[index] = 0.5 + (45*(x/epsilon) - 50*(x/epsilon)**3 + 21*(x/epsilon)**5)/32
        if x > epsilon:
            y[index] = 1
    return y
# Retrieve the cells dimensions
tdim = domain.topology.dim
num_cells = domain.topology.index_map(tdim).size_local
h = cpp.mesh.h(domain._cpp_object, tdim, range(num_cells))
epsilon = h.max()

'''
You cannot. A quadrature function space are point evaluations. They do not have an underlying polynomial 
that can be used as the basis on a single cell.
You can project a quadrature function into any space though.
'''
phi_n_project = fem.Function(D)
project(phi_n, phi_n_project)
sigma.x.array[:] = conductivity_anode + (conductivity_bath - conductivity_anode)*smoothed_heaviside(phi_n_project, epsilon)
sigma.x.scatter_forward()
V = TrialFunction(W)
csi = TestFunction(W)


