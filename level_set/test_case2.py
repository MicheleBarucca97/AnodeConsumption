import matplotlib.pyplot as plt
import pyvista
import ufl
from ufl import SpatialCoordinate, TestFunction, TrialFunction, dot, dx, grad, div, FacetNormal,VectorElement, inner
import numpy as np
from dolfinx import default_scalar_type

from petsc4py import PETSc
from mpi4py import MPI

from dolfinx import fem, mesh, io, plot
from dolfinx.fem.petsc import assemble_vector, assemble_matrix, create_vector, apply_lifting, set_bc
from dolfinx.io import XDMFFile

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

# First define two domains
Q = fem.FunctionSpace(domain, ("DG", 0))

def Omega_0(x):
    y = np.zeros(len(x[1]), dtype=bool)
    for index, x_coord in enumerate(x[0]):
        if x_coord <= 0.34 and x[1][index] <= 0.5:
            y[index] = True
        if 0.34 <= x_coord <= 0.64 and x[1][index] <= 0.74:
            y[index] = True
        if x_coord >= 0.64 and x[1][index] <= 0.5:
            y[index] = True
    return y
def Omega_1(x):
    y = np.zeros(len(x[1]), dtype=bool)

    for index, x_coord in enumerate(x[0]):
        if x_coord <= 0.34 and x[1][index] >= 0.5:
            y[index] = True
        if 0.34 <= x_coord <= 0.64 and x[1][index] >= 0.74:
            y[index] = True
        if x_coord >= 0.64 and x[1][index] >= 0.5:
            y[index] = True
    return y

sigma = fem.Function(Q)
cells_0 = mesh.locate_entities(domain, domain.topology.dim, Omega_0)
cells_1 = mesh.locate_entities(domain, domain.topology.dim, Omega_1)


def anode_conductivity(T):
    return 1. / (5.929e-5 - T * 1.235e-8)
conductivity_anode = anode_conductivity(800)
sigma.x.array[cells_0] = np.full_like(cells_0, 210, dtype=default_scalar_type)
sigma.x.array[cells_1] = np.full_like(cells_1, conductivity_anode, dtype=default_scalar_type)

with XDMFFile(MPI.COMM_WORLD, "marker_sigma.xdmf", "w") as xdmf:
    xdmf.write_mesh(domain)
    # sigma.name = 'markers [-]'
    xdmf.write_function(sigma)

# Write the weak formulation
V = TrialFunction(W)
csi = TestFunction(W)
a = inner(sigma * grad(V), grad(csi)) * dx

# Force term in case V(x,y) = arctan(pi*y)
def V_exact_ufl(mode):
    #return lambda x: mode.atan(mode.pi * x[1])
    return lambda x: (10000 -8000 * x[1]) * x[1]

def V_exact_numpy(mode):
    #return lambda x: mode.arctan(mode.pi * x[1])
    return lambda x: (10000 -8000 * x[1]) * x[1]

V_numpy = V_exact_numpy(np)  # which will be used for interpolation
V_ufl = V_exact_ufl(ufl)  # which will be used for defining the source term

V_ex = fem.Function(W)
V_ex.interpolate(V_numpy)

x = SpatialCoordinate(domain)
f = div(-sigma * grad(V_ufl(x)))
g = sigma * dot(grad(V_ufl(x)), n)
ds = ufl.Measure("ds", domain=domain) # This command or you can just import it from ufl
L = f * csi * dx + g * csi * ds

def boundary_D(x):
    return np.logical_or(np.isclose(x[0], 0), np.isclose(x[0], 1))
dofs_D = fem.locate_dofs_geometrical(W, boundary_D)

BCs = [fem.dirichletbc(V_ex, dofs_D)]

########################################################################################################################
'''
Remember that if you have non-homogeneous b.c. those have to be enforced on the boundary, LOOK HOW TO DO THAT ON PETSC
'''
########################################################################################################################
default_problem = fem.petsc.LinearProblem(a, L, bcs=BCs,
                                petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
Vh = default_problem.solve()

L2_error = fem.form(inner(Vh - V_ex, Vh - V_ex) * dx)
error_local = fem.assemble_scalar(L2_error)
error_L2 = np.sqrt(domain.comm.allreduce(error_local, op=MPI.SUM))

# Only print the error on one process
if domain.comm.rank == 0:
    print(f"Error_L2 : {error_L2:.2e}")

# Plotting the solution
tdim = domain.topology.dim
topology, cell_types, geometry = plot.vtk_mesh(domain, tdim)
grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)

V_topology, V_cell_types, V_geometry = plot.vtk_mesh(W)

V_grid = pyvista.UnstructuredGrid(V_topology, V_cell_types, V_geometry)
V_grid.point_data["V"] = np.abs(Vh.x.array.real - V_ex.x.array.real)
V_grid.set_active_scalars("V")
V_plotter = pyvista.Plotter()
V_plotter.add_mesh(V_grid, show_edges=True)
_ = V_plotter.add_axes(
    line_width=5,
    cone_radius=0.6,
    shaft_length=0.7,
    tip_length=0.3,
    ambient=0.5,
    label_size=(0.4, 0.16),
)
V_plotter.view_xy()
V_plotter.show()

flux_V = grad(Vh)
W_flux_V = fem.FunctionSpace(domain, VectorElement("DG", domain.ufl_cell(), 0)) # You should use a TensorFunctionSpace
flux_V_expr = fem.Expression(flux_V, W_flux_V.element.interpolation_points())
flux = fem.Function(W_flux_V)
flux.interpolate(flux_V_expr)

xdmf1 = XDMFFile(domain.comm, "flux_potential.xdmf", "w")
xdmf1.write_mesh(domain)
xdmf1.write_function(flux)
xdmf1.close()

##########################################################################################
# LEVEL SET PROBLEM
##########################################################################################
