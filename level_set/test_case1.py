import matplotlib.pyplot as plt
import pyvista
import ufl
from ufl import SpatialCoordinate, TestFunction, TrialFunction, dot, ds, dx, grad, div, FacetNormal,VectorElement, inner
import numpy as np

from petsc4py import PETSc
from mpi4py import MPI

from dolfinx import fem, mesh, io, plot
from dolfinx.fem.petsc import assemble_vector, assemble_matrix, create_vector, apply_lifting, set_bc

# Define temporal parameters
t = 0  # Start time
T = 1.0  # Final time
num_steps = 50
dt = T / num_steps  # time step size

# Define mesh
nx, ny = 50, 50
# For arctan manufactured solution
#domain = mesh.create_rectangle(MPI.COMM_WORLD, [np.array([-2, -2]), np.array([2, 2])],
#                               [nx, ny], mesh.CellType.triangle)
domain = mesh.create_rectangle(MPI.COMM_WORLD, [np.array([0, 0]), np.array([1, 1])],
                               [nx, ny], mesh.CellType.triangle)
W = fem.FunctionSpace(domain, ("Lagrange", 1))
n = FacetNormal(domain)

# Define Neumann boundary condition for Potential problem
'''
def g(x):
    return np.vstack((np.zeros_like(x[0]), 1/(1+(np.pi*x[1])**2)))
vec_fe = VectorElement("Lagrange", domain.ufl_cell(), 1)
W_vec = fem.FunctionSpace(domain, vec_fe)
g_fe = fem.Function(W_vec)
g_fe.interpolate(g)
'''

# Write the weak formulation
V = TrialFunction(W)
csi = TestFunction(W)
sigma = fem.Constant(domain, PETSc.ScalarType(500))
a = dot(sigma * grad(V), grad(csi)) * dx

# Force term in case V(x,y) = arctan(pi*y)
def V_exact_ufl(mode):
    #return lambda x: mode.atan(mode.pi * x[1])
    return lambda x: (10000 -8000 * x[1]) * x[1]

def V_exact_numpy(mode):
    #return lambda x: mode.arctan(mode.pi * x[1])
    return lambda x: (10000 -8000 * x[1]) * x[1]

V_numpy = V_exact_numpy(np) # which will be used for interpolation
V_ufl = V_exact_ufl(ufl) # which will be used for defining the source term

V_ex = fem.Function(W)
V_ex.interpolate(V_numpy)

x = SpatialCoordinate(domain)
f = div(-sigma * grad(V_ufl(x)))
g = sigma * dot(grad(V_ufl(x)), n)
ds = ufl.Measure("ds", domain=domain) # This command or you can just import it from ufl
L = f * csi * dx + g * csi * ds

# Since for this problem the Potential is only determined up to a
# constant, we pin the Potential at the point (1, 0)
'''Gamma = mesh.locate_entities_boundary(domain, dim=1, marker=lambda x: np.logical_and.reduce((
                                                                np.isclose(x[1], 1),  # y=0
                                                                np.isclose(x[0], 0))))# x=0
Gamma2 = mesh.locate_entities_boundary(domain, dim=1, marker=lambda x: np.logical_and.reduce((
                                                                np.isclose(x[1], 0),  # y=0
                                                                np.isclose(x[0], 1))))# x=0
# The entity dimension in this case has to be set to zero, since you want to apply this condition only in one node.
dofs_p = fem.locate_dofs_topological(W, 0, Gamma)
dofs_p2 = fem.locate_dofs_topological(W, 0, Gamma2)
BCs = [fem.dirichletbc(PETSc.ScalarType(2000), dofs_p, W),fem.dirichletbc(PETSc.ScalarType(0), dofs_p2, W)]'''

#################################
'''
Without Dirichlet boundary condition the simulation will not converge because the COMPATIBILITY CONDITION IS NOT 
SATISFIED with this manifactured solution
'''
#################################
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

'''#Linear algebra structures
bilinear_form = fem.form(a)
linear_form = fem.form(L)

A = assemble_matrix(bilinear_form, bcs=BCs)
A.assemble()
b = create_vector(linear_form)
# Set Dirichlet boundary condition values in the RHS
set_bc(b, BCs)

#linear algebra solver using PETSc, and assign the matrix A to the solver, and choose the solution strategy.
solver = PETSc.KSP().create(domain.comm)
solver.setOperators(A)
solver.setType(PETSc.KSP.Type.PREONLY)
solver.getPC().setType(PETSc.PC.Type.LU)

# Compute the solution
Vh = fem.Function(W)
solver.solve(b, Vh.vector)
Vh.x.scatter_forward()'''

L2_error = fem.form(inner(Vh - V_ex, Vh - V_ex) * dx)
error_local = fem.assemble_scalar(L2_error)
error_L2 = np.sqrt(domain.comm.allreduce(error_local, op=MPI.SUM))

# Only print the error on one process
if domain.comm.rank == 0:
    print(f"Error_L2 : {error_L2:.2e}")

# Plotting the solution
#pyvista.set_jupyter_backend('client')
tdim = domain.topology.dim
topology, cell_types, geometry = plot.vtk_mesh(domain, tdim)
grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)

V_topology, V_cell_types, V_geometry = plot.vtk_mesh(W)

V_grid = pyvista.UnstructuredGrid(V_topology, V_cell_types, V_geometry)
V_grid.point_data["V"] = Vh.x.array.real
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

from dolfinx.io import XDMFFile
xdmf1 = XDMFFile(domain.comm, "flux_potential.xdmf", "w")
xdmf1.write_mesh(domain)
xdmf1.write_function(flux)
xdmf1.close()
########################################################################################################################
'''
In Alucell for the problem of the potential they also fix the value of it in one point in the boundary so that they have
a well posed problem (ref 2007_alucell_user_guide.pdf). 
Calculate j in a weak sense int_(j*v) = int_(sigma*grad(V)*v) and move to the Level set problem.
'''
########################################################################################################################
# Obtain the current density j in a weak sense (no need of b.c.)
vec_fe = VectorElement("Lagrange", domain.ufl_cell(), 1)
W_vec = fem.FunctionSpace(domain, vec_fe)

j = TrialFunction(W_vec)
tetha = TestFunction(W_vec)

a2 = dot(j, tetha) * dx
L2 = -sigma*dot(grad(Vh), tetha) *dx - sigma * dot(grad(V_ufl(x)), tetha) *dx

default_problem = fem.petsc.LinearProblem(a2, L2, bcs=[],
                                petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
jh = default_problem.solve()

xdmf1 = XDMFFile(domain.comm, "current.xdmf", "w")
xdmf1.write_mesh(domain)
xdmf1.write_function(jh)
xdmf1.close()

