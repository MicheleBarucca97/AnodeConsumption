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

########################################################################################################################
'''
LEVEL SET FUNCTION
'''
########################################################################################################################
# x = ufl.SpatialCoordinate(domain)
# Define temporal parameters
t = 0  # Start time
T = 1.0  # Final time
num_steps = 250
dt = T / num_steps  # time step size

# Define mesh
nx, ny = 35, 35
domain = mesh.create_rectangle(MPI.COMM_WORLD, [np.array([0, 0]), np.array([1, 1])],
                               [nx, ny], mesh.CellType.triangle)
W = fem.FunctionSpace(domain, ("Lagrange", 1))
n = FacetNormal(domain)

phi_n = fem.Function(W)  # Level set function

# Initial condition for phi s.t. it is negative in the anode and positive in the bath
# The unit normal in this case (n = Grad(phi)/||Grad(phi)||) will always point inside the bath (phi >0)
def phi_zero():
    return lambda x: x[1] - 0.5

phi_n.interpolate(phi_zero())

xdmf_levelset = io.XDMFFile(domain.comm, "level_set.xdmf", "w")
xdmf_levelset.write_mesh(domain)

# Define solution variable, and interpolate initial solution for visualization in Paraview
phi_h = fem.Function(W)
phi_h.name = "phi_h"
phi_h.interpolate(phi_zero())
xdmf_levelset.write_function(phi_h, t)

'''
Variational problem and solver for level set function
'''
# Create boundary condition
'''fdim = domain.topology.dim - 1
boundary_facets = mesh.locate_entities_boundary(
    domain, fdim, lambda x: np.isclose(x[1], 1))
BCs = fem.dirichletbc(PETSc.ScalarType(0), fem.locate_dofs_topological(W, fdim, boundary_facets), W)'''
def boundary_D(x):
    return np.isclose(x[1], 1)
dofs_D = fem.locate_dofs_geometrical(W, boundary_D)
phi_ex = phi_n
BCs = fem.dirichletbc(phi_ex, dofs_D)

phi, v = ufl.TrialFunction(W), ufl.TestFunction(W)

def u_ex(x):
    values = np.zeros((2, x.shape[1]))
    '''values[0] = -2 * (np.sin(np.pi * x[0])**2) * np.sin(np.pi * x[1]) * np.cos(np.pi * x[1]) * np.cos((np.pi * t) / T)
    values[1] = 2*np.cos(np.pi*x[0])*np.sin(np.pi*x[0])*((np.sin(np.pi*x[1]))**2)* np.cos((np.pi*t)/T)'''
    values[1] = -0.01
    return values

vec_fe = VectorElement("Lagrange", domain.ufl_cell(), 1)
W_vec = fem.FunctionSpace(domain, vec_fe)

jh = fem.Function(W_vec)
jh.interpolate(u_ex)

average_potGrad = fem.form(inner(jh, jh) * dx)
average = fem.assemble_scalar(average_potGrad)
L2_average = np.sqrt(domain.comm.allreduce(average, op=MPI.SUM))
# Retrieve the cells dimensions
tdim = domain.topology.dim
num_cells = domain.topology.index_map(tdim).size_local
h = cpp.mesh.h(domain._cpp_object, tdim, range(num_cells))
delta = h.max()/(2*L2_average)


w = v + delta * dot(jh, grad(v))
theta = 0.5
a_levelSet = (phi * w * dx - (dt * theta) * dot(jh, grad(phi)) * w * dx)
L_levelSet = (phi_n * w * dx + dt * (1-theta) * dot(jh, grad(phi_n)) * w * dx)

#Preparing linear algebra structures for time dependent problems.
bilinear_form = fem.form(a_levelSet)
linear_form = fem.form(L_levelSet)

# Observe that the left hand side of the system does not change from one time step to another, thus we
# only need to assemble it once. The right hand side, which is dependent on the previous time step u_n, has
# to be assembled every time step.
A = assemble_matrix(bilinear_form, bcs=[BCs])
A.assemble()
b = create_vector(linear_form)

# We create a linear algebra solver using PETSc, and assign the matrix A to the solver
solver = PETSc.KSP().create(domain.comm)
solver.setOperators(A)
solver.setType(PETSc.KSP.Type.PREONLY)
solver.getPC().setType(PETSc.PC.Type.LU)

for i in range(num_steps):
    t += dt

    distance = fem.form(inner(grad(phi_n), grad(phi_n)) * dx)
    average_dist = fem.assemble_scalar(distance)
    L2_average_dist = np.sqrt(domain.comm.allreduce(average_dist, op=MPI.SUM))
    if domain.comm.rank == 0:
        print(f"Gradient distance : {L2_average_dist:.2e}")

    # Update the right hand side reusing the initial vector
    with b.localForm() as loc_b:
        loc_b.set(0)
    assemble_vector(b, linear_form)

    # Apply Dirichlet boundary condition to the vector
    apply_lifting(b, [bilinear_form], [[BCs]])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    set_bc(b, [BCs])

    # Solve linear problem
    solver.solve(b, phi_h.vector)
    phi_h.x.scatter_forward()

    # Update solution at previous time step (u_n)
    phi_n.x.array[:] = phi_h.x.array
    phi_n.x.scatter_forward()

    # Write solution to file
    xdmf_levelset.write_function(phi_h, t)

xdmf_levelset.close()