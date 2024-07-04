import matplotlib.pyplot as plt
import pyvista
import ufl
from ufl import SpatialCoordinate, TestFunction, TrialFunction, dot, dx, grad, div, FacetNormal,VectorElement, inner
import numpy as np
from dolfinx import default_scalar_type

from petsc4py import PETSc
from mpi4py import MPI

from dolfinx import fem, mesh, io, plot, cpp
from dolfinx.fem.petsc import assemble_vector, assemble_matrix, create_vector, apply_lifting, set_bc, create_matrix
from dolfinx.io import XDMFFile

########################################################################################################################
'''
LEVEL SET FUNCTION
'''
########################################################################################################################
# Define temporal parameters
t = 0  # Start time
T = 0.05  # Final time
alpha = 0.1

# Define mesh
N = 40
space_step = 1/N
dt = alpha * space_step**2 # time step size
num_steps = int(T/dt)
domain = mesh.create_rectangle(MPI.COMM_WORLD, [np.array([0, 0]), np.array([1, 1])],
                               [N, N], mesh.CellType.triangle)
W = fem.FunctionSpace(domain, ("Lagrange", 1))
n = FacetNormal(domain)

phi_n = fem.Function(W)  # Level set function

# Initial condition for phi s.t. it is negative in the anode and positive in the bath
# The unit normal in this case (n = Grad(phi)/||Grad(phi)||) will always point inside the bath (phi >0)

'''
Variational problem and solver for level set function
'''
class exact_solution():
    def __init__(self, t):
        self.t = t

    def __call__(self, x):
        # return x[1] - 0.5 - self.t*np.sin(2*np.pi*x[0])/8 # Rio Tinto Thesis Sonia PAIN
        return np.tanh(-60*((x[1] - 10*self.t - (10*self.t**3)/3 - 0.25)**2 - 0.01))

phi_ex = exact_solution(t)

phi_n.interpolate(phi_ex)

xdmf_levelset = io.XDMFFile(domain.comm, "level_set.xdmf", "w")
xdmf_levelset.write_mesh(domain)

# Define solution variable, and interpolate initial solution for visualization in Paraview
phi_h = fem.Function(W)
phi_h.name = "phi_h"
phi_h.interpolate(phi_ex)
xdmf_levelset.write_function(phi_h, t)

def boundary_D(x):
    return np.isclose(x[1], 0)
dofs_D = fem.locate_dofs_geometrical(W, boundary_D)

phi_D = fem.Function(W)
phi_D.interpolate(phi_ex)

BCs = fem.dirichletbc(phi_D, dofs_D) ########################### YOu have to update those every iteration
# Create boundary condition
'''fdim = domain.topology.dim - 1
boundary_facets = mesh.locate_entities_boundary(
    domain, fdim, lambda x: np.isclose(x[1], 1))
BCs = fem.dirichletbc(PETSc.ScalarType(0.5), fem.locate_dofs_topological(W, fdim, boundary_facets), W)'''

phi, v = ufl.TrialFunction(W), ufl.TestFunction(W)

'''def u_ex(x):
    values = np.zeros((2, x.shape[1]))
    values[0] = -2 * (np.sin(np.pi * x[0])**2) * np.sin(np.pi * x[1]) * np.cos(np.pi * x[1]) * np.cos((np.pi * t) / T)
    values[1] = 2*np.cos(np.pi*x[0])*np.sin(np.pi*x[0])*((np.sin(np.pi*x[1]))**2)* np.cos((np.pi*t)/T)
    values[1] = -0.01
    return values'''

class u_exact():
    def __init__(self, t):
        self.t = t

    def __call__(self, x):
        values = np.zeros((2, x.shape[1]))
        #values[0] = x[0]*(1-x[0])
        #values[1] = np.sin(2*x[0]*np.pi)/8 + x[0]*(1-x[0])*2*np.pi*self.t*np.cos(2*x[0]*np.pi)/8
        values[1] = 10 + 10*self.t**2
        return values

vec_fe = VectorElement("Lagrange", domain.ufl_cell(), 1)
W_vec = fem.FunctionSpace(domain, vec_fe)

u_ex = u_exact(t)
jh = fem.Function(W_vec)
jh.interpolate(u_ex)


# Retrieve the cells dimensions
tdim = domain.topology.dim
num_cells = domain.topology.index_map(tdim).size_local
h = cpp.mesh.h(domain._cpp_object, tdim, range(num_cells))
class delta_func():
    def __init__(self, t, jh, h):
        self.t = t
        self.jh = jh
        self.h = h

    def __call__(self):
        average_potGrad = fem.form(inner(self.jh, self.jh) * dx)
        average = fem.assemble_scalar(average_potGrad)
        L2_average = np.sqrt(domain.comm.allreduce(average, op=MPI.SUM))
        return self.h.max()/(2*L2_average)

delta = delta_func(t, jh, h)

w = v #+ delta() * dot(jh, grad(v))
theta = 0.5
a_levelSet = (phi * w * dx + (dt * theta) * dot(jh, grad(phi)) * w * dx)
L_levelSet = (phi_n * w * dx - dt * (1-theta) * dot(jh, grad(phi_n)) * w * dx)

#Preparing linear algebra structures for time dependent problems.
bilinear_form = fem.form(a_levelSet)
linear_form = fem.form(L_levelSet)

# Observe that the left hand side of the system does not change from one time step to another, thus we
# only need to assemble it once. The right hand side, which is dependent on the previous time step u_n, has
# to be assembled every time step.
A = create_matrix(bilinear_form)
# A = assemble_matrix(bilinear_form, bcs=[BCs])
# A.assemble()
b = create_vector(linear_form)

# We create a linear algebra solver using PETSc, and assign the matrix A to the solver
solver = PETSc.KSP().create(domain.comm)
solver.setOperators(A)
solver.setType(PETSc.KSP.Type.PREONLY)
solver.getPC().setType(PETSc.PC.Type.LU)

for i in range(num_steps):
    print(i)
    t += dt

    phi_ex.t = t
    phi_D.interpolate(phi_ex)

    u_ex.t = (t + (t-dt))/2
    jh.interpolate(u_ex)

    delta.jh = jh

    distance = fem.form(inner(grad(phi_n), grad(phi_n)) * dx)
    average_dist = fem.assemble_scalar(distance)
    L2_average_dist = np.sqrt(domain.comm.allreduce(average_dist, op=MPI.SUM))
    #if domain.comm.rank == 0:
        #print(f"Gradient distance : {L2_average_dist:.2e}")

    A.zeroEntries()
    assemble_matrix(A, bilinear_form, bcs=[BCs])
    A.assemble()
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

    '''if i == 1000:
        error_L2 = np.sqrt(
            domain.comm.allreduce(fem.assemble_scalar(fem.form((phi_n - phi_D) ** 2 * ufl.dx)), op=MPI.SUM))
        eh = phi_n - phi_D
        error_H10 = fem.form(inner(grad(eh), grad(eh)) * dx)
        E_H10 = np.sqrt(domain.comm.allreduce(fem.assemble_scalar(error_H10), op=MPI.SUM))
        if domain.comm.rank == 0:
            print(f"L2-error: {error_L2:.2e}")
            print(f"H01-error: {E_H10:.2e}")
        break'''

error_L2 = np.sqrt(
            domain.comm.allreduce(fem.assemble_scalar(fem.form((phi_n - phi_D) ** 2 * ufl.dx)), op=MPI.SUM))
eh = phi_n - phi_D
error_H10 = fem.form(inner(grad(eh), grad(eh)) * dx)
E_H10 = np.sqrt(domain.comm.allreduce(fem.assemble_scalar(error_H10), op=MPI.SUM))
if domain.comm.rank == 0:
    print(f"L2-error: {error_L2:.2e}")
    print(f"H01-error: {E_H10:.2e}")

xdmf_levelset.close()