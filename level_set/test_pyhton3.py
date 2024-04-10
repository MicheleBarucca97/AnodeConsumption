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

#def reinitialization():

# x = ufl.SpatialCoordinate(domain)
# Define temporal parameters
t = 0  # Start time
T = 1.0  # Final time
num_steps = 1000
dt = T / num_steps  # time step size

# Define mesh
nx, ny = 300, 300
domain = mesh.create_rectangle(MPI.COMM_WORLD, [np.array([0, 0]), np.array([1, 1])],
                               [nx, ny], mesh.CellType.triangle)
W = fem.FunctionSpace(domain, ("Lagrange", 1))
n = FacetNormal(domain)

phi_n = fem.Function(W)  # Level set function

# Initial condition for phi s.t. it is negative in the anode and positive in the bath
# The unit normal in this case (n = Grad(phi)/||Grad(phi)||) will always point inside the bath (phi >0)
tol = 10e-3
def phi_zero(x):
    return 0.5 - x[1]
    '''y = np.ones(len(x[1]))
    for index, x_coord in enumerate(x[0]):
        if x_coord < 0.35 or x_coord > 0.65:
            if x[1][index] < 0.5 - tol:
                y[index] = 0.5 - x[1][index]
            if 0.5 - tol <= x[1][index] <= 0.5 + tol:
                y[index] = 0
            if x[1][index] > 0.5 + tol:
                y[index] = 0.5 - x[1][index]
        if 0.35 <= x_coord <= 0.65:
            if x[1][index] < 0.7 - tol:
                y[index] = 0.7 - x[1][index]
            if 0.7 - tol <= x[1][index] <= 0.7 + tol:
                y[index] = 0
            if x[1][index] > 0.7 + tol:
                y[index] = 0.7 - x[1][index]
    return y'''

phi_n.interpolate(phi_zero)

# Boundary condition for phi, inflow at the top of the domain = 0, so nothing as to be specified

# Declare variables for weak formulations
# Level set
xdmf_levelset = io.XDMFFile(domain.comm, "level_set.xdmf", "w")
xdmf_levelset.write_mesh(domain)

# Define solution variable, and interpolate initial solution for visualization in Paraview
phi_h = fem.Function(W)
phi_h.name = "phi_h"
phi_h.interpolate(phi_zero)
xdmf_levelset.write_function(phi_h, t)

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
    count1 = 0
    count2 = 0
    count3 = 0
    # The method .array is to transform the PETSC vector into a numpy vector
    for index, x in enumerate(var.vector.array):

        if x < -epsilon:
            y[index] = 0
            count1 += 1
        if -epsilon <= x <= epsilon:
            y[index] = 0.5*(1 + (x/epsilon) + ufl.sin((ufl.pi*x)/epsilon)/ufl.pi)
            count2 += 1
        if x > epsilon:
            y[index] = 1
            count3 += 1
    print(count1)
    print(count2)
    print(count3)
    return y
# Retrieve the cells dimensions
tdim = domain.topology.dim
num_cells = domain.topology.index_map(tdim).size_local
h = cpp.mesh.h(domain._cpp_object, tdim, range(num_cells))
epsilon = h.max()

'''
You cannot interpolate. A quadrature function space are point evaluations. They do not have an underlying polynomial 
that can be used as the basis on a single cell.
You can project a quadrature function into any space though.
'''
phi_n_project = fem.Function(D)
project(phi_n, phi_n_project)

sigma.x.array[:] = conductivity_anode + (conductivity_bath - conductivity_anode)*smoothed_heaviside(phi_n_project, epsilon)
sigma.x.scatter_forward()
xdmf_sigma = io.XDMFFile(domain.comm, "sigma.xdmf", "w")
xdmf_sigma.write_mesh(domain)
xdmf_sigma.write_function(sigma, t)

'''
Variational problem and solver for the potential
'''
V = TrialFunction(W)
csi = TestFunction(W)

a = dot(sigma * grad(V), grad(csi)) * dx

def V_exact_ufl(mode):
    #return lambda x: mode.atan(mode.pi * x[1])
    return lambda x: (0.26 + 1.64 * x[1]) * x[1]

def V_exact_numpy(mode):
    #return lambda x: mode.arctan(mode.pi * x[1])
    return lambda x: (0.26 + 1.64 * x[1]) * x[1]

V_numpy = V_exact_numpy(np)  # which will be used for interpolation
V_ufl = V_exact_ufl(ufl)  # which will be used for defining the source term

V_ex = fem.Function(W)
V_ex.interpolate(V_numpy)

x = SpatialCoordinate(domain)
f = div(-sigma * grad(V_ufl(x)))
g = dot(sigma * grad(V_ufl(x)), n)
ds = ufl.Measure("ds", domain=domain) # This command or you can just import it from ufl
L = f * csi * dx + g * csi * ds

def boundary_D(x):
    return np.logical_or(np.isclose(x[0], 0), np.isclose(x[0], 1))
dofs_D = fem.locate_dofs_geometrical(W, boundary_D)

BCs = [fem.dirichletbc(V_ex, dofs_D)]

left_form = fem.form(a)
right_form = fem.form(L)

A_potential = assemble_matrix(left_form, bcs=BCs)
A_potential.assemble()
b_potential = create_vector(right_form)
assemble_vector(b_potential, right_form)
# Apply Dirichlet boundary condition to the vector
apply_lifting(b_potential, [left_form], [BCs])
b_potential.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
set_bc(b_potential, BCs)

# We create a linear algebra solver using PETSc, and assign the matrix A to the solver
solver_pot = PETSc.KSP().create(domain.comm)
solver_pot.setOperators(A_potential)
solver_pot.setType(PETSc.KSP.Type.PREONLY)
solver_pot.getPC().setType(PETSc.PC.Type.LU)

Vh = fem.Function(W)
solver_pot.solve(b_potential, Vh.vector)
Vh.x.scatter_forward()

'''
Variational problem and solver for current (is a vector)
'''
vec_fe = VectorElement("Lagrange", domain.ufl_cell(), 1)
W_vec = fem.FunctionSpace(domain, vec_fe)

j = TrialFunction(W_vec)
tetha = TestFunction(W_vec)

a2 = dot(j, tetha) * dx
L2 = -dot(sigma*grad(Vh), tetha) * dx - dot(sigma * grad(V_ufl(x)), tetha) *dx

left_form2 = fem.form(a2)
right_form2 = fem.form(L2)

A_curr = assemble_matrix(left_form2)
A_curr.assemble()
b_curr = create_vector(right_form2)
assemble_vector(b_curr, right_form2)

# We create a linear algebra solver using PETSc, and assign the matrix A to the solver
solver_cur = PETSc.KSP().create(domain.comm)
solver_cur.setOperators(A_curr)
solver_cur.setType(PETSc.KSP.Type.PREONLY)
solver_cur.getPC().setType(PETSc.PC.Type.LU)

jh = fem.Function(W_vec)
solver_cur.solve(b_curr, jh.vector)
jh.x.scatter_forward()

'''
Variational problem and solver for level set function
'''
phi, v = ufl.TrialFunction(W), ufl.TestFunction(W)

average_potGrad = fem.form(inner(jh, jh) * dx)
average = fem.assemble_scalar(average_potGrad)
L2_average = np.sqrt(domain.comm.allreduce(average, op=MPI.SUM))
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
A = assemble_matrix(bilinear_form)
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

    A.zeroEntries()
    fem.petsc.assemble_matrix(A, bilinear_form)  # type: ignore
    A.assemble()

    # Update the right hand side reusing the initial vector
    with b.localForm() as loc_b:
        loc_b.set(0)
    assemble_vector(b, linear_form)

    # Solve linear problem
    solver.solve(b, phi_h.vector)
    phi_h.x.scatter_forward()

    # Update solution at previous time step (u_n)
    phi_n.x.array[:] = phi_h.x.array

    # Update sigma
    with open("phi_n.txt", "w") as file:
        # Write each element of the vector to the file
        for index, x in enumerate(sigma.vector.array):
            file.write(str(x) + "\n")  # Add a newline after each element

    project(phi_n, phi_n_project)
    sigma.x.array[:] = conductivity_anode + (conductivity_bath - conductivity_anode) * smoothed_heaviside(phi_n_project,
                                                                                                          epsilon)
    sigma.x.scatter_forward()

    # Calculate the new potential
    A_potential.zeroEntries()
    fem.petsc.assemble_matrix(A_potential, left_form, bcs=BCs)  # type: ignore
    A_potential.assemble()
    with b_potential.localForm() as loc_b:
        loc_b.set(0)
    assemble_vector(b_potential, right_form)
    # Apply Dirichlet boundary condition to the vector
    apply_lifting(b_potential, [left_form], [BCs])
    b_potential.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    set_bc(b_potential, BCs)
    solver_pot.solve(b_potential, Vh.vector)
    Vh.x.scatter_forward()

    L2_error = fem.form(inner(Vh - V_ex, Vh - V_ex) * dx)
    error_local = fem.assemble_scalar(L2_error)
    error_L2 = np.sqrt(domain.comm.allreduce(error_local, op=MPI.SUM))
    '''if domain.comm.rank == 0:
        print(f"Error_L2 : {error_L2:.2e}")'''

    # Calculate the new current
    A_curr.zeroEntries()
    fem.petsc.assemble_matrix(A_curr, left_form2)  # type: ignore
    A_curr.assemble()
    with b_curr.localForm() as loc_b:
        loc_b.set(0)
    assemble_vector(b_curr, right_form2)
    solver_cur.solve(b_curr, jh.vector)
    jh.x.scatter_forward()

    average_potGrad = fem.form(inner(jh, jh) * dx)
    average = fem.assemble_scalar(average_potGrad)
    L2_average = np.sqrt(domain.comm.allreduce(average, op=MPI.SUM))
    delta = h.max() / (2 * L2_average)

    # Write solution to file
    xdmf_levelset.write_function(phi_h, t)
    xdmf_sigma.write_function(sigma, t)

xdmf_levelset.close()
xdmf_sigma.close()

