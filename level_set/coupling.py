from ufl import TestFunction, TrialFunction, dot, dx, grad, inner
import numpy as np
import ufl
from mpi4py import MPI
from dolfinx import fem, mesh, cpp
from dolfinx.fem.petsc import assemble_matrix, create_vector, create_matrix

from utilities import (SmoothedHeaviside, LinearSolver, project,
                       setup_output_files, define_facet_tags)
from boundary_conditions import apply_boundary_conditions

#####################################################################
# Script to solve a two media problem solving a laplace equacion for
# the eletrctric potential with discontinuous conductivity and by
# modelling the interface movement through the Level Set function.
#####################################################################


class ExactSolution():
    """
    Class to define the analytical solution for the level set function,
    STEP FUNCTION in this case
    """
    def __init__(self, t):
        self.t = t

    def __call__(self, x):
        value = np.where(x[0] < 0.35, x[1] - 0.5,
                         np.where(x[0] > 0.65, x[1] - 0.5, x[1] - 0.7))
        return value


# Define temporal parameters
t = 0  # Start time
T = 0.1  # Final time

# Define mesh
nx, ny = 120, 120
space_step = 1/nx
alpha = 2
dt = alpha * space_step**2  # time step size
num_steps = int(T/dt)

domain = mesh.create_rectangle(MPI.COMM_WORLD, [np.array([0, 0]),
                                                np.array([1, 1])],
                               [nx, ny], mesh.CellType.triangle)

# Define the needed function space
# For the Level Set and the Potential
W = fem.functionspace(domain, ("Lagrange", 1))
# For the Conductivity
D = fem.functionspace(domain, ("DG", 0))
# For the current density
J = fem.functionspace(domain, ("Lagrange", 1, (domain.geometry.dim, )))

# Level set function
phi_n = fem.Function(W)
phi_ex = ExactSolution(t)
phi_n.interpolate(phi_ex)
# Potential - definition of the conductivity in the two media
sigma = fem.Function(D)

# Post-processing for level set and potential
xdmf_levelset = setup_output_files(domain, "problem.xdmf")
xdmf_levelset.write_mesh(domain)


###############################################################################
# Define the conductivity in the two media
def anode_conductivity(T):
    return 1. / (5.929e-5 - T * 1.235e-8)


conductivity_anode = anode_conductivity(800)
conductivity_bath = 210

# Retrieve the cells dimensions
tdim = domain.topology.dim
num_cells = domain.topology.index_map(tdim).size_local
cells = np.arange(num_cells, dtype=np.int32)
h = cpp.mesh.h(domain._cpp_object, tdim, cells)
epsilon = h.max()
'''
You cannot interpolate. A quadrature function space are point evaluations.
They do not have an underlying polynomial that can be used as the basis on
a single cell. You can project a quadrature function into any space though.
'''
phi_n_project = fem.Function(D)
project(phi_n, phi_n_project)
smoothed_heaviside = SmoothedHeaviside(phi_n_project.x.petsc_vec, epsilon)
sigma.x.array[:] = np.add(conductivity_bath,
                          (conductivity_anode - conductivity_bath)
                          * smoothed_heaviside(phi_n_project.x.array))
sigma.x.scatter_forward()
xdmf_sigma = setup_output_files(domain, "sigma.xdmf")
xdmf_sigma.write_mesh(domain)
xdmf_sigma.write_function(sigma, t)

# Variational problem and solver for the potential
V = TrialFunction(W)
csi = TestFunction(W)

a = dot(sigma * grad(V), grad(csi)) * dx

facet_tag = define_facet_tags(domain)

domain.topology.create_connectivity(domain.topology.dim-1, domain.topology.dim)
'''with XDMFFile(domain.comm, "facet_tags.xdmf", "w") as xdmf:
    xdmf.write_mesh(domain)
    xdmf.write_meshtags(facet_tag, domain.geometry)'''
# We can then inspect individual boundaries using Threshold-filter in Paraview

BCs, L, flag = apply_boundary_conditions(domain, facet_tag, csi, 0)

left_form = fem.form(a)
right_form = fem.form(L)

A_potential = assemble_matrix(left_form, bcs=BCs)
b_potential = create_vector(right_form)
# Solution vector
Vh = fem.Function(W)

solver_pot = LinearSolver(domain.comm, A_potential, b_potential,
                          left_form, right_form, BCs, flag)
# Solve the system and update Vh (the PETSc vector for the potential).
solver_pot.solve(Vh.x)

# Post-processing for the potential
Vh.name = "V"
xdmf_levelset.write_function(Vh, t)

###############################################################################
# # Current density
jh = fem.Function(J)
jh_expr = fem.Expression(ufl.as_vector((-sigma * Vh.dx(0), -sigma * Vh.dx(1))),
                         J.element.interpolation_points())
jh.interpolate(jh_expr)

xdmf_current = setup_output_files(domain, "current.xdmf")
xdmf_current.write_function(jh, t)

###############################################################################
# Level set
phi_h = fem.Function(W)
phi_h.name = "phi_h"
phi_h.interpolate(phi_ex)
xdmf_levelset.write_function(phi_h, t)


def boundary_D(x):
    return np.isclose(x[1], 0)


dofs_D = fem.locate_dofs_geometrical(W, boundary_D)
phi_D = fem.Function(W)
phi_D.interpolate(phi_ex)
# You have to update those every iteration
BCs_phi = [fem.dirichletbc(phi_D, dofs_D)]

phi, v = ufl.TrialFunction(W), ufl.TestFunction(W)


class DeltaFunc():
    def __init__(self, t, jh, h):
        self.t = t
        self.jh = jh
        self.h = h

    def __call__(self, x):
        average_curr = fem.form(inner(self.jh, self.jh) * dx)
        average = fem.assemble_scalar(average_curr)
        L2_average = np.sqrt(domain.comm.allreduce(average, op=MPI.SUM))

        return self.h.max()/(2*L2_average)


delta = DeltaFunc(t, jh, h)
# Add a coefficient k to slow done the velocity of consumption of the anode
k = -0.01
w = v + delta(t) * dot(k*jh, grad(v))
theta = 0.5
a_levelSet = (phi * w * dx -
              (dt * theta) * dot(k*sigma*grad(Vh), grad(phi)) * w * dx)
L_levelSet = (phi_n * w * dx +
              dt * (1-theta) * dot(k*sigma*grad(Vh), grad(phi_n)) * w * dx)
# Preparing linear algebra structures for time dependent problems.
bilinear_form = fem.form(a_levelSet)
linear_form = fem.form(L_levelSet)
# Since the left hand side depends on the potential that is recalculated each
# iteration, we have to update both the left hand side and the right hand side,
# which is dependent on the previous time step u_n.
A = create_matrix(bilinear_form)
b = create_vector(linear_form)
# Instantiate the unified solver for this system.
solver_phi = LinearSolver(domain.comm, A, b, bilinear_form,
                          linear_form, BCs_phi, True)

###############################################################################
# Resolution of the problem
for i in range(num_steps):
    t += dt
    phi_ex.t = t
    phi_D.interpolate(phi_ex)
    # Calculate the new Level set
    solver_phi.solve(phi_h.x)
    # Update solution at previous time step (u_n)
    phi_n.x.petsc_vec[:] = phi_h.x.petsc_vec
    # Update the conductivity
    project(phi_n, phi_n_project)
    smoothed_heaviside.phi_n_project = phi_n_project.x.petsc_vec
    sigma.x.petsc_vec[:] = (conductivity_bath +
                            (conductivity_anode-conductivity_bath)
                            * smoothed_heaviside(phi_n_project.x.petsc_vec))
    sigma.x.scatter_forward()
    # Calculate the new potential
    solver_pot.solve(Vh.x)
    # Calculate the new current density
    jh_expr = fem.Expression(ufl.as_vector((-sigma * Vh.dx(0),
                                            -sigma * Vh.dx(1))),
                             J.element.interpolation_points())
    jh.interpolate(jh_expr)
    # Update stabilization parameter for Level Set
    average_potGrad = fem.form(inner(jh, jh) * dx)
    average = fem.assemble_scalar(average_potGrad)
    L2_average = np.sqrt(domain.comm.allreduce(average, op=MPI.SUM))
    delta = h.max() / (2 * L2_average)

    # Write solution to file
    xdmf_levelset.write_function(phi_h, t)
    xdmf_levelset.write_function(Vh, t)
    xdmf_sigma.write_function(sigma, t)
    xdmf_current.write_function(jh, t)

xdmf_levelset.close()
xdmf_sigma.close()
xdmf_current.close()
