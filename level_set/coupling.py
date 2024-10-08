from ufl import SpatialCoordinate, TestFunction, TrialFunction, dot, dx, grad, div, FacetNormal, inner
import numpy as np
import ufl

from petsc4py import PETSc
from mpi4py import MPI
from basix.ufl import element
from dolfinx import fem, mesh, io, cpp
from dolfinx.fem.petsc import assemble_vector, assemble_matrix, create_vector, create_matrix, apply_lifting, set_bc

def project(e, target_func, bcs=[]):
    """Project UFL expression.
    Note
    ----
    This method solves a linear system (using KSP defaults).

    Parameters:
    e (function): function to be projected
    target_func (function): new projected function
    bcs (function): possible boundary conditions
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


# x = ufl.SpatialCoordinate(domain)
# Define temporal parameters
t = 0  # Start time
T = 0.1  # Final time
alpha = 2

# Define mesh
nx, ny = 120, 120
space_step = 1/nx
dt = alpha * space_step**2 # time step size
num_steps = int(T/dt)

domain = mesh.create_rectangle(MPI.COMM_WORLD, [np.array([0, 0]), np.array([1, 1])],
                               [nx, ny], mesh.CellType.triangle)
W = fem.FunctionSpace(domain, ("Lagrange", 1))
n = FacetNormal(domain)

phi_n = fem.Function(W)  # Level set function

# Initial condition for phi s.t. it is negative in the anode and positive in the bath
# The unit normal in this case (n = Grad(phi)/||Grad(phi)||) will always point inside the bath (phi >0)
class exact_solution():
    """
    Class to define the analytical solution for the level set function, STEP FUNCTION
    """
    def __init__(self, t):
        self.t = t
    def __call__(self, x):
        # return x[1] - 0.5
        value = np.zeros(x.shape[1])
        value[x[0] < 0.35] = x[1][x[0] < 0.35] - 0.5
        value[x[0] > 0.65] = x[1][x[0] > 0.65] - 0.5
        value[(x[0] >= 0.35) & (x[0] <= 0.65)] = x[1][(x[0] >= 0.35) & (x[0] <= 0.65)] - 0.7
        return value
phi_ex = exact_solution(t)

phi_n.interpolate(phi_ex)

# Boundary condition for phi, inflow at the top of the domain = 0, so nothing as to be specified

# Declare variables for weak formulations
# Level set
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

# You have to update those every iteration
BCs_phi = fem.dirichletbc(phi_D, dofs_D) 

#######################################################################################################################
# Potential

D = fem.FunctionSpace(domain, ("DG", 0))
sigma = fem.Function(D)
def anode_conductivity(T):
    return 1. / (5.929e-5 - T * 1.235e-8)

conductivity_anode = anode_conductivity(800)
conductivity_bath = 210
class smoothed_heaviside_func():
    """ 
    Class to define the Heaviside function for the definition of the conductivities 
    in the two media.
    """
    def __init__(self, phi_n_project, epsilon):
        self.phi_n_project = phi_n_project
        self.epsilon = epsilon

    def __call__(self, x):
        value = np.zeros(len(self.phi_n_project))
        value[self.phi_n_project < -self.epsilon] = 0
        value[self.phi_n_project > self.epsilon] = 1
        value[(self.phi_n_project >= -self.epsilon) & (self.phi_n_project <= self.epsilon)] = \
            (0.5*(1 + (self.phi_n_project
                       [(self.phi_n_project >= -self.epsilon) & (self.phi_n_project <= self.epsilon)]/epsilon)
                  + np.sin((np.pi*self.phi_n_project
                    [(self.phi_n_project >= -self.epsilon) & (self.phi_n_project <= self.epsilon)])/self.epsilon)/np.pi))
        return value

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
smoothed_heaviside = smoothed_heaviside_func(phi_n_project.vector.array, epsilon)
sigma.x.array[:] = conductivity_bath + (conductivity_anode - conductivity_bath)*smoothed_heaviside(phi_n_project.vector.array)
sigma.x.scatter_forward()
xdmf_sigma = io.XDMFFile(domain.comm, "sigma.xdmf", "w")
xdmf_sigma.write_mesh(domain)
xdmf_sigma.write_function(sigma, t)

#######################################################################################################################
# Variational problem and solver for the potential

V = TrialFunction(W)
csi = TestFunction(W)

a = dot(sigma * grad(V), grad(csi)) * dx

# We start by identifying the facets contained in each boundary and create a custom integration measure ds
boundaries = [(1, lambda x: np.isclose(x[0], 0)),
              (2, lambda x: np.isclose(x[0], 1)),
              (3, lambda x: np.isclose(x[1], 0)),
              (4, lambda x: np.isclose(x[1], 1))]
# We now loop through all the boundary conditions and create MeshTags identifying the facets for each boundary condition
facet_indices, facet_markers = [], []
fdim = domain.topology.dim - 1
for (marker, locator) in boundaries:
    facets = mesh.locate_entities(domain, fdim, locator)
    facet_indices.append(facets)
    facet_markers.append(np.full_like(facets, marker))
facet_indices = np.hstack(facet_indices).astype(np.int32)
facet_markers = np.hstack(facet_markers).astype(np.int32)
sorted_facets = np.argsort(facet_indices)
facet_tag = mesh.meshtags(domain, fdim, facet_indices[sorted_facets], facet_markers[sorted_facets])

domain.topology.create_connectivity(domain.topology.dim-1, domain.topology.dim)
'''with XDMFFile(domain.comm, "facet_tags.xdmf", "w") as xdmf:
    xdmf.write_mesh(domain)
    xdmf.write_meshtags(facet_tag, domain.geometry)'''
# We can then inspect individual boundaries using the Threshold-filter in Paraview
ds = ufl.Measure("ds", domain=domain, subdomain_data=facet_tag)

# Dirichlet condition
facets = facet_tag.find(3)
dofs = fem.locate_dofs_topological(W, fdim, facets)
facets2 = facet_tag.find(4)
dofs2 = fem.locate_dofs_topological(W, fdim, facets2)
#BCs = [fem.dirichletbc(PETSc.ScalarType(0.26), dofs, W), fem.dirichletbc(PETSc.ScalarType(1.9), dofs2, W)]
#L = fem.Constant(domain, PETSc.ScalarType(0.))*csi*dx
BCs = []
L = 300 * csi * ds(4) - 300 * csi * ds(3)

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

Vh.name = "V"
xdmf_levelset.write_function(Vh, t)

#######################################################################################################################
J = fem.FunctionSpace(domain, ("Lagrange", 1, (domain.geometry.dim, )))
jh = fem.Function(J)
jh_expr = fem.Expression(ufl.as_vector((-sigma * Vh.dx(0), -sigma * Vh.dx(1))), J.element.interpolation_points())
jh.interpolate(jh_expr)

xdmf_current = io.XDMFFile(domain.comm, "current.xdmf", "w")
xdmf_current.write_mesh(domain)
xdmf_current.write_function(jh, t)
#######################################################################################################################
# Variational problem and solver for level set function

phi, v = ufl.TrialFunction(W), ufl.TestFunction(W)
class delta_func():
    def __init__(self, t, jh, h):
        self.t = t
        self.jh = jh
        self.h = h

    def __call__(self, x):
        average_curr= fem.form(inner(self.jh, self.jh) * dx)
        average = fem.assemble_scalar(average_curr)
        L2_average = np.sqrt(domain.comm.allreduce(average, op=MPI.SUM))

        return self.h.max()/(2*L2_average)

delta = delta_func(t, jh, h)

# Add a coefficient k to slow done the velocity of consumption of the anode
k = -0.01
w = v + delta(t) * dot(k*jh, grad(v))
theta = 0.5
a_levelSet = (phi * w * dx + (dt * theta) * dot(k*jh, grad(phi)) * w * dx)
L_levelSet = (phi_n * w * dx - dt * (1-theta) * dot(k*jh, grad(phi_n)) * w * dx)

#Preparing linear algebra structures for time dependent problems.
bilinear_form = fem.form(a_levelSet)
linear_form = fem.form(L_levelSet)

# Since the left hand side depends on the potential that is recalculated each iteration, we have to update
# both the left hand side and the right hand side, which is dependent on the previous time step u_n.
A = create_matrix(bilinear_form)
b = create_vector(linear_form)

# We create a linear algebra solver using PETSc, and assign the matrix A to the solver
solver = PETSc.KSP().create(domain.comm)
solver.setOperators(A)
solver.setType(PETSc.KSP.Type.PREONLY)
solver.getPC().setType(PETSc.PC.Type.LU)

for i in range(num_steps):
    t += dt

    phi_ex.t = t
    phi_D.interpolate(phi_ex)

    A.zeroEntries()
    assemble_matrix(A, bilinear_form, bcs=[BCs_phi])
    A.assemble()
    # Update the right hand side reusing the initial vector
    with b.localForm() as loc_b:
        loc_b.set(0)
    assemble_vector(b, linear_form)
    # Apply Dirichlet boundary condition to the vector
    apply_lifting(b, [bilinear_form], [[BCs_phi]])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    set_bc(b, [BCs_phi])

    # Solve linear problem
    solver.solve(b, phi_h.vector)
    phi_h.x.scatter_forward()

    # Update solution at previous time step (u_n)
    phi_n.x.array[:] = phi_h.x.array

    project(phi_n, phi_n_project)
    smoothed_heaviside.phi_n_project = phi_n_project.vector.array
    sigma.x.array[:] = conductivity_bath + (conductivity_anode - conductivity_bath) * smoothed_heaviside(phi_n_project.vector.array)
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

    jh_expr = fem.Expression(ufl.as_vector((-sigma * Vh.dx(0), -sigma * Vh.dx(1))), J.element.interpolation_points())
    jh.interpolate(jh_expr)

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
