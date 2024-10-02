import matplotlib.pyplot as plt
import pyvista
import ufl
from ufl import SpatialCoordinate, TestFunction, TrialFunction, dot, ds, dx, grad, div, FacetNormal,VectorElement, inner
import numpy as np
from dolfinx import default_scalar_type
from petsc4py import PETSc
from mpi4py import MPI
from dolfinx.io import XDMFFile
from dolfinx import fem, mesh, io, plot
from dolfinx.fem.petsc import assemble_vector, assemble_matrix, create_vector, apply_lifting, set_bc

def V_exact(mode, interface_position):
    # The analytical solution need to be chosen in order to respect the condition:
    # [sigma Grad(V) . n] = 0
    return lambda x: -mode.cos(mode.pi*x[1]/interface_position)

interface_position = 0.54

#def solve_poisson(domain, interface_position, iter, degree=1):
def solve_poisson(N, interface_position, degree=1):
    domain = mesh.create_rectangle(MPI.COMM_WORLD, [np.array([0, 0]), np.array([1, 1])],
                          [N, N], mesh.CellType.triangle)
    xdmf_sigma = io.XDMFFile(domain.comm, "sigma.xdmf", "w")
    xdmf_sigma.write_mesh(domain)

    xdmf_V = io.XDMFFile(domain.comm, "potential.xdmf", "w")
    xdmf_V.write_mesh(domain)

    Q = fem.FunctionSpace(domain, ("DG", 0))

    def Omega_0(x):
        return x[1] <= interface_position
    def Omega_1(x):
        return x[1] >= interface_position

    x = SpatialCoordinate(domain)
    sigma = fem.Function(Q)
    cells_0 = mesh.locate_entities(domain, domain.topology.dim, Omega_0)
    cells_1 = mesh.locate_entities(domain, domain.topology.dim, Omega_1)
    def anode_conductivity(T):
        return 1. / (5.929e-5 - T * 1.235e-8)
    sigma.x.array[cells_0] = np.full_like(cells_0, 210, dtype=default_scalar_type)
    sigma.x.array[cells_1] = np.full_like(cells_1, anode_conductivity(800), dtype=default_scalar_type)
    xdmf_sigma.write_function(sigma)
    # sigma = fem.Constant(domain, PETSc.ScalarType(500))
    f = -div(sigma * grad(V_ufl(x)))
    W = fem.FunctionSpace(domain, ("Lagrange", 1))
    V = TrialFunction(W)
    csi = TestFunction(W)
    a = dot(sigma * grad(V), grad(csi)) * dx

    # Boundary conditions
    boundaries = [(1, lambda x: np.isclose(x[0], 0)),
                  (2, lambda x: np.isclose(x[0], 1)),
                  (3, lambda x: np.isclose(x[1], 0)),
                  (4, lambda x: np.isclose(x[1], 1)),
                  (5, lambda x: np.isclose(x[1], interface_position))]
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

    domain.topology.create_connectivity(domain.topology.dim - 1, domain.topology.dim)
    with XDMFFile(domain.comm, "facet_tags.xdmf", "w") as xdmf:
        xdmf.write_mesh(domain)
        xdmf.write_meshtags(facet_tag, domain.geometry)
    # We can then inspect individual boundaries using the Threshold-filter in Paraview
    dS = ufl.Measure("dS", domain=domain, subdomain_data=facet_tag, subdomain_id=5)

    g = grad(V_ufl(x))
    n = FacetNormal(domain)
    L = (f * csi * dx) #- (dot(sigma('+')*g('+'), n('+'))*csi('+') - dot(sigma('-')*g('-'),n('-'))*csi('-'))*dS(5)
    # Dirichlet condition
    facets = facet_tag.find(3)
    dofs = fem.locate_dofs_topological(W, fdim, facets)
    facets2 = facet_tag.find(4)
    dofs2 = fem.locate_dofs_topological(W, fdim, facets2)
    BCs = [fem.dirichletbc(PETSc.ScalarType(-1), dofs, W), fem.dirichletbc(PETSc.ScalarType(-np.cos(np.pi/(interface_position))), dofs2, W)]

    # petsc_options={"ksp_type": "preonly", "pc_type": "lu"}
    default_problem = fem.petsc.LinearProblem(a, L, bcs=BCs,
                                              petsc_options={"ksp_type": "cg", "pc_type": "ilu", "monitor_convergence": True})
    '''name_file = f"gmres_output_{iter}.txt"
    gmres_solver = default_problem.solver
    viewer = PETSc.Viewer().createASCII(name_file)
    gmres_solver.view(viewer)'''

    '''solver = PETSc.KSP().create(domain.comm)
    A = assemble_matrix(fem.form(a), bcs=BCs)
    A.assemble()
    solver.setOperators(A)
    solver.setType(PETSc.KSP.Type.CG)
    pc1 = solver.getPC()
    pc1.setType(PETSc.PC.Type.ILU)
    opts = PETSc.Options()
    opts["monitor_convergence"] = True
    b = assemble_vector(fem.form(L))
    apply_lifting(b, [fem.form(a)], [BCs])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    set_bc(b, BCs)
    Vh = fem.Function(W)
    solver.solve(b, Vh.vector)'''

    xdmf_sigma.close()
    V_sol = default_problem.solve()
    xdmf_V.write_function(V_sol)
    xdmf_V.close()
    #return Vh.x.scatter_forward(), V_ufl(x)
    return V_sol, V_ufl(x)

def error_L2_func(Vh, V_ex, degree_raise=3):
    # Create higher order function space
    degree = 1 #Vh.function_space.ufl_element().degree
    family = Vh.function_space.ufl_element().family_name
    mesh = Vh.function_space.mesh
    Q = fem.functionspace(mesh, (family, degree + degree_raise))
    # Interpolate approximate solution
    V_W = fem.Function(Q)
    V_W.interpolate(Vh)

    # Interpolate exact solution, special handling if exact solution
    # is a ufl expression or a python lambda function
    V_ex_W = fem.Function(Q)
    if isinstance(V_ex, ufl.core.expr.Expr):
        u_expr = fem.Expression(V_ex, Q.element.interpolation_points)
        V_ex_W.interpolate(u_expr)
    else:
        V_ex_W.interpolate(V_ex)

    # Compute the error in the higher order function space
    e_W = fem.Function(Q)
    e_W.x.array[:] = V_W.x.array - V_ex_W.x.array

    # Integrate the error
    error = fem.form(ufl.inner(e_W, e_W) * ufl.dx)
    error_local = fem.assemble_scalar(error)
    error_global = mesh.comm.allreduce(error_local, op=MPI.SUM)
    return np.sqrt(error_global)

N = [20, 40] #, 80, 160, 320, 640]
error_L2 = []
error_H1 = []
h = []
mpi_rank = 10
V_numpy = V_exact(np, interface_position)  # which will be used for interpolation
V_ufl = V_exact(ufl, interface_position)  # which will be used for defining the source term

for i in range(len(N)):
    '''domain = mesh.create_rectangle(MPI.COMM_WORLD, [np.array([0, 0]), np.array([1, 1])],
                                   [N[i], N[i]], mesh.CellType.triangle)'''
    Vh, Vex = solve_poisson(N[i], interface_position, i)
    comm = Vh.function_space.mesh.comm

    error_L2 += [error_L2_func(Vh, V_numpy)]

    eh = Vh - Vex
    error_H10 = fem.form(dot(grad(eh), grad(eh)) * dx)
    E_H10 = np.sqrt(comm.allreduce(fem.assemble_scalar(error_H10), op=MPI.SUM))
    error_H1 += [E_H10]

    h += [1. / N[i]]

    if comm.rank == 0:
        mpi_rank = comm.rank
        print(f"h: {h[i]:.2e} Error L2: {error_L2[i]:.2e}")
        print(f"h: {h[i]:.2e} Error H1: {error_H1[i]:.2e}")

'''if mpi_rank == 0:
    plt.figure(figsize=(10, 6))

    plt.loglog(interface_position, error_L2, label='$L^2$ error')
    plt.loglog(interface_position, error_H1, label='$H^1$ error')

    plt.xlabel('N')
    plt.ylabel('Error')
    plt.legend()
    plt.grid(True)
    plt.show()'''
if mpi_rank == 0:
    plt.figure(figsize=(10, 6))

    plt.loglog(N, error_L2, label='$L^2$ error')
    plt.loglog(N, error_H1, label='$H^1$ error')
    plt.loglog(N, h, label='h')
    h_square = [x**2 for x in h]
    plt.loglog(N, h_square, label='$h^2$')
    h_half = [x ** 0.5 for x in h]
    plt.loglog(N, h_half, label='$\\sqrt{h}$')

    plt.xlabel('N')
    plt.ylabel('Error')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=1, fontsize='small')
    plt.grid(True)
    plt.show()
