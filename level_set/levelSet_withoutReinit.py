import matplotlib.pyplot as plt
import ufl
from ufl import dot, dx, grad, FacetNormal, inner
import numpy as np
from petsc4py import PETSc
from mpi4py import MPI
from basix.ufl import element
from dolfinx import fem, mesh, io, cpp
from dolfinx.fem import Constant, Function, functionspace, assemble_scalar, form, locate_dofs_geometrical
from dolfinx.fem.petsc import assemble_vector, assemble_matrix, create_vector, create_matrix, apply_lifting, set_bc

class exact_solution():
    """
    Class to define the analytical solution for the level set function 
    """
    def __init__(self, t):
        self.t = t

    def __call__(self, x):
        #return np.tanh(-60*((x[1] - 10*self.t - (10*self.t**3)/3 - 0.25)**2 - 0.01))
        return np.sqrt((x[0]-0.5)**2 + (x[1]-0.75)**2)- 0.15

class u_exact():
    """
    Class to define the analytical solution for the velocity
    """
    def __init__(self, t, T):
        self.t = t
        self.T = T

    def __call__(self, x):
        values = np.zeros((2, x.shape[1]))
        #values[1] = 10 + 10*self.t**2
        values[0] = -2*np.sin(np.pi*x[1])*np.cos(np.pi*x[1])*np.sin(np.pi*x[0])*np.sin(np.pi*x[0])*np.cos(np.pi*self.t/self.T)
        values[1] = 2*np.sin(np.pi*x[0])*np.cos(np.pi*x[0])*np.sin(np.pi*x[1])*np.sin(np.pi*x[1])*np.cos(np.pi*self.t/self.T)
        return values

class delta_func():
    """
    Class to define the coefficient for the stabilization term in the level set equation
    coeff = h_max / (2*||u||)
    """
    def __init__(self, t, jh, h, domain):
        self.t = t
        self.jh = jh
        self.h = h
        self.domain = domain

    def __call__(self):
        average_potGrad = fem.form(inner(self.jh, self.jh) * dx)
        average = fem.assemble_scalar(average_potGrad)
        L2_average = np.sqrt(self.domain.comm.allreduce(average, op=MPI.SUM))
        return self.h/(2*L2_average)
    
def eikonal_L2proj(domain, u, beta, h, phi, DEBUG = 0, bcs = []):
    #W = fem.FunctionSpace(domain, ("Lagrange", 1))
    elem = element("Lagrange", domain.topology.cell_name(),1)
    W = functionspace(domain, elem)
    p, q = ufl.TrialFunction(W), ufl.TestFunction(W)
    norm_L2 = fem.form(inner(u, u) * dx)
    integral_L2 = fem.assemble_scalar(norm_L2)
    L2 = np.sqrt(domain.comm.allreduce(integral_L2, op=MPI.SUM))
    lambda2 = beta * L2 * h**2 / 2
    print("lambda value for the stabilization: ", lambda2)
    a = (p * q * dx)
    L = (lambda2 * q * (ufl.sqrt(inner(grad(phi), grad(phi)))-1) * dx)
    # Assemble linear system
    A = assemble_matrix(fem.form(a), bcs)
    A.assemble()
    b = assemble_vector(fem.form(L))
    if bcs:
        apply_lifting(b, [a], [bcs])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        set_bc(b, bcs)
    p_h = fem.Function(W)
    # Solve linear system
    solver = PETSc.KSP().create(A.getComm())
    #solver.setType("bcgs")
    #solver.getPC().setType("bjacobi")
    solver.setType("gmres")
    solver.getPC().setType("hypre")  # or "ilu" for direct solve
    solver.rtol = 1.0e-05
    solver.setOperators(A)
    if DEBUG:
        def monitor(ksp, its, rnorm):
            print(f"    Iteration {its}, residual norm {rnorm}")
        solver.setMonitor(monitor)

    solver.solve(b, p_h.x.petsc_vec)
    if DEBUG:
        r, c = A.getDiagonal().array.min(), A.getDiagonal().array.max()
        cond_num = c / r
        print(f"    Condition number: {cond_num}")
    assert solver.reason > 0, f"Solver failed with reason: {solver.reason}"
    p_h.x.scatter_forward()

    # Destroy PETSc linear algebra objects and solver
    solver.destroy()
    A.destroy()
    b.destroy()
    return p_h

def solve_levelSet(N, DEBUG = 1):
    """
    Function to retrive the solution of the level set problem

    Parameter:
    N (int): number of mesh element
    """
    t = 0  # Start time
    #T = 0.05  # Final time
    T = 8  # Final time
    alpha = 0.1
    phi_ex = exact_solution(t)

    h = 1 / N
    dt = alpha * h ** 2  # time step size
    num_steps = int(T / dt)
    domain = mesh.create_rectangle(MPI.COMM_WORLD, [np.array([0, 0]), np.array([1, 1])],
                                   [N, N], mesh.CellType.triangle)
    elem = element("Lagrange", domain.topology.cell_name(),1)
    W = functionspace(domain, elem)
    n = FacetNormal(domain)

    phi_n = fem.Function(W)  # Level set function
    phi_n.interpolate(phi_ex)

    phi_h = fem.Function(W)
    phi_h.name = "phi_h"
    phi_h.interpolate(phi_ex)

    '''def boundary_D(x):
        return np.isclose(x[1], 0)

    dofs_D = fem.locate_dofs_geometrical(W, boundary_D)

    phi_D = fem.Function(W)
    phi_D.interpolate(phi_ex)

    BCs = fem.dirichletbc(phi_D, dofs_D)
    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.isclose(x[1], 0))
    BCs2 = fem.dirichletbc(PETSc.ScalarType(0), fem.locate_dofs_topological(W, fdim, boundary_facets), W)'''

    phi, v = ufl.TrialFunction(W), ufl.TestFunction(W)

    #vec_fe = VectorElement("Lagrange", domain.ufl_cell(), 1)
    vec_fe = element("Lagrange", domain.topology.cell_name(),1, shape=(domain.geometry.dim, ))
    W_vec = functionspace(domain, vec_fe)

    u_ex = u_exact(t, T)
    jh = fem.Function(W_vec)
    jh.interpolate(u_ex)

    # Retrieve the cells dimensions
    #tdim = domain.topology.dim
    #num_cells = domain.topology.index_map(tdim).size_local
    #h = cpp.mesh.h(domain._cpp_object, tdim, range(num_cells))

    delta = delta_func(t, jh, h, domain)

    w = v + delta() * dot(jh, grad(v))
    theta = 0.5
    a_levelSet = (phi * w * dx + (dt * theta) * dot(jh, grad(phi)) * w * dx)
    L_levelSet = (phi_n * w * dx - dt * (1 - theta) * dot(jh, grad(phi_n)) * w * dx)

    beta = 0.8
    eps = 1e-10
    phi_temp = fem.Function(W)
    p = fem.Function(W) # It is the projection of the Eikonal equation and for the moment I consider it P1
    norm_gradPhi = ufl.sqrt(inner(grad(phi_temp), grad(phi_temp)) +eps)
    norm_gradPhi_n = ufl.sqrt(inner(grad(phi_n), grad(phi_n)) +eps)
    #F = (-(phi_temp-phi_n)/dt * w * dx - dot(jh, grad((phi_temp+ phi_n)/2)) * w * dx
    #      - p*dot(grad((phi_temp + phi_n)/2), grad(v))/ norm_module_gradPhi * dx)
    F = (-(phi_temp-phi_n)/dt * w * dx - dot(jh, grad((phi_temp+ phi_n)*theta)) * w * dx
           - p*theta*dot((grad(phi_temp)/norm_gradPhi + grad(phi_n)/norm_gradPhi), grad(v)) * dx)
    delta_phi = ufl.TrialFunction(W)
    ##########################################################
    # HOW THE OTHER TERMS BECAME IN THE NON-LINEAR FOMULATION, 
    # DO I STILL HAVE TO INTEGRATE IN TIME?
    ##########################################################
    DF = (delta_phi/dt * w * dx + theta*dot(jh, grad(delta_phi)) * w * dx
          + p*theta*dot(grad(delta_phi),grad(v))/ norm_gradPhi * dx
          - p*theta*dot(grad(phi_temp), grad(phi_temp)) * dot(grad(delta_phi),grad(v))/ norm_gradPhi**3 * dx)
    #DF = (delta_phi/dt * w * dx + dot(jh, grad(delta_phi)) * w * dx)
    delta_phi_h = fem.Function(W)

    # Preparing linear algebra structures for time dependent problems.
    bilinear_form = fem.form(a_levelSet)
    linear_form = fem.form(L_levelSet)

    A2 = create_matrix(fem.form(DF))
    b2 = create_vector(fem.form(F))

    A = create_matrix(bilinear_form)
    # A = assemble_matrix(bilinear_form, bcs=[BCs])
    # A.assemble()
    b = create_vector(linear_form)

    # We create a linear algebra solver using PETSc, and assign the matrix A to the solver
    solver = PETSc.KSP().create(domain.comm)
    solver.setOperators(A)
    solver.setType(PETSc.KSP.Type.PREONLY)
    solver.getPC().setType(PETSc.PC.Type.LU)

    solver2 = PETSc.KSP().create(domain.comm)
    solver2.setOperators(A2)
    solver2.setType(PETSc.KSP.Type.PREONLY)
    solver2.getPC().setType(PETSc.PC.Type.LU)
    if DEBUG:
        def monitor(ksp, its, rnorm):
            print(f"    Iteration {its}, residual norm {rnorm}")
        solver2.setMonitor(monitor)
    tol = 1e-3        # Tolerance for Newton solver
    max_iter = 100

    xdmf_levelset = io.XDMFFile(domain.comm, "level_set_conserv.xdmf", "w")
    xdmf_levelset.write_mesh(domain)
    xdmf_levelset.write_function(phi_n, t)

    for i in range(num_steps):
        t += dt

        phi_ex.t = t
        #phi_D.interpolate(phi_ex)

        #u_ex.t = (t + (t - dt)) / 2
        u_ex.t = t
        jh.interpolate(u_ex)

        delta.jh = jh
        if i == 0:
            print("Get initial guess for phi^(n+1)")
            A.zeroEntries()
            #assemble_matrix(A, bilinear_form, bcs=[BCs])
            assemble_matrix(A, bilinear_form, bcs=[])
            A.assemble()
            # Update the right hand side reusing the initial vector
            with b.localForm() as loc_b:
                loc_b.set(0)
            assemble_vector(b, linear_form)

            # Apply Dirichlet boundary condition to the vector
            '''apply_lifting(b, [bilinear_form], [[BCs]])
            b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
            set_bc(b, [BCs])'''

            # Solve linear problem
            solver.solve(b, phi_h.x.petsc_vec)
            phi_h.x.scatter_forward()

        err = 1.
        phi_temp.interpolate(phi_h)
        k = 0
        print("Newton loop at time step: ", i)
        while (err > tol) and (k <= max_iter):
            print("    Newton iteration: ", k)
            p = eikonal_L2proj(domain, jh, beta, h, phi_temp)
            # Command to print all the elements inside an array
            #np.set_printoptions(threshold=np.inf)
            #print("Print value of p: ", p.x.array[:])
            #print("Print |grad(phi)|", fem.assemble_scalar(fem.form(norm_module_gradPhi * dx)))
            A2.zeroEntries()
            #assemble_matrix(A2, fem.form(DF), bcs=[BCs2])
            assemble_matrix(A2, fem.form(DF), bcs=[])
            A2.assemble()
            # Update the right hand side reusing the initial vector
            with b2.localForm() as loc_b:
                loc_b.set(0)
            assemble_vector(b2, fem.form(F))
            # Apply Dirichlet boundary condition to the vector
            ''' apply_lifting(b2, [fem.form(DF)], [[BCs2]])
            b2.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
            set_bc(b2, [BCs2])'''
            # Solve linear problem
            solver2.solve(b2, delta_phi_h.x.petsc_vec)
            if DEBUG:
                r, c = A2.getDiagonal().array.min(), A2.getDiagonal().array.max()
                cond_num = c / r
                print(f"    Condition number: {cond_num}")
                #print(A2.view())
            delta_phi_h.x.scatter_forward()
            phi_temp.x.array[:] += delta_phi_h.x.array[:] 
            err = error_L2_func(phi_temp, phi_h)
            phi_h.interpolate(phi_temp)
            k += 1
            print("    Error: ", err)
        assert k < max_iter, "The system does not converge, you reach the max amount of iterations."
        # Update solution at previous time step (u_n)
        phi_n.x.array[:] = phi_temp.x.array
        phi_n.x.scatter_forward()
        xdmf_levelset.write_function(phi_n, t)
    xdmf_levelset.close()

    return phi_n, phi_D

def error_L2_func(Vh, V_ex, degree_raise=3):
    """
    Function to calculate the L2 error
    
    Parameter:
    Vh (function): numerical solution
    V_ex (function): analytical dolution
    degree_raise (int): dimension of the higher order FE space
    """
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

    V_ex_W.interpolate(V_ex)

    # Compute the error in the higher order function space
    e_W = fem.Function(Q)
    e_W.x.array[:] = V_W.x.array - V_ex_W.x.array

    # Integrate the error
    error = fem.form(ufl.inner(e_W, e_W) * ufl.dx)
    error_local = fem.assemble_scalar(error)
    error_global = mesh.comm.allreduce(error_local, op=MPI.SUM)
    return np.sqrt(error_global)

N = [80]
error_L2 = []
h = []
mpi_rank = 5
for i in range(len(N)):
    phi_n, phi_D = solve_levelSet(N[i])
    comm = phi_n.function_space.mesh.comm
    error_L2 += [error_L2_func(phi_n, phi_D)]

    h += [1. / N[i]]

    if comm.rank == 0:
        mpi_rank = comm.rank
        print(f"h: {h[i]:.2e} Error L2: {error_L2[i]:.2e}")

if mpi_rank == 0:
    plt.figure(figsize=(10, 6))

    plt.loglog(N, error_L2, label='$L^2$ error')
    plt.loglog(N, h)
    h_square = [x**2 for x in h]
    plt.loglog(N, h_square)

    plt.xlabel('N')
    plt.ylabel('Error')
    plt.legend()
    plt.grid(True)
    plt.show()
