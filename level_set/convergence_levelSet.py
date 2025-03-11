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

def solve_levelSet(N):
    """
    Function to retrive the solution of the level set problem

    Parameter:
    N (int): number of mesh element
    """
    t = 0  # Start time
    T = 8  # Final time # 0.05 for anysotropic case, 8.0 for vortex case
    alpha = 0.1
    phi_ex = exact_solution(t)

    h = 1 / N
    dt = alpha * h  # time step size
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

    #def boundary_D(x):
        #return np.isclose(x[1], 0)

    #dofs_D = fem.locate_dofs_geometrical(W, boundary_D)

    phi_D = fem.Function(W)
    phi_D.interpolate(phi_ex)

    #BCs = fem.dirichletbc(phi_D, dofs_D)

    phi, v = ufl.TrialFunction(W), ufl.TestFunction(W)

    vec_fe = element("Lagrange", domain.topology.cell_name(),1, shape=(domain.geometry.dim, ))
    W_vec = functionspace(domain, vec_fe)

    u_ex = u_exact(t, T)
    jh = fem.Function(W_vec)
    jh.interpolate(u_ex)

    # Retrieve the cells dimensions
    '''tdim = domain.topology.dim
    num_cells = domain.topology.index_map(tdim).size_local
    h = cpp.mesh.h(domain._cpp_object, tdim, range(num_cells))'''

    delta = delta_func(t, jh, h, domain)

    w = v + delta() * dot(jh, grad(v))
    theta = 0.5
    a_levelSet = (phi * w * dx + (dt * theta) * dot(jh, grad(phi)) * w * dx)
    L_levelSet = (phi_n * w * dx - dt * (1 - theta) * dot(jh, grad(phi_n)) * w * dx)

    # Preparing linear algebra structures for time dependent problems.
    bilinear_form = fem.form(a_levelSet)
    linear_form = fem.form(L_levelSet)

    A = create_matrix(bilinear_form)
    # A = assemble_matrix(bilinear_form, bcs=[BCs])
    # A.assemble()
    b = create_vector(linear_form)

    # We create a linear algebra solver using PETSc, and assign the matrix A to the solver
    solver = PETSc.KSP().create(domain.comm)
    solver.setOperators(A)
    #solver.setType(PETSc.KSP.Type.PREONLY)
    solver.setType(PETSc.KSP.Type.GMRES)
    #solver.getPC().setType(PETSc.PC.Type.LU)
    solver.getPC().setType(PETSc.PC.Type.ILU)
    solver.setTolerances(rtol=1e-5, max_it=1000)
    if 1:
        def monitor(ksp, its, rnorm):
            print(f"    Iteration {its}, residual norm {rnorm}")
        solver.setMonitor(monitor)

    xdmf_levelset = io.XDMFFile(domain.comm, "level_set.xdmf", "w")
    xdmf_levelset.write_mesh(domain)
    xdmf_levelset.write_function(phi_n, t)

    # Project the gradient norm onto a DG0 space (P0)
    #V0 = fem.FunctionSpace(domain, ("DG", 0))
    #grad_norm_p0 = fem.Function(V0)
    
    for i in range(num_steps):
        print("Time step: ", i)
        t += dt

        phi_ex.t = t
        #phi_D.interpolate(phi_ex)

        #u_ex.t = (t + (t - dt)) / 2
        u_ex.t = t
        jh.interpolate(u_ex)

        delta.jh = jh

        #distance = fem.form(inner(grad(phi_n), grad(phi_n)) * dx)
        #average_dist = fem.assemble_scalar(distance)
        #L2_average_dist = np.sqrt(domain.comm.allreduce(average_dist, op=MPI.SUM))
        # if domain.comm.rank == 0:
        # print(f"Gradient distance : {L2_average_dist:.2e}")

        A.zeroEntries()
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
        if 1:
            r, c = A.getDiagonal().array.min(), A.getDiagonal().array.max()
            cond_num = c / r
            print(f"    Condition number: {cond_num}")
        phi_h.x.scatter_forward()

        # Update solution at previous time step (u_n)
        phi_n.x.array[:] = phi_h.x.array
        phi_n.x.scatter_forward()

        # Compute the gradient of phi, which will be a P0 function
        '''grad_phi = ufl.grad(phi_n)
        grad_norm = ufl.sqrt(ufl.dot(grad_phi, grad_phi))
        grad_norm_squared = ufl.inner(grad_phi, grad_phi)
        integral = fem.form(grad_norm_squared * ufl.dx)
        norm_squared = fem.assemble_scalar(integral)
        grad_norm_value = np.sqrt(norm_squared)
        grad_norm_value_global = MPI.COMM_WORLD.allreduce(grad_norm_value, op=MPI.SUM)
        # Use fem.Expression to handle interpolation
        expr = fem.Expression(grad_norm, V0.element.interpolation_points())
        grad_norm_p0.interpolate(expr)
        # Collect the gradient norm values from each element and find the minimum
        grad_norm_values = grad_norm_p0.x.array
        min_grad_norm = np.min(grad_norm_values)
        result = np.sum(grad_norm_values - 1) / grad_norm_values.shape[0]
        # Handle the global minimum in parallel
        min_grad_norm_global = MPI.COMM_WORLD.allreduce(min_grad_norm, op=MPI.MIN)
        result_global = MPI.COMM_WORLD.allreduce(result, op=MPI.SUM)
        if MPI.COMM_WORLD.rank == 0:
            print(f"Minimum gradient norm: {min_grad_norm_global}")
            print(f"Norm of the gradient: {grad_norm_value_global}")
            print(f"MAE (Mean absolut error): {result_global}")
        '''
        xdmf_levelset.write_function(phi_n, t)
    xdmf_levelset.close()

    return phi_n , phi_D

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
