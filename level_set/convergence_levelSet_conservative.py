import matplotlib.pyplot as plt
import ufl
from ufl import TestFunction, TrialFunction, dot, dx, grad, FacetNormal, VectorElement, inner
import numpy as np
from petsc4py import PETSc
from mpi4py import MPI

from dolfinx import fem, mesh, io, cpp
from dolfinx.fem.petsc import assemble_vector, assemble_matrix, create_vector, create_matrix, apply_lifting, set_bc

def L2_norm(u, domain):
    norm = fem.assemble_scalar(fem.form(inner(u, u)*dx))
    return np.sqrt(domain.comm.allreduce(norm, op=MPI.SUM))

class exact_solution():
    """
    Class to define the analytical solution for the level set function 
    """
    def __init__(self, t, eps, stab):
        self.t = t
        self.eps = eps
        self.stab = stab

    def __call__(self, x):
        # return x[1] - 0.5 - self.t*np.sin(2*np.pi*x[0])/8 # Rio Tinto Thesis Sonia PAIN
        if (self.stab < 3):
            return np.tanh(-60*(np.sqrt((x[0]-0.5)**2 + (x[1]-0.75)**2) - 0.15))
        else:
            return 0.5 * ( 1 + np.tanh((np.sqrt((x[0]-0.5)**2 + (x[1]-0.75)**2) - 0.15)/ (2 * self.eps)) )

class distance_function():
    """
    Class to define the analytical solution for the level set function 
    """
    def __init__(self, t, eps):
        self.t = t
        self.eps = eps

    def __call__(self, x):
        # return x[1] - 0.5 - self.t*np.sin(2*np.pi*x[0])/8 # Rio Tinto Thesis Sonia PAIN
        # return np.tanh(-60*(np.sqrt((x[0]-0.5)**2 + (x[1]-0.75)**2) - 0.15))
        return np.sqrt((x[0]-0.5)**2 + (x[1]-0.75)**2) - 0.15


class u_exact():
    """
    Class to define the analytical solution for the velocity
    """
    def __init__(self, t, T, stab):
        self.t = t
        self.T = T
        self.stab = stab

    def __call__(self, x):
        values = np.zeros((2, x.shape[1]))
        #values[0] = x[0]*(1-x[0])
        #values[1] = np.sin(2*x[0]*np.pi)/8 + x[0]*(1-x[0])*2*np.pi*self.t*np.cos(2*x[0]*np.pi)/8
        if (self.stab < 3):
            values[0] = - 2 * np.cos(np.pi*x[1]) * np.sin(np.pi*x[0])**2 * np.sin(np.pi*x[1]) * np.cos(np.pi*self.t*0.25)
            values[1] = 2 * np.cos(np.pi*x[0]) * np.sin(np.pi*x[0]) * np.sin(np.pi*x[1])**2 * np.cos(np.pi*self.t*0.25) 
        else:
            values[0] = - np.sin(np.pi*x[0])**2 * np.sin(2*np.pi*x[1]) * np.cos(np.pi*self.t/self.T)
            values[1] = np.sin(2*np.pi*x[0]) * np.sin(np.pi*x[1])**2 * np.cos(np.pi*self.t/self.T)
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
        L2_average = L2_norm(self.jh, self.domain)
        return self.h.max()/(2*L2_average)

def solve_levelSet(N, stab = 2):
    """
    Function to retrive the solution of the level set problem

    Parameter:
    N (int): number of mesh element
    """
    t = 0  # Start time
    T = 4  # Final time

    space_step = 1 / N
    dt = 2.5e-4 #space_step/8  # time step size
    num_steps = int(T / dt)
    domain = mesh.create_rectangle(MPI.COMM_WORLD, [np.array([0, 0]), np.array([1, 1])],
                                   [N, N], mesh.CellType.triangle)
    W = fem.FunctionSpace(domain, ("Lagrange", 1))
    n = FacetNormal(domain)
    # Retrieve the cells dimensions
    tdim = domain.topology.dim
    num_cells = domain.topology.index_map(tdim).size_local
    h = cpp.mesh.h(domain._cpp_object, tdim, range(num_cells))
    eps = h.max()

    #print(space_step, "  ",eps,  "   ", h )

    phi_ex = exact_solution(t, eps, stab)
    csi_ex = distance_function(t, eps)

    phi_n = fem.Function(W)  # Level set function
    phi_n.interpolate(phi_ex)

    csi_n = fem.Function(W)  # Level set function
    csi_n.interpolate(csi_ex)

    phi_h = fem.Function(W)
    phi_h.name = "phi_h"
    phi_h.interpolate(phi_ex)

    def boundary_D(x):
        return np.isclose(x[1], 0)

    dofs_D = fem.locate_dofs_geometrical(W, boundary_D)

    phi_D = fem.Function(W)
    phi_D.interpolate(phi_ex)

    BCs = fem.dirichletbc(phi_D, dofs_D)

    phi, v = ufl.TrialFunction(W), ufl.TestFunction(W)

    vec_fe = VectorElement("Lagrange", domain.ufl_cell(), 1)
    W_vec = fem.FunctionSpace(domain, vec_fe)

    u_ex = u_exact(t,T, stab)
    jh = fem.Function(W_vec)
    jh.interpolate(u_ex)

    delta = delta_func(t, jh, h, domain)
    theta = 0.5
    if(stab == 1):
        w = v + delta() * dot(jh, grad(v))  #### PROBLEMA PER IL CASO CONSERVATIVO, AVREI IL LAPLACIANO
        #a_levelSet = (phi * w * dx + (dt * theta) * dot(jh, grad(phi)) * w * dx)
        #L_levelSet = (phi_n * w * dx - dt * (1 - theta) * dot(jh, grad(phi_n)) * w * dx)
    elif (stab == 2):
        s = delta() * dot(jh, grad(v)) 
        a_stabilization = (phi * s *dx + (dt * theta) * s * dot(grad(phi), jh) *dx)
        L_stabilization = (phi_n * s *dx - dt * (1-theta) * s * dot(grad(phi_n), jh) *dx)
        a_levelSet = (phi * v * dx - (dt * theta) * dot(jh, grad(v)) * phi * dx) #+ a_stabilization
        L_levelSet = (phi_n * v * dx + dt * (1 - theta) * dot(jh, grad(v)) * phi_n * dx) #+ L_stabilization
    else:
        
        gamma = max(jh.x.array)
        delta_stab = 1e-10 
        # Phase field formulation
        a_levelSet = (phi * v * dx - (dt * theta) * dot(jh, grad(v)) * phi * dx +
                      (dt * theta) * eps *gamma*dot(grad(phi), grad(v))*dx ) 
        L_levelSet = (phi_n * v * dx + dt * (1 - theta) * dot(jh, grad(v)) * phi_n * dx -
                      dt * (1 - theta) * eps *gamma*dot(grad(phi_n), grad(v))*dx ) # +
                      #gamma/4 * (1-ufl.tanh(csi_n/(2*eps))**2)*dot(grad(csi_n), grad(v))/L2_norm(grad(csi_n), domain)* dx)


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
    solver.setType(PETSc.KSP.Type.PREONLY)
    solver.getPC().setType(PETSc.PC.Type.LU)
    
    xdmf_levelset = io.XDMFFile(domain.comm, "level_set_noStab.xdmf", "w")
    xdmf_levelset.write_mesh(domain)
    xdmf_levelset.write_function(phi_h, t)

    for i in range(num_steps):
        t += dt

        phi_ex.t = t
        phi_D.interpolate(phi_ex)

        u_ex.t = (t + (t - dt)) / 2
        jh.interpolate(u_ex)

        delta.jh = jh

        A.zeroEntries()
        assemble_matrix(A, bilinear_form, bcs=[]) #BCs
        A.assemble()
        # Update the right hand side reusing the initial vector
        with b.localForm() as loc_b:
            loc_b.set(0)
        assemble_vector(b, linear_form)

        # Apply Dirichlet boundary condition to the vector
        apply_lifting(b, [bilinear_form], [[]]) #BCs
        b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        set_bc(b, [])

        # Solve linear problem
        solver.solve(b, phi_h.vector)
        phi_h.x.scatter_forward()

        # Update solution at previous time step (u_n)
        phi_n.x.array[:] = phi_h.x.array
        phi_n.x.scatter_forward()
        if stab > 2:
            print(phi_n.x.array[:])
            assert np.any(phi_n.x.array < 0), "The kernel is negative"
            csi_n.x.array[:] = eps*np.log(phi_n.x.array + delta_stab / (1 - phi_n.x.array + delta_stab))
            csi_n.x.scatter_forward()
            print(csi_n.x.array) 
            
        xdmf_levelset.write_function(phi_h, t)

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

N = [20]
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
