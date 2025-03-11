import ufl
import numpy as np
from petsc4py import PETSc
from dolfinx import fem, io, mesh
from dolfinx.fem.petsc import (assemble_vector, assemble_matrix,
                               apply_lifting, set_bc)


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
    if bcs:
        apply_lifting(b, [a], [bcs])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD,
                      mode=PETSc.ScatterMode.REVERSE)
        set_bc(b, bcs)

    # Solve linear system
    solver = PETSc.KSP().create(A.getComm())
    # solver.setType("bcgs")
    # solver.getPC().setType("bjacobi")
    solver.setType("gmres")
    solver.getPC().setType("hypre")  # or "ilu" for direct solve
    solver.rtol = 1.0e-05
    solver.setOperators(A)

    def monitor(ksp, its, rnorm):
        print(f"Iteration {its}, residual norm {rnorm}")

    solver.setMonitor(monitor)

    solver.solve(b, target_func.x.petsc_vec)
    r, c = A.getDiagonal().array.min(), A.getDiagonal().array.max()
    cond_num = c / r
    print(f"Condition number: {cond_num}")
    assert solver.reason > 0, f"Solver failed with reason: {solver.reason}"
    target_func.x.scatter_forward()

    # Destroy PETSc linear algebra objects and solver
    solver.destroy()
    A.destroy()
    b.destroy()


def setup_output_files(domain, name):
    xdmf_file = io.XDMFFile(domain.comm, name, "w")
    return xdmf_file


def define_facet_tags(domain):
    """
    Funtion to define the facet of a 2D domain

    Parameters:
    domain (dolfinx.mesh): geometry of the domain
    """
    boundaries = [(1, lambda x: np.isclose(x[0], 0)),
                  (2, lambda x: np.isclose(x[0], 1)),
                  (3, lambda x: np.isclose(x[1], 0)),
                  (4, lambda x: np.isclose(x[1], 1))]
    facet_indices, facet_markers = [], []
    fdim = domain.topology.dim - 1
    for (marker, locator) in boundaries:
        facets = mesh.locate_entities(domain, fdim, locator)
        facet_indices.append(facets)
        facet_markers.append(np.full_like(facets, marker))
    facet_indices = np.hstack(facet_indices).astype(np.int32)
    facet_markers = np.hstack(facet_markers).astype(np.int32)
    sorted_facets = np.argsort(facet_indices)
    facet_tag = mesh.meshtags(domain, fdim, facet_indices[sorted_facets],
                              facet_markers[sorted_facets])
    return facet_tag


class SmoothedHeaviside:
    """
    Class to define the Heaviside function for the definition of the
    conductivities in the two media.

    Parameters:
    phi_n_project [petsc4py.PETSc.Vec] : Input vector
    epsilon [float] : Smoothing parameter
    """
    def __init__(self, phi_n_project, epsilon):
        self.phi_n_project = phi_n_project
        self.epsilon = epsilon

    def __call__(self, x):
        # Convert PETSc Vec to NumPy array
        phi_array = self.phi_n_project.getArray()

        # Create output array
        value = np.zeros_like(phi_array)

        # Apply Heaviside function with smoothing
        value[phi_array < -self.epsilon] = 0
        value[phi_array > self.epsilon] = 1
        mask = (phi_array >= -self.epsilon) & (phi_array <= self.epsilon)
        value[mask] = (0.5 *
                       (1 + (phi_array[mask] / self.epsilon) +
                        np.sin(np.pi * phi_array[mask] / self.epsilon)/np.pi))

        return value


class LinearSolver:
    """
    A unified class that encapsulates both the PETSc solver setup and the
    assembly/solve process for a linear system defined by a bilinear form and
    a linear form.

    Attributes:
        comm: The MPI communicator.
        A: The PETSc matrix.
        b: The PETSc vector (RHS).
        left_form: The bilinear form used for assembling A.
        right_form: The linear form used for assembling b.
        BCs: A list of Dirichlet boundary conditions.
        flag: A boolean flag that, if True, applies additional BC adjustments
        to the RHS.
        solver_type: The PETSc KSP solver type (default is PREONLY).
        pc_type: The PETSc preconditioner type (default is LU).
        solver: The PETSc KSP solver instance.
    """
    def __init__(self, comm, A, b, left_form, right_form, BCs, flag,
                 solver_type=PETSc.KSP.Type.PREONLY, pc_type=PETSc.PC.Type.LU):
        self.comm = comm
        self.A = A
        self.b = b
        self.left_form = left_form
        self.right_form = right_form
        self.BCs = BCs
        self.flag = flag
        self.solver_type = solver_type
        self.pc_type = pc_type
        self.solver = self.setup_solver()

    def setup_solver(self):
        """
        Set up the PETSc KSP solver with the provided matrix A.
        """
        solver = PETSc.KSP().create(self.comm)
        solver.setOperators(self.A)
        solver.setType(self.solver_type)
        solver.getPC().setType(self.pc_type)
        return solver

    def assemble_system(self):
        """
        Assemble the matrix and right-hand side vector for the linear system.
        """
        # Assemble the matrix.
        self.A.zeroEntries()
        fem.petsc.assemble_matrix(self.A, self.left_form, bcs=self.BCs)
        self.A.assemble()

        # Assemble the right-hand side vector.
        with self.b.localForm() as loc_b:
            loc_b.set(0)
        assemble_vector(self.b, self.right_form)

        # Optionally apply the Dirichlet boundary conditions to the RHS.
        if self.flag:
            apply_lifting(self.b, [self.left_form], [self.BCs])
            self.b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES,
                               mode=PETSc.ScatterMode.REVERSE)
            set_bc(self.b, self.BCs)

    def solve(self, solution_vec):
        """
        Assemble the system and solve for the given solution vector.

        Parameters:
            solution_vec: The PETSc vector to store the solution.
        """
        self.assemble_system()
        self.solver.solve(self.b, solution_vec.petsc_vec)
        solution_vec.scatter_forward()
