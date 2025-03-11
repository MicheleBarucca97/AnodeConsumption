from ufl import dx, Measure
from petsc4py import PETSc
from mpi4py import MPI
from dolfinx import fem, mesh
import numpy as np

from utilities import setup_output_files


# I could write everything inside a class and call it directly
# this is from where I can start
class Problem2D:
    """
    Solves Laplace's equation with discontinuous conductivity,
    using the Level Set function to model interface movement.
    """
    def __init__(self, nx: int = 120, ny: int = 120, T: float = 0.1,
                 alpha: float = 2):
        self.nx, self.ny = nx, ny
        self.T, self.alpha = T, alpha
        self.dt = self.alpha * (1 / self.nx) ** 2  # Time step size
        self.num_steps = int(self.T / self.dt)
        self.t = 0  # Initial time
        self._initialize_mesh()
        self._initialize_functions()

    def _initialize_mesh(self):
        """Creates mesh, function spaces, and output files."""
        self.domain = mesh.create_rectangle(
            MPI.COMM_WORLD, [np.array([0, 0]), np.array([1, 1])],
            [self.nx, self.ny], mesh.CellType.triangle
        )
        # Potential & Level Set
        self.W = fem.functionspace(self.domain, ("Lagrange", 1))
        # Conductivity
        self.D = fem.functionspace(self.domain, ("DG", 0))
        # Current Density
        self.J = fem.functionspace(self.domain, ("Lagrange", 1,
                                                 (self.domain.geometry.dim,)))
        # Output files
        self.xdmf_levelset = setup_output_files(self.domain, "levelset.xdmf")
        self.xdmf_sigma = setup_output_files(self.domain, "sigma.xdmf")
        self.xdmf_current = setup_output_files(self.domain, "current.xdmf")

    def _initialize_functions(self):
        """Initializes function variables."""
        self.phi_n = fem.Function(self.W)
        self.phi_n.interpolate(lambda x: np.where(x[0] < 0.35, x[1] - 0.5,
                                                  x[1] - 0.7))
        self.sigma = fem.Function(self.D)
        self.Vh = fem.Function(self.W, name="V")
        self.jh = fem.Function(self.J, name="J")


def apply_boundary_conditions(domain, facet_tag, test_func,
                              use_dirichlet: bool):
    """
    Funtion to choose the boundary condition of the problem for the Potential

    Parameters:
    domain [dolfinx.mesh]: geometry of the domain
    face_tags []: tag of the boundaries
    test_func [fuction]: test function of the problem
    use_dirichlet [bool]: to choose if using Dichlet or Neumann b.c.
    """
    fdim = domain.topology.dim - 1
    W = test_func.ufl_function_space
    ds = Measure("ds", domain=domain, subdomain_data=facet_tag)
    flag = 0

    if use_dirichlet:
        facets_3 = facet_tag.find(3)
        dofs_3 = fem.locate_dofs_topological(W, fdim, facets_3)
        facets_4 = facet_tag.find(4)
        dofs_4 = fem.locate_dofs_topological(W, fdim, facets_4)

        BCs = [
            fem.dirichletbc(PETSc.ScalarType(0.26), dofs_3, W),
            fem.dirichletbc(PETSc.ScalarType(1.9), dofs_4, W)
        ]

        L = fem.Constant(domain, PETSc.ScalarType(0.0)) * test_func * dx

        flag = 1
    else:
        BCs = []
        L = 300 * test_func * ds(4) - 300 * test_func * ds(3)

    return BCs, L, flag
