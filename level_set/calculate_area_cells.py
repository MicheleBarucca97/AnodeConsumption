import ufl
from ufl import SpatialCoordinate, TestFunction, TrialFunction, dot, dx, grad, div, FacetNormal, VectorElement, inner
import numpy as np
from dolfinx import default_scalar_type

from petsc4py import PETSc
from mpi4py import MPI

from dolfinx import fem, mesh, io, plot, cpp
from dolfinx.fem.petsc import assemble_vector, assemble_matrix, create_vector, apply_lifting, set_bc
from dolfinx.io import XDMFFile

nx, ny = 50, 50
domain = mesh.create_rectangle(MPI.COMM_WORLD, [np.array([0, 0]), np.array([1, 1])],
                               [nx, ny], mesh.CellType.triangle)

D = fem.FunctionSpace(domain, ("DG", 0))
v = ufl.TestFunction(D)
cell_area_form = fem.form(v*dx)
cell_area = fem.assemble_vector(cell_area_form)