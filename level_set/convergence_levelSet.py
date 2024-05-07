import matplotlib.pyplot as plt
import pyvista
import ufl
from ufl import SpatialCoordinate, TestFunction, TrialFunction, dot, dx, grad, div, FacetNormal,VectorElement, inner
import numpy as np
from dolfinx import default_scalar_type

from petsc4py import PETSc
from mpi4py import MPI

from dolfinx import fem, mesh, io, plot, cpp
from dolfinx.fem.petsc import assemble_vector, assemble_matrix, create_vector, apply_lifting, set_bc, create_matrix
from dolfinx.io import XDMFFile


t = 0  # Start time
T = 0.45  # Final time
alpha = 5

# Define mesh
nx, ny = 40, 40
space_step = 1/nx
dt = alpha * space_step**2 # time step size
num_steps = int(T/dt)
domain = mesh.create_rectangle(MPI.COMM_WORLD, [np.array([-1, -1]), np.array([1, 1])],
                               [nx, ny], mesh.CellType.triangle)
W = fem.FunctionSpace(domain, ("Lagrange", 1))
n = FacetNormal(domain)

phi_n = fem.Function(W)  # Level set function

class exact_solution():
    def __init__(self, t):
        self.t = t

    def __call__(self, x):
        return x[1] - 0.5 + np.exp(self.t)

phi_ex = exact_solution(t)
phi_n.interpolate(phi_ex)

# Define solution variable, and interpolate initial solution for visualization in Paraview
phi_h = fem.Function(W)
phi_h.name = "phi_h"
phi_h.interpolate(phi_ex)

def boundary_D(x):
    return np.isclose(x[1], 0)
dofs_D = fem.locate_dofs_geometrical(W, boundary_D)