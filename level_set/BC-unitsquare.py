


from mpi4py import MPI
import numpy as np
import ufl
from basix.ufl import element
from dolfinx.fem import (Constant, Function, dirichletbc,
                         form, functionspace, locate_dofs_topological,
                         locate_dofs_geometrical,assemble_scalar)
from dolfinx.fem.petsc import assemble_matrix_block,assemble_vector_block#, create_vector
from dolfinx.mesh import *
from ufl import div, dx, ds, grad, inner, FacetNormal, dot
from petsc4py import PETSc
from utilities import error_L2, convergence_plot, plot_solution
from datetime import datetime


# mesh, spaces, functions
msh = create_unit_square(MPI.COMM_WORLD, 20,20,)
P1 = element("Lagrange", "triangle", 1) 
XV = functionspace(msh, P1)
Xp = functionspace(msh, P1)
x = ufl.SpatialCoordinate(msh)
n = FacetNormal(msh)
V = ufl.TrialFunction(XV)
p = ufl.TrialFunction(Xp)
W = ufl.TestFunction(XV)
q = ufl.TestFunction(Xp)
V_old = Function(XV)
V_new = Function(XV)
p_old = Function(Xp)
p_new = Function(Xp)

# exact boundary value function
class V_exact:
    def __init__(self):
        self.t = 0.0
        
    def eval(self, x):
        pi = np.pi
        cosx = np.cos(pi*x[0])
        siny = np.sin(pi*x[1])
        cost = np.cos(self.t)
        return pi*siny - pi*cosx*siny*cost

# Boundary regions
Gamma = locate_entities_boundary(msh, dim=1,marker=lambda x: np.logical_or.reduce((
                                                            np.isclose(x[0], 0.0),  # x=0
                                                            np.isclose(x[0], 1.0),  # x=1
                                                            np.isclose(x[1], 0.0),  # y=0
                                                            np.isclose(x[1], 1.0))))# y=1
Gamma_x0 = locate_entities_boundary(msh, dim=1, marker=lambda x: np.isclose(x[0], 0.0)) # x=0
Gamma_x1 = locate_entities_boundary(msh, dim=1, marker=lambda x: np.isclose(x[0], 1.0)) # x=1
Gamma_y0 = locate_entities_boundary(msh, dim=1, marker=lambda x: np.isclose(x[1], 0.0)) # y=0
Gamma_y1 = locate_entities_boundary(msh, dim=1, marker=lambda x: np.isclose(x[1], 1.0)) # y=1

# example dof arrays
dofs_p = locate_dofs_topological(Xp, 1, Gamma)
dofs_Vx0 = locate_dofs_topological(XV, 1, Gamma_x0)
dofs_Vx1 = locate_dofs_topological(XV, 1, Gamma_x1)
dofs_Vy = locate_dofs_topological(XV, 1, Gamma)

BCs = [dirichletbc(0.0, dofs_p, Xp),
       dirichletbc(123.0, dofs_Vx0, XV),
       dirichletbc(0.0, dofs_Vy, XV)] 