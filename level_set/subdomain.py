from dolfinx import default_scalar_type
from dolfinx.fem import (Constant, dirichletbc, Function, FunctionSpace, assemble_scalar,
                         form, locate_dofs_geometrical, locate_dofs_topological)
from dolfinx.fem.petsc import LinearProblem
from dolfinx.io import XDMFFile, gmshio
from dolfinx.mesh import create_unit_square, locate_entities
from dolfinx.plot import vtk_mesh

from ufl import (SpatialCoordinate, TestFunction, TrialFunction,
                 dx, grad, inner)

from mpi4py import MPI

import meshio
import gmsh
import numpy as np
import pyvista

pyvista.start_xvfb()

mesh = create_unit_square(MPI.COMM_WORLD, 10, 10)
Q = FunctionSpace(mesh, ("DG", 0))

def Omega_0(x):
    return x[1] <= 0.5
def Omega_1(x):
    return x[1] >= 0.5

# We will solve a variable-coefficient extension of the Poisson equation
kappa = Function(Q)
cells_0 = locate_entities(mesh, mesh.topology.dim, Omega_0)
cells_1 = locate_entities(mesh, mesh.topology.dim, Omega_1)
# In the previous code block, we found which cells (triangular elements) which satisfies the condition for being in
# Omega0 or Omega1. As the DG - 0 function contain only one degree of freedom per mesh, there is a one to one mapping
# between the cell indicies and the degrees of freedom.
kappa.x.array[cells_0] = np.full_like(cells_0, 1, dtype=default_scalar_type)
kappa.x.array[cells_1] = np.full_like(cells_1, 0.1, dtype=default_scalar_type)

V = FunctionSpace(mesh, ("Lagrange", 1))
u, v = TrialFunction(V), TestFunction(V)
a = inner(kappa * grad(u), grad(v)) * dx
x = SpatialCoordinate(mesh)
L = Constant(mesh, default_scalar_type(1)) * v * dx
dofs = locate_dofs_geometrical(V, lambda x: np.isclose(x[0], 0))
bcs = [dirichletbc(default_scalar_type(1), dofs, V)]

problem = LinearProblem(a, L, bcs=bcs, petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
uh = problem.solve()

# Filter out ghosted cells
num_cells_local = mesh.topology.index_map(mesh.topology.dim).size_local
marker = np.zeros(num_cells_local, dtype=np.int32)
cells_0 = cells_0[cells_0 < num_cells_local]
cells_1 = cells_1[cells_1 < num_cells_local]
marker[cells_0] = 1
marker[cells_1] = 2


'''from dolfinx.io import XDMFFile

with XDMFFile(MPI.COMM_WORLD, "marker.xdmf", "w") as xdmf:
    xdmf.write_mesh(mesh)
    kappa.name = 'markers [-]'
    xdmf.write_function(kappa)

xdmf.close()

with XDMFFile(MPI.COMM_WORLD, "velocity.xdmf", "w") as xdmf:
    xdmf.write_mesh(mesh)
    uh.name = 'U [m/s]'
    xdmf.write_function(uh)
xdmf.close()'''

gmsh.initialize()
proc = MPI.COMM_WORLD.rank
top_marker = 2
bottom_marker = 1
left_marker = 1
if proc == 0:
    # We create one rectangle for each subdomain
    bath = gmsh.model.occ.addRectangle(0, 0, 0, 1, 0.75, tag=1)
    anode = gmsh.model.occ.addRectangle(0, 0.5, 0, 1, 0.5, tag=2)
    # Add a rectangle that will be removed
    gmsh.model.occ.addRectangle(0.0, 0.5, 0, 0.35, 0.25, tag=3)
    gmsh.model.occ.addRectangle(0.65, 0.5, 0, 0.35, 0.25, tag=4)
    gmsh.model.occ.addRectangle(0.35, 0.5, 0, 0.65, 0.25, tag=5)

    # Remove the small rectangle to the anode domain
    bath2 = gmsh.model.occ.cut([(2, bath)], [(2, 3)])
    gmsh.model.occ.synchronize()
    bath3 = gmsh.model.occ.cut([(2, bath2)], [(2, 4)])
    anode2 = gmsh.model.occ.cut([(2, 2)], [(2, 5)])
    gmsh.model.occ.synchronize()

    # We fuse the two rectangles and keep the interface between them
    gmsh.model.occ.fragment([(2, bath3)], [(2, anode2)])
    gmsh.model.occ.synchronize()

    # Mark the top (2) and bottom (1) rectangle
    top, bottom = None, None
    for surface in gmsh.model.getEntities(dim=2):
        com = gmsh.model.occ.getCenterOfMass(surface[0], surface[1])
        if np.allclose(com, [0.5, 0.25, 0]):
            bottom = surface[1]
        else:
            top = surface[1]
    gmsh.model.addPhysicalGroup(2, [bottom], bottom_marker)
    gmsh.model.addPhysicalGroup(2, [top], top_marker)
    # Tag the left boundary
    left = []
    for line in gmsh.model.getEntities(dim=1):
        com = gmsh.model.occ.getCenterOfMass(line[0], line[1])
        if np.isclose(com[0], 0):
            left.append(line[1])
    gmsh.model.addPhysicalGroup(1, left, left_marker)
    gmsh.model.mesh.generate(2)
    gmsh.write("mesh.msh")
gmsh.finalize()

mesh, cell_markers, facet_markers = gmshio.read_from_msh("mesh.msh", MPI.COMM_WORLD, gdim=2)


