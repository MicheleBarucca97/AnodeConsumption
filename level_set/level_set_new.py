import numpy as np
from dolfinx import fem
from dolfinx.mesh import (locate_entities, create_rectangle, CellType, meshtags)
from mpi4py import MPI
from ufl import Measure, conditional, ge, dx
from matplotlib import pyplot as plt
import dolfinx.cpp as _cpp
from dolfinx.io import XDMFFile

radius = 0.25


def tag_subdomains(msh, levelset):  # Identifies and marks subdomains accoring to locator function
    dim = msh.topology.dim
    num_cells = msh.topology.index_map(dim).size_local
    cells = np.arange(num_cells, dtype=np.int32)
    markers = np.zeros_like(cells)
    markers[locate_entities(msh, dim, levelset)] = 1
    cell_tag = meshtags(msh, dim, cells, markers)
    return cell_tag


tol = 1e-8


# simple example with circle inside a square
def phi(x):
    return np.sqrt((x[0] - 0.5) ** 2 + (x[1] - 0.5) ** 2) - radius


subdomains = lambda x: phi(x) >= -tol

N_list = np.array([5, 10, 20, 40, 80, 160, 320])
err = [];
err2 = []
for N in N_list:
    comm = MPI.COMM_WORLD
    msh = create_rectangle(comm=MPI.COMM_WORLD,
                           points=((0.0, 0.0), (1.0, 1.0)), n=(N, N),
                           cell_type=CellType.triangle, diagonal=_cpp.mesh.DiagonalType.crossed)

    # Strategy 1: analytical expression
    cell_tag = tag_subdomains(msh, subdomains)
    dx = Measure("dx", domain=msh, subdomain_data=cell_tag)
    one = fem.Constant(msh, 1.0)
    area = 1 - np.pi * radius ** 2
    err.append(np.abs(area - fem.assemble_scalar(fem.form(one * dx(1)))))

    # Strategy 2: if no analytical expression is known
    dx = Measure("dx", domain=msh)
    V = fem.FunctionSpace(msh, ("CG", 1))
    phi_ = fem.Function(V)
    phi_.interpolate(phi)
    err2.append(np.abs(area - fem.assemble_scalar(fem.form(conditional(ge(phi_, tol), 1, 0) * dx))))
    with XDMFFile(msh.comm, f"tag_{N}.xdmf", "w") as xdmf:
        xdmf.write_mesh(msh)
        xdmf.write_meshtags(cell_tag, msh.geometry)

