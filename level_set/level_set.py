from dolfin import *
import numpy as np
from matplotlib import pyplot as plt
from tabulate import tabulate

r = 0.2
def phi(x):
    return (x[0] - 0.5)**2 + (x[1]-0.5)**2 - r**2 #Zero level set is a circle centered in (0.5,0.5) with radius 0.2
class Omega(SubDomain):
    def inside(self, x, on_boundary):
        return phi(x) > 0


class PhiExpr(UserExpression):
    def eval(self, value, x):
        value[0] = phi(x)
    def value_shape(self):
        return ()

#Total domain area: 1
#Interior expected area: pi*r^2
#Exterior expected area: 1 - pi*r^2

a = 10; q = 2; length = 6
size = [a *  q** (n - 1) for n in range(1, length + 1)]
a_int = []; a_ext = []; e_int = []; e_ext=[]; k = 0
algebraic_int, algebraic_ext, e_algebraic_int, e_algebraic_ext = [], [], [], []

for N in size:
    print('Mesh size:', N)
    mesh = UnitSquareMesh(N, N,'crossed')

    V = FunctionSpace(mesh, "CG", 1)
    phi_function = Function(V)
    phi_function.interpolate(PhiExpr())

    algebraic_int.append(assemble(conditional(lt(phi_function, 0.0), 1, 0)*dx))
    algebraic_ext.append(assemble(conditional(gt(phi_function, 0.0), 1, 0)*dx))
    e_algebraic_int.append(pi*r**2 - algebraic_int[k])
    e_algebraic_ext.append(1 - pi*r**2 - algebraic_ext[k])

    domains = MeshFunction("size_t", mesh, mesh.topology().dim(), 0)
    omega = Omega()
    omega.mark(domains, 1)
    dx_int = Measure("dx", domain=mesh, subdomain_data=domains)
    a_int.append(assemble(Constant(1)*dx_int(0)))
    a_ext.append(assemble(Constant(1)*dx_int(1)))
    e_int.append(pi*r**2 - a_int[k])
    e_ext.append(1 - pi*r**2 - a_ext[k])
    print('Interior area:', a_int[k])
    print('Error on interior area:', e_int[k])
    print('Exterior area:', a_ext[k])
    print('Error on extrior area:', e_ext[k])

    print('Interior algebraic area:', algebraic_int[k])
    print('Error on algebraic interior area:', e_algebraic_int[k])
    print('Exterior algebraic area:', algebraic_ext[k])
    print('Error on algebraic exterior area:', e_algebraic_ext[k])

    k+=1

e_size = np.ones(length)/size
headers=['mesh size','element size', 'interior area', 'exterior area', 'interior error', 'exterior error']
table = [size, e_size,a_int,a_ext, e_int, e_ext]
print(tabulate(np.transpose(table),headers=headers, floatfmt=".3e"))

headers2=['mesh size','element size', 'algebraic interior area', 'algebraic exterior area', 'algebraic interior error', 'algebraic exterior error']
table2 = [size, e_size,algebraic_int,algebraic_ext, e_algebraic_int, e_algebraic_ext]
print(tabulate(np.transpose(table2),headers=headers2, floatfmt=".3e"))