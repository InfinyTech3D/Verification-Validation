import json
import os
import sys

CASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(CASE_DIR, *[os.pardir] * 5))
for path in (CASE_DIR, REPO_ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)

import numpy as np
import ufl
from dolfinx import default_scalar_type, fem, mesh as dmesh
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.nls.petsc import NewtonSolver
from mpi4py import MPI

from common.fenics import BeamMesh, strain_energy_density
from common.geometry import Beam3D


def build(case):
    geometry = Beam3D(length=case['geometry']['length'], width=case['geometry']['width'],
                      height=case['geometry']['height'])
    grid = BeamMesh(geometry, case['mesh']['resolution'], element=case['element'],
                    swapping=case.get('swapping', True))
    domain = grid.build()
    V = fem.functionspace(domain, ("Lagrange", 1, (3,)))

    u, v = fem.Function(V), ufl.TestFunction(V)
    load_factor = fem.Constant(domain, default_scalar_type(0))

    top_facets = dmesh.locate_entities_boundary(
        domain, domain.topology.dim - 1, lambda x: np.isclose(x[1], geometry.width))
    facet_tags = dmesh.meshtags(domain, domain.topology.dim - 1, top_facets,
                                np.full_like(top_facets, 1, dtype=np.int32))
    ds = ufl.Measure('ds', domain=domain, subdomain_data=facet_tags)

    traction = fem.Constant(domain, default_scalar_type((0.0, -case['load']['magnitude'], 0.0)))

    G = ufl.variable(ufl.grad(u))
    P = ufl.diff(strain_energy_density(case['material'], G), G)
    # SOFA's hexahedron rule: 2x2x2 Gauss
    dx = ufl.dx(metadata={'quadrature_degree': 3}) if case['element'] == 'hexa' else ufl.dx
    residual = ufl.inner(P, ufl.grad(v)) * dx - load_factor * ufl.dot(traction, v) * ds(1)

    clamped = fem.locate_dofs_geometrical(V, lambda x: np.isclose(x[0], 0.0))
    bc = fem.dirichletbc(np.zeros(3, dtype=default_scalar_type), clamped, V)

    return grid, V, u, residual, [bc], load_factor


def solve(case):
    grid, V, u, residual, bcs, load_factor = build(case)

    solver = NewtonSolver(MPI.COMM_WORLD, NonlinearProblem(residual, u, bcs=bcs))
    solver.rtol, solver.atol, solver.max_it = 1e-10, 0, 20
    solver.krylov_solver.setType('preonly')
    solver.krylov_solver.getPC().setType('lu')

    steps = case['load']['steps']
    for step in range(1, steps + 1):
        load_factor.value = step / steps
        solver.solve(u)  # raises if Newton does not converge

    coords = V.tabulate_dof_coordinates()[:, :3]
    values = u.x.array.reshape(-1, 3)
    return coords[grid.inverse], values[grid.inverse]


def main(case):
    x, u = solve(case)

    reference_file = os.path.join(CASE_DIR, 'reference_solution.json')
    if os.path.exists(reference_file):
        with open(reference_file) as f:
            reference = json.load(f)
    else:
        reference = {}
    key = f"{case['material']['type']}_{case['element']}"
    reference.setdefault(key, {})['fenics'] = {'x': x.tolist(), 'u': u.tolist()}
    with open(reference_file, 'w') as f:
        json.dump(reference, f)

    print(f"{key}: solved {len(x)} nodes")


if __name__ == '__main__':
    with open(os.path.join(CASE_DIR, 'case.json')) as f:
        for sub_case in json.load(f).values():
            main(sub_case)
