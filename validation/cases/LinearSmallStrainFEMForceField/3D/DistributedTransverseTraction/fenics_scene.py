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
from dolfinx.fem.petsc import LinearProblem

from common.fenics import BeamMesh
from common.geometry import Beam3D

PETSC_OPTIONS = {'ksp_type': 'preonly', 'pc_type': 'lu'}


def build(case):
    geometry = Beam3D(length=case['geometry']['length'], width=case['geometry']['width'],
                      height=case['geometry']['height'])
    grid = BeamMesh(geometry, case['mesh']['resolution'], element=case['element'],
                    swapping=case.get('swapping', True))
    domain = grid.build()
    V = fem.functionspace(domain, ("Lagrange", 1, (3,)))

    young_modulus = case['material']['youngModulus']
    poisson_ratio = case['material']['poissonRatio']
    mu = young_modulus / (2 * (1 + poisson_ratio))
    lame_lambda = young_modulus * poisson_ratio / ((1 + poisson_ratio) * (1 - 2 * poisson_ratio))

    def sigma(u):
        eps = ufl.sym(ufl.grad(u))
        return lame_lambda * ufl.tr(eps) * ufl.Identity(3) + 2 * mu * eps

    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)

    top_facets = dmesh.locate_entities_boundary(
        domain, domain.topology.dim - 1, lambda x: np.isclose(x[1], geometry.width))
    facet_tags = dmesh.meshtags(domain, domain.topology.dim - 1, top_facets,
                                np.full_like(top_facets, 1, dtype=np.int32))
    ds = ufl.Measure('ds', domain=domain, subdomain_data=facet_tags)

    traction = fem.Constant(domain, default_scalar_type((0.0, -case['load']['magnitude'], 0.0)))

    a = ufl.inner(sigma(u), ufl.sym(ufl.grad(v))) * ufl.dx
    L = ufl.dot(traction, v) * ds(1)

    clamped = fem.locate_dofs_geometrical(V, lambda x: np.isclose(x[0], 0.0))
    bc = fem.dirichletbc(np.zeros(3, dtype=default_scalar_type), clamped, V)

    return grid, V, a, L, bc


def solve(case):
    grid, V, a, L, bc = build(case)

    problem = LinearProblem(a, L, bcs=[bc], petsc_options=PETSC_OPTIONS)
    solution = problem.solve()

    coords = V.tabulate_dof_coordinates()[:, :3]
    values = solution.x.array.reshape(-1, 3)
    return coords[grid.inverse], values[grid.inverse]


def main(case):
    x, u = solve(case)

    reference_file = os.path.join(CASE_DIR, 'reference_solution.json')
    if os.path.exists(reference_file):
        with open(reference_file) as f:
            reference = json.load(f)
    else:
        reference = {}
    reference.setdefault(case['element'], {})['fenics'] = {'x': x.tolist(), 'u': u.tolist()}
    with open(reference_file, 'w') as f:
        json.dump(reference, f)

    print(f"{case['element']}: solved {len(x)} nodes")


if __name__ == '__main__':
    with open(os.path.join(CASE_DIR, 'case.json')) as f:
        whole_case = json.load(f)

    for sub_case in ([whole_case] if 'geometry' in whole_case else whole_case.values()):
        main(sub_case)
