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
from common.geometry import Beam2D

PETSC_OPTIONS = {'ksp_type': 'preonly', 'pc_type': 'lu'}


def build(case):
    geometry = Beam2D(length=case['geometry']['length'], width=case['geometry']['width'])
    grid = BeamMesh(geometry, case['mesh']['resolution'], element=case['element'],
                    swapping=case.get('swapping', True))
    domain = grid.build()
    V = fem.functionspace(domain, ("Lagrange", 1, (2,)))

    # Plane stress: SOFA's reduction of isotropic elasticity to dim=2 (LameParameters.h).
    young_modulus = case['material']['youngModulus']
    poisson_ratio = case['material']['poissonRatio']
    mu = young_modulus / (2 * (1 + poisson_ratio))
    lame_lambda = young_modulus * poisson_ratio / ((1 + poisson_ratio) * (1 - poisson_ratio))

    def sigma(u):
        eps = ufl.sym(ufl.grad(u))
        return lame_lambda * ufl.tr(eps) * ufl.Identity(2) + 2 * mu * eps

    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)

    right_facets = dmesh.locate_entities_boundary(
        domain, domain.topology.dim - 1, lambda x: np.isclose(x[0], geometry.length))
    facet_tags = dmesh.meshtags(domain, domain.topology.dim - 1, right_facets,
                                np.full_like(right_facets, 1, dtype=np.int32))
    ds = ufl.Measure('ds', domain=domain, subdomain_data=facet_tags)

    traction = fem.Constant(domain, default_scalar_type((-case['load']['magnitude'], 0.0)))

    a = ufl.inner(sigma(u), ufl.sym(ufl.grad(v))) * ufl.dx
    L = ufl.dot(traction, v) * ds(1)

    # Fix the axial displacement along x = 0, leaving it free to contract laterally. 
    # Pin the lateral displacement at the corner to remove the remaining rigid-body mode.
    Vx, _ = V.sub(0).collapse()
    clamped_x = fem.locate_dofs_geometrical((V.sub(0), Vx), lambda x: np.isclose(x[0], 0.0))
    bc_x = fem.dirichletbc(default_scalar_type(0.0), clamped_x[0], V.sub(0))

    Vy, _ = V.sub(1).collapse()
    corner = fem.locate_dofs_geometrical((V.sub(1), Vy),
                                         lambda x: np.isclose(x[0], 0.0) & np.isclose(x[1], 0.0))
    bc_corner = fem.dirichletbc(default_scalar_type(0.0), corner[0], V.sub(1))

    return grid, V, a, L, [bc_x, bc_corner]


def solve(case):
    grid, V, a, L, bcs = build(case)

    problem = LinearProblem(a, L, bcs=bcs, petsc_options=PETSC_OPTIONS)
    solution = problem.solve()

    coords = V.tabulate_dof_coordinates()[:, :2]
    values = solution.x.array.reshape(-1, 2)
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
