import json
import os
import sys

CASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(CASE_DIR, *[os.pardir] * 4))
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


def build(deck):
    geometry = Beam3D(length=deck['geometry']['length'], width=deck['geometry']['width'],
                      height=deck['geometry']['height'])
    grid = BeamMesh(geometry, deck['mesh']['resolution'], element=deck['element'],
                    swapping=deck.get('swapping', True))
    domain = grid.build()
    V = fem.functionspace(domain, ("Lagrange", 1, (3,)))

    young_modulus = deck['material']['youngModulus']
    poisson_ratio = deck['material']['poissonRatio']
    mu = young_modulus / (2 * (1 + poisson_ratio))
    lame_lambda = young_modulus * poisson_ratio / ((1 + poisson_ratio) * (1 - 2 * poisson_ratio))

    def sigma(u):
        eps = ufl.sym(ufl.grad(u))
        return lame_lambda * ufl.tr(eps) * ufl.Identity(3) + 2 * mu * eps

    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)

    right_facets = dmesh.locate_entities_boundary(
        domain, domain.topology.dim - 1, lambda x: np.isclose(x[0], geometry.length))
    facet_tags = dmesh.meshtags(domain, domain.topology.dim - 1, right_facets,
                                np.full_like(right_facets, 1, dtype=np.int32))
    ds = ufl.Measure('ds', domain=domain, subdomain_data=facet_tags)

    traction = fem.Constant(domain, default_scalar_type((-deck['load']['magnitude'], 0.0, 0.0)))

    a = ufl.inner(sigma(u), ufl.sym(ufl.grad(v))) * ufl.dx
    L = ufl.dot(traction, v) * ds(1)

    # Fix the axial displacement along x = 0, leaving it free to contract laterally.
    # Pin one corner in y and z to remove translation; pin a second corner off that axis in z
    # only, to remove the remaining rigid-body mode: rotation about x.
    Vx, _ = V.sub(0).collapse()
    clamped_x = fem.locate_dofs_geometrical((V.sub(0), Vx), lambda x: np.isclose(x[0], 0.0))
    bc_x = fem.dirichletbc(default_scalar_type(0.0), clamped_x[0], V.sub(0))

    Vy, _ = V.sub(1).collapse()
    corner_y = fem.locate_dofs_geometrical(
        (V.sub(1), Vy), lambda x: np.isclose(x[0], 0.0) & np.isclose(x[1], 0.0) & np.isclose(x[2], 0.0))
    bc_corner_y = fem.dirichletbc(default_scalar_type(0.0), corner_y[0], V.sub(1))

    Vz, _ = V.sub(2).collapse()
    corner_z = fem.locate_dofs_geometrical(
        (V.sub(2), Vz), lambda x: np.isclose(x[0], 0.0) & np.isclose(x[1], 0.0) & np.isclose(x[2], 0.0))
    bc_corner_z = fem.dirichletbc(default_scalar_type(0.0), corner_z[0], V.sub(2))

    second_corner_z = fem.locate_dofs_geometrical(
        (V.sub(2), Vz), lambda x: np.isclose(x[0], 0.0) & np.isclose(x[1], geometry.width) & np.isclose(x[2], 0.0))
    bc_second_corner_z = fem.dirichletbc(default_scalar_type(0.0), second_corner_z[0], V.sub(2))

    return grid, V, a, L, [bc_x, bc_corner_y, bc_corner_z, bc_second_corner_z]


def solve(deck):
    grid, V, a, L, bcs = build(deck)

    problem = LinearProblem(a, L, bcs=bcs, petsc_options=PETSC_OPTIONS)
    solution = problem.solve()

    coords = V.tabulate_dof_coordinates()[:, :3]
    values = solution.x.array.reshape(-1, 3)
    return coords[grid.inverse], values[grid.inverse]


def main(deck):
    x, u = solve(deck)

    reference_file = os.path.join(CASE_DIR, 'reference_solution.json')
    if os.path.exists(reference_file):
        with open(reference_file) as f:
            reference = json.load(f)
    else:
        reference = {}
    reference.setdefault(deck['element'], {})['fenics'] = {'x': x.tolist(), 'u': u.tolist()}
    with open(reference_file, 'w') as f:
        json.dump(reference, f)

    print(f"{deck['element']}: solved {len(x)} nodes")


if __name__ == '__main__':
    with open(os.path.join(CASE_DIR, 'deck.json')) as f:
        whole_deck = json.load(f)

    for sub_deck in ([whole_deck] if 'geometry' in whole_deck else whole_deck.values()):
        main(sub_deck)
