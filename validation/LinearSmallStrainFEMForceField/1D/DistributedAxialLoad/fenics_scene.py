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
from dolfinx import default_scalar_type, fem
from dolfinx.fem.petsc import LinearProblem

from common.fenics import BarMesh
from common.geometry import Bar1D

PETSC_OPTIONS = {'ksp_type': 'preonly', 'pc_type': 'lu'}


def build(deck):
    geometry = Bar1D(length=deck['geometry']['length'])
    grid = BarMesh(geometry, deck['mesh']['resolution'])
    domain = grid.build()
    V = fem.functionspace(domain, ("Lagrange", 1))

    # 1D axial stiffness: lambda + 2*mu collapses to E exactly (poissonRatio is inert in 1D).
    stiffness = fem.Constant(domain, default_scalar_type(deck['material']['youngModulus']))
    load = fem.Constant(domain, default_scalar_type(deck['load']['magnitude']))

    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    a = stiffness * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = load * v * ufl.dx

    clamped = fem.locate_dofs_geometrical(V, lambda x: np.isclose(x[0], 0.0))
    bc = fem.dirichletbc(default_scalar_type(0.0), clamped, V)

    return grid, V, a, L, bc


def solve(deck):
    grid, V, a, L, bc = build(deck)

    problem = LinearProblem(a, L, bcs=[bc], petsc_options=PETSC_OPTIONS)
    solution = problem.solve()

    x = V.tabulate_dof_coordinates()[:, 0]
    return x[grid.inverse], solution.x.array[grid.inverse]


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
