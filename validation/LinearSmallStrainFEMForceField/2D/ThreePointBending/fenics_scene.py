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
from dolfinx.fem.petsc import apply_lifting, assemble_matrix, assemble_vector, set_bc

from petsc4py import PETSc

from common.fenics import BeamMesh
from common.geometry import Beam2D


def build(deck):
    geometry = Beam2D(length=deck['geometry']['length'], width=deck['geometry']['width'])
    grid = BeamMesh(geometry, deck['mesh']['resolution'], element=deck['element'],
                    swapping=deck.get('swapping', True))
    domain = grid.build()
    V = fem.functionspace(domain, ("Lagrange", 1, (2,)))

    # Plane stress: SOFA's reduction of isotropic elasticity to dim=2 (LameParameters.h).
    young_modulus = deck['material']['youngModulus']
    poisson_ratio = deck['material']['poissonRatio']
    mu = young_modulus / (2 * (1 + poisson_ratio))
    lame_lambda = young_modulus * poisson_ratio / ((1 + poisson_ratio) * (1 - poisson_ratio))

    def sigma(u):
        eps = ufl.sym(ufl.grad(u))
        return lame_lambda * ufl.tr(eps) * ufl.Identity(2) + 2 * mu * eps

    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    a = ufl.inner(sigma(u), ufl.sym(ufl.grad(v))) * ufl.dx
    L = ufl.inner(fem.Constant(domain, default_scalar_type((0.0, 0.0))), v) * ufl.dx

    length = geometry.length
    pinned = fem.locate_dofs_geometrical(
        V, lambda x: np.isclose(x[0], 0.0) & np.isclose(x[1], 0.0))
    bc_pin = fem.dirichletbc(np.zeros(2, dtype=default_scalar_type), pinned, V)

    Vy, _ = V.sub(1).collapse()
    roller, _ = fem.locate_dofs_geometrical(
        (V.sub(1), Vy), lambda x: np.isclose(x[0], length) & np.isclose(x[1], 0.0))
    bc_roller = fem.dirichletbc(default_scalar_type(0.0), roller, V.sub(1))

    # The point load is applied on the midspan. This restricts the use of an odd number of nodes.
    midspan_dofs, _ = fem.locate_dofs_geometrical(
        (V.sub(1), Vy), lambda x: np.isclose(x[0], length / 2) & np.isclose(x[1], geometry.width))
    load_dof = midspan_dofs[0]

    return grid, V, a, L, [bc_pin, bc_roller], load_dof


def solve(deck):
    grid, V, a, L, bcs, load_dof = build(deck)
    bilinear_form, linear_form = fem.form(a), fem.form(L)

    A = assemble_matrix(bilinear_form, bcs=bcs)
    A.assemble()
    b = assemble_vector(linear_form)
    b.array[load_dof] -= deck['load']['magnitude']

    # Hijack the assemble/lift/scatter/set_bc sequence of LinearProblem to apply the singular point load
    apply_lifting(b, [bilinear_form], bcs=[bcs])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    set_bc(b, bcs)

    ksp = PETSc.KSP().create(V.mesh.comm)
    ksp.setOperators(A)
    ksp.setType('preonly')
    ksp.getPC().setType('lu')

    solution = fem.Function(V)
    ksp.solve(b, solution.x.petsc_vec)

    coords = V.tabulate_dof_coordinates()[:, :2]
    values = solution.x.array.reshape(-1, 2)
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
