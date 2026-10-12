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
from dolfinx.fem.petsc import apply_lifting, assemble_matrix, assemble_vector, set_bc

from petsc4py import PETSc

from common.fenics import BeamMesh
from common.geometry import Beam3D


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
    a = ufl.inner(sigma(u), ufl.sym(ufl.grad(v))) * ufl.dx
    L = ufl.inner(fem.Constant(domain, default_scalar_type((0.0, 0.0, 0.0))), v) * ufl.dx

    length = geometry.length

    # Line support at x=0, y=0 (all z): pin ux, uy. Line roller at x=length, y=0: pin uy only.
    Vx, _ = V.sub(0).collapse()
    pin_x, _ = fem.locate_dofs_geometrical(
        (V.sub(0), Vx), lambda x: np.isclose(x[0], 0.0) & np.isclose(x[1], 0.0))
    bc_pin_x = fem.dirichletbc(default_scalar_type(0.0), pin_x, V.sub(0))

    Vy, _ = V.sub(1).collapse()
    pin_y, _ = fem.locate_dofs_geometrical(
        (V.sub(1), Vy), lambda x: np.isclose(x[0], 0.0) & np.isclose(x[1], 0.0))
    bc_pin_y = fem.dirichletbc(default_scalar_type(0.0), pin_y, V.sub(1))

    roller, _ = fem.locate_dofs_geometrical(
        (V.sub(1), Vy), lambda x: np.isclose(x[0], length) & np.isclose(x[1], 0.0))
    bc_roller = fem.dirichletbc(default_scalar_type(0.0), roller, V.sub(1))

    # Both support lines run the whole depth (z): nothing pins uz, leaving one rigid-body mode
    # (translation along z) free. Pin it at a single node, off the load line.
    Vz, _ = V.sub(2).collapse()
    z_pin, _ = fem.locate_dofs_geometrical(
        (V.sub(2), Vz), lambda x: np.isclose(x[0], 0.0) & np.isclose(x[1], 0.0) & np.isclose(x[2], 0.0))
    bc_z_pin = fem.dirichletbc(default_scalar_type(0.0), z_pin, V.sub(2))

    # The point load runs along the midspan, top, the whole depth (z): one concentrated force per
    # node there. This restricts the use of an odd number of nodes along x.
    load_dofs, _ = fem.locate_dofs_geometrical(
        (V.sub(1), Vy), lambda x: np.isclose(x[0], length / 2) & np.isclose(x[1], geometry.width))

    return grid, V, a, L, [bc_pin_x, bc_pin_y, bc_roller, bc_z_pin], load_dofs


def solve(case):
    grid, V, a, L, bcs, load_dofs = build(case)
    bilinear_form, linear_form = fem.form(a), fem.form(L)

    A = assemble_matrix(bilinear_form, bcs=bcs)
    A.assemble()
    b = assemble_vector(linear_form)
    b.array[load_dofs] -= case['load']['magnitude']

    apply_lifting(b, [bilinear_form], bcs=[bcs])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    set_bc(b, bcs)

    ksp = PETSc.KSP().create(V.mesh.comm)
    ksp.setOperators(A)
    ksp.setType('preonly')
    ksp.getPC().setType('lu')

    solution = fem.Function(V)
    ksp.solve(b, solution.x.petsc_vec)

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
