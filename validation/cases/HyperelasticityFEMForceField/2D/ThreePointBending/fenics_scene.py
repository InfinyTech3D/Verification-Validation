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

from common.fenics import BeamMesh, strain_energy_density
from common.geometry import Beam2D


def build(case):
    geometry = Beam2D(length=case['geometry']['length'], width=case['geometry']['width'])
    grid = BeamMesh(geometry, case['mesh']['resolution'], element=case['element'],
                    swapping=case.get('swapping', True))
    domain = grid.build()
    V = fem.functionspace(domain, ("Lagrange", 1, (2,)))

    u, v = fem.Function(V), ufl.TestFunction(V)

    G = ufl.variable(ufl.grad(u))
    P = ufl.diff(strain_energy_density(case['material'], G), G)
    # SOFA's quad rule: 3 points, degree 2, mapped from [-1, 1]² to [0, 1]²
    points = (np.array([[np.sqrt(2 / 3), 0], [-1 / np.sqrt(6), -1 / np.sqrt(2)], [-1 / np.sqrt(6), 1 / np.sqrt(2)]]) + 1) / 2
    dx = ufl.dx(metadata={'quadrature_rule': 'custom', 'quadrature_points': points,
                          'quadrature_weights': np.full(3, 1 / 3)}) if case['element'] == 'quad' else ufl.dx
    residual = ufl.inner(P, ufl.grad(v)) * dx

    length = geometry.length
    Vx, _ = V.sub(0).collapse()
    Vy, _ = V.sub(1).collapse()
    rollers, _ = fem.locate_dofs_geometrical(
        (V.sub(1), Vy), lambda x: (np.isclose(x[0], 0.0) | np.isclose(x[0], length)) & np.isclose(x[1], 0.0))
    bc_rollers = fem.dirichletbc(default_scalar_type(0.0), rollers, V.sub(1))

    # Removes the rigid translation along x while keeping the solution symmetric about the midspan
    anchor, _ = fem.locate_dofs_geometrical(
        (V.sub(0), Vx), lambda x: np.isclose(x[0], length / 2) & np.isclose(x[1], 0.0))
    bc_anchor = fem.dirichletbc(default_scalar_type(0.0), anchor, V.sub(0))

    # The point load is applied on the midspan. This restricts the use of an odd number of nodes.
    midspan_dofs, _ = fem.locate_dofs_geometrical(
        (V.sub(1), Vy), lambda x: np.isclose(x[0], length / 2) & np.isclose(x[1], geometry.width))
    load_dof = midspan_dofs[0]

    return grid, V, u, residual, [bc_rollers, bc_anchor], load_dof


def solve(case):
    grid, V, u, residual, bcs, load_dof = build(case)
    residual_form, jacobian_form = fem.form(residual), fem.form(ufl.derivative(residual, u))

    ksp = PETSc.KSP().create(V.mesh.comm)
    ksp.setType('preonly')
    ksp.getPC().setType('lu')
    du = fem.Function(V)

    steps = case['load']['steps']
    for step in range(1, steps + 1):
        for iteration in range(20):
            b = assemble_vector(residual_form)
            # The residual is internal minus external forces, and the load points along -y
            b.array[load_dof] += case['load']['magnitude'] * step / steps

            # Hijack the assemble/lift/scatter/set_bc sequence of NonlinearProblem to apply the singular point load
            apply_lifting(b, [jacobian_form], bcs=[bcs])
            b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
            set_bc(b, bcs)

            residual_norm = b.norm()
            if iteration == 0:
                initial_residual_norm = residual_norm
            elif residual_norm < 1e-10 * initial_residual_norm:
                break

            A = assemble_matrix(jacobian_form, bcs=bcs)
            A.assemble()
            ksp.setOperators(A)
            ksp.solve(b, du.x.petsc_vec)
            du.x.scatter_forward()
            u.x.array[:] -= du.x.array
        else:
            raise RuntimeError(f"Newton did not converge at load step {step}")

    coords = V.tabulate_dof_coordinates()[:, :2]
    values = u.x.array.reshape(-1, 2)
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
