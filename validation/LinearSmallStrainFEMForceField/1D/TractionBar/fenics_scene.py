"""1D bar under an end traction, solved in FEniCS (dolfinx).

    python fenics_scene.py

Writes `fenics.json` next to this file. That file is committed: CI has no FEniCS, so the SOFA
scene compares against the stored reference rather than re-solving here.
"""

import json
import os
import sys

import numpy as np
import ufl
from dolfinx import default_scalar_type, fem, la, mesh
from mpi4py import MPI

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from case import CELLS, FENICS_FILE, LENGTH, POISSON_RATIO, TRACTION, YOUNG_MODULUS

LOADED_END = 1

PETSC_OPTIONS = {'ksp_type': 'preonly', 'pc_type': 'lu'}


def lame_parameters(dim):
    """SOFA's reduction of isotropic elasticity to `dim` dimensions (see LameParameters.h)."""
    mu = YOUNG_MODULUS / (2 * (1 + POISSON_RATIO))
    lame_lambda = YOUNG_MODULUS * POISSON_RATIO / ((1 + POISSON_RATIO) * (1 - (dim - 1) * POISSON_RATIO))
    return lame_lambda, mu


def solve_with_petsc(a, L, bc):
    from dolfinx.fem.petsc import LinearProblem

    problem = LinearProblem(a, L, bcs=[bc], petsc_options_prefix='traction_bar',
                            petsc_options=PETSC_OPTIONS)
    solution = problem.solve()
    if isinstance(solution, tuple):  # dolfinx >= 0.10: (uh, converged_reason, iterations)
        solution = solution[0]
    return solution.x.array


def solve_with_scipy(a, L, bc):
    from scipy.sparse.linalg import spsolve

    A = fem.assemble_matrix(fem.form(a), bcs=[bc])
    A.scatter_reverse()
    b = fem.assemble_vector(fem.form(L))
    fem.apply_lifting(b.array, [fem.form(a)], bcs=[[bc]])
    b.scatter_reverse(la.InsertMode.add)
    fem.set_bc(b.array, [bc])
    return spsolve(A.to_scipy().tocsr(), b.array)


def solve():
    domain = mesh.create_interval(MPI.COMM_WORLD, CELLS, [0.0, LENGTH])
    V = fem.functionspace(domain, ("Lagrange", 1))

    lame_lambda, mu = lame_parameters(dim=1)
    stiffness = fem.Constant(domain, default_scalar_type(lame_lambda + 2 * mu))
    load = fem.Constant(domain, default_scalar_type(TRACTION))

    loaded = mesh.locate_entities_boundary(domain, 0, lambda x: np.isclose(x[0], LENGTH))
    facets = mesh.meshtags(domain, 0, loaded, np.full(loaded.size, LOADED_END, dtype=np.int32))
    ds = ufl.Measure("ds", domain=domain, subdomain_data=facets)

    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    a = stiffness * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = load * v * ds(LOADED_END)

    clamped = fem.locate_dofs_geometrical(V, lambda x: np.isclose(x[0], 0.0))
    bc = fem.dirichletbc(default_scalar_type(0.0), clamped, V)

    # Not every dolfinx build ships petsc4py; these problems are small enough that assembling
    # to scipy and solving directly is an equivalent answer rather than a degraded one.
    try:
        solution = solve_with_petsc(a, L, bc)
    except ImportError:
        solution = solve_with_scipy(a, L, bc)

    x = V.tabulate_dof_coordinates()[:, 0]
    order = np.argsort(x)
    return x[order], solution[order]


def main():
    x, u = solve()
    reference = {
        'solver': 'dolfinx',
        'element': 'P1',
        'cells': CELLS,
        'length': LENGTH,
        'youngModulus': YOUNG_MODULUS,
        'poissonRatio': POISSON_RATIO,
        'traction': TRACTION,
        'x': x.tolist(),
        'u': u.tolist(),
    }
    with open(FENICS_FILE, 'w') as f:
        json.dump(reference, f, indent=2)
    print(f"wrote {FENICS_FILE}")


if __name__ == '__main__':
    main()
