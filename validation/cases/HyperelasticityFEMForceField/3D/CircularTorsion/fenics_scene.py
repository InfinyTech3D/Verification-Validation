import json
import os
import sys

CASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(CASE_DIR, *[os.pardir] * 5))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np
import ufl
from dolfinx import default_scalar_type, fem, mesh as dmesh
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.nls.petsc import NewtonSolver
from dolfinx.io import gmshio
from mpi4py import MPI

from common.fenics import strain_energy_density


def build(case):
    domain = gmshio.read_from_msh(os.path.join(CASE_DIR, case['mesh']['file']), MPI.COMM_WORLD, gdim=3)[0]
    V = fem.functionspace(domain, ("Lagrange", 1, (3,)))

    u, v = fem.Function(V), ufl.TestFunction(V)

    right_facets = dmesh.locate_entities_boundary(
        domain, domain.topology.dim - 1, lambda x: np.isclose(x[0], case['geometry']['length']))
    facet_tags = dmesh.meshtags(domain, domain.topology.dim - 1, right_facets,
                                np.full_like(right_facets, 1, dtype=np.int32))
    # The traction is linear over a face: degree 2 integrates it against the P1 test functions exactly, as in SOFA
    ds = ufl.Measure('ds', domain=domain, subdomain_data=facet_tags, metadata={'quadrature_degree': 2})

    # The torque T as the traction (T / J) (0, -z, y), J = pi R^4 / 2, interpolated from its nodal values as in SOFA
    polar_moment = np.pi * case['geometry']['radius'] ** 4 / 2
    scale = case['load']['magnitude'] / polar_moment
    traction = fem.Function(V)
    traction.interpolate(lambda x: np.vstack([np.zeros_like(x[0]), -scale * x[2], scale * x[1]]))
    load_factor = fem.Constant(domain, default_scalar_type(0))

    G = ufl.variable(ufl.grad(u))
    P = ufl.diff(strain_energy_density(case['material'], G), G)
    residual = ufl.inner(P, ufl.grad(v)) * ufl.dx - load_factor * ufl.dot(traction, v) * ds(1)

    clamped = fem.locate_dofs_geometrical(V, lambda x: np.isclose(x[0], 0.0))
    bc = fem.dirichletbc(np.zeros(3, dtype=default_scalar_type), clamped, V)

    return V, u, residual, [bc], load_factor


def solve(case):
    V, u, residual, bcs, load_factor = build(case)

    solver = NewtonSolver(MPI.COMM_WORLD, NonlinearProblem(residual, u, bcs=bcs))
    solver.rtol, solver.atol, solver.max_it = 1e-10, 0, 20
    solver.krylov_solver.setType('preonly')
    solver.krylov_solver.getPC().setType('lu')

    steps = case['load']['steps']
    for step in range(1, steps + 1):
        load_factor.value = step / steps
        solver.solve(u)  # raises if Newton does not converge

    # dolfinx renumbers the nodes it reads: put them back in the mesh file's order, which is SOFA's
    domain = V.mesh
    file_index = np.empty(V.dofmap.index_map.size_local, dtype=np.int64)
    file_index[V.dofmap.list.ravel()] = np.asarray(domain.geometry.input_global_indices)[domain.geometry.dofmap.ravel()]

    x, displacement = np.empty((len(file_index), 3)), np.empty((len(file_index), 3))
    x[file_index] = V.tabulate_dof_coordinates()[:, :3]
    displacement[file_index] = u.x.array.reshape(-1, 3)
    return x, displacement


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
