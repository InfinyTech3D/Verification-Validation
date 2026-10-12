import json
import os

CASE_DIR = os.path.dirname(os.path.abspath(__file__))

import numpy as np
import ufl
from dolfinx import default_scalar_type, fem, mesh as dmesh
from dolfinx.fem.petsc import LinearProblem
from dolfinx.io import gmshio
from mpi4py import MPI

PETSC_OPTIONS = {'ksp_type': 'preonly', 'pc_type': 'lu'}


def build(case):
    domain = gmshio.read_from_msh(os.path.join(CASE_DIR, case['mesh']['file']), MPI.COMM_WORLD, gdim=3)[0]
    V = fem.functionspace(domain, ("Lagrange", 1, (3,)))

    young_modulus = case['material']['youngModulus']
    poisson_ratio = case['material']['poissonRatio']
    mu = young_modulus / (2 * (1 + poisson_ratio))
    lame_lambda = young_modulus * poisson_ratio / ((1 + poisson_ratio) * (1 - 2 * poisson_ratio))

    def sigma(u):
        eps = ufl.sym(ufl.grad(u))
        return lame_lambda * ufl.tr(eps) * ufl.Identity(3) + 2 * mu * eps

    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)

    right_facets = dmesh.locate_entities_boundary(
        domain, domain.topology.dim - 1, lambda x: np.isclose(x[0], case['geometry']['length']))
    facet_tags = dmesh.meshtags(domain, domain.topology.dim - 1, right_facets,
                                np.full_like(right_facets, 1, dtype=np.int32))
    ds = ufl.Measure('ds', domain=domain, subdomain_data=facet_tags)

    traction = fem.Constant(domain, default_scalar_type((0.0, -case['load']['magnitude'], 0.0)))

    a = ufl.inner(sigma(u), ufl.sym(ufl.grad(v))) * ufl.dx
    L = ufl.dot(traction, v) * ds(1)

    clamped = fem.locate_dofs_geometrical(V, lambda x: np.isclose(x[0], 0.0))
    bc = fem.dirichletbc(np.zeros(3, dtype=default_scalar_type), clamped, V)

    return V, a, L, [bc]


def solve(case):
    V, a, L, bcs = build(case)

    problem = LinearProblem(a, L, bcs=bcs, petsc_options=PETSC_OPTIONS)
    solution = problem.solve()

    # dolfinx renumbers the nodes it reads: put them back in the mesh file's order, which is SOFA's
    domain = V.mesh
    file_index = np.empty(V.dofmap.index_map.size_local, dtype=np.int64)
    file_index[V.dofmap.list.ravel()] = np.asarray(domain.geometry.input_global_indices)[domain.geometry.dofmap.ravel()]

    x, u = np.empty((len(file_index), 3)), np.empty((len(file_index), 3))
    x[file_index] = V.tabulate_dof_coordinates()[:, :3]
    u[file_index] = solution.x.array.reshape(-1, 3)
    return x, u


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
        for sub_case in json.load(f).values():
            main(sub_case)
