"""1D bar under an end traction, solved in SOFA.

    runSofa -l SofaPython3 sofa_scene.py   inspect the scene graph
    python sofa_scene.py                   solve, compare against FEniCS, plot

The python path is canonical: it produces the cross-validation plot. FEniCS is not available
on CI, so the comparison reads the reference written by `fenics_scene.py` instead of solving.
"""

import json
import os
import sys

CASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(CASE_DIR, *[os.pardir] * 4))
for path in (CASE_DIR, REPO_ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)

from case import CELLS, FENICS_FILE, LENGTH, POISSON_RATIO, RESULTS_DIR, TRACTION, YOUNG_MODULUS, \
    analytic_displacement
from common.geometry import Bar1D
from common.sofa.prefabs import ElasticBar

PLUGINS = [
    "Sofa.Component.Constraint.Projective",
    "Sofa.Component.LinearSolver.Direct",
    "Sofa.Component.MechanicalLoad",
    "Sofa.Component.ODESolver.Backward",
    "Sofa.Component.SolidMechanics.FEM.Elastic",
    "Sofa.Component.StateContainer",
    "Sofa.Component.Topology.Container.Dynamic",
    "Sofa.Component.Topology.Container.Grid",
    "Sofa.Component.Visual",
]

MATERIAL = {'type': 'LinearElastic', 'youngModulus': YOUNG_MODULUS, 'poissonRatio': POISSON_RATIO}
FORCE_FIELD = {'type': 'LinearSmallStrainFEMForceField'}
SOLVERS = {
    'integration':  {'type': 'StaticSolver'},
    'newton':       {'type': 'NewtonRaphsonSolver', 'maxNbIterationsNewton': 1,
                     'relativeInitialStoppingThreshold': 1e-12,
                     'relativeSuccessiveStoppingThreshold': 0,
                     'absoluteResidualStoppingThreshold': 0},
    'linearSolver': {'type': 'SparseLDLSolver', 'template': 'CompressedRowSparseMatrixd'},
}

GEOMETRY = Bar1D(length=LENGTH)


def build(root):
    """Build the scene graph and hand back the node holding the dofs."""
    root.addObject('RequiredPlugin', pluginName=PLUGINS)
    root.addObject('DefaultAnimationLoop')
    root.addObject('VisualStyle', displayFlags=['showBehaviorModels', 'showForceFields'])

    configs = {'material': MATERIAL, 'forceField': FORCE_FIELD, 'solvers': SOLVERS}
    bar = root.addChild(ElasticBar(name='ElasticBar', geometry=GEOMETRY, element='edge',
                                   grid_resolution=[CELLS + 1, 1, 1], configs=configs))
    body = bar.mechanical_node()

    # The prefab meshes but imposes nothing: the case owns its boundary conditions, placed by
    # the geometry's named regions rather than by index.
    spacing = LENGTH / CELLS
    nodes = [[i * spacing] for i in range(CELLS + 1)]

    body.addObject('FixedProjectiveConstraint', name='clamp',
                   indices=GEOMETRY.region_indices('left', nodes))
    body.addObject('ConstantForceField', name='traction',
                   indices=GEOMETRY.region_indices('right', nodes),
                   forces=[[TRACTION]], showArrowSize=1e-4)
    return body


def createScene(root):
    build(root)
    return root


def solve():
    """One static step; returns the node coordinates and their axial displacement."""
    import Sofa.Core
    import Sofa.Simulation

    root = Sofa.Core.Node('root')
    body = build(root)
    Sofa.Simulation.init(root)
    Sofa.Simulation.animate(root, root.dt.value)

    dofs = body.dofs
    x = dofs.rest_position.array()[:, 0].copy()
    return x, dofs.position.array()[:, 0] - x


def compare(x, u):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    with open(FENICS_FILE) as f:
        reference = json.load(f)
    x_ref, u_ref = np.array(reference['x']), np.array(reference['u'])

    u_interp = np.interp(x_ref, x, u)
    scale = np.max(np.abs(u_ref)) or 1.0
    error = np.abs(u_interp - u_ref) / scale

    fine = np.linspace(0.0, LENGTH, 200)
    figure, (top, bottom) = plt.subplots(2, 1, figsize=(7, 7), sharex=True,
                                         gridspec_kw={'height_ratios': [3, 1]})
    top.plot(fine, analytic_displacement(fine), 'k--', lw=1, label='analytic')
    top.plot(x_ref, u_ref, '-', color='tab:blue', lw=2, alpha=.6, label='FEniCS')
    top.plot(x, u, 'o', color='tab:red', ms=5, label='SOFA')
    top.set_ylabel('axial displacement u(x)')
    top.legend()
    top.grid(alpha=.3)
    top.set_title(f'1D traction bar, {CELLS} P1 elements, E={YOUNG_MODULUS:g}, F={TRACTION:g}')

    bottom.semilogy(x_ref, np.maximum(error, 1e-18), 'o-', color='tab:purple', ms=4)
    bottom.set_xlabel('x')
    bottom.set_ylabel('|SOFA - FEniCS|\nrelative to max|u|')
    bottom.grid(alpha=.3)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    plot_path = os.path.join(RESULTS_DIR, 'traction_bar.png')
    figure.tight_layout()
    figure.savefig(plot_path, dpi=150)

    return error.max(), plot_path


def main():
    if not os.path.exists(FENICS_FILE):
        sys.exit(f"missing {FENICS_FILE}: run `python fenics_scene.py` in a FEniCS environment")

    x, u = solve()
    worst, plot_path = compare(x, u)
    print(f"max relative difference SOFA vs FEniCS: {worst:.3e}")
    print(f"plot: {plot_path}")


if __name__ == '__main__':
    main()
