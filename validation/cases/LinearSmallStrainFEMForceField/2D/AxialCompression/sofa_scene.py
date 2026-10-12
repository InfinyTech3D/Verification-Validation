import json
import os
import sys

CASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(CASE_DIR, *[os.pardir] * 5))
for path in (CASE_DIR, REPO_ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)

from common.geometry import Beam2D
from common.sofa.conventions import VEC_DIM
from common.sofa.controllers import RegionClamp
from common.sofa.prefabs import ElasticBeam

PLUGINS = [
    "Sofa.Component.Constraint.Projective",
    "Sofa.Component.LinearSolver.Direct",
    "Sofa.Component.ODESolver.Backward",
    "Sofa.Component.SolidMechanics.FEM.Elastic",
    "Sofa.Component.StateContainer",
    "Sofa.Component.Topology.Container.Dynamic",
    "Sofa.Component.Topology.Container.Grid",
    "Sofa.Component.Topology.Mapping",
]


def build(root, case):
    root.addObject('RequiredPlugin', pluginName=PLUGINS)
    root.addObject('DefaultAnimationLoop')

    geometry = Beam2D(length=case['geometry']['length'], width=case['geometry']['width'])
    configs = {'material': case['material'], 'forceField': case['forceField'], 'solvers': case['solvers']}
    beam = root.addChild(ElasticBeam(name='ElasticBeam', geometry=geometry, element=case['element'],
                                     grid_resolution=case['mesh']['resolution'], configs=configs))

    beam.Grid.grid.init()
    rest_positions = beam.Grid.grid.position.array()[:, :2]
    left_nodes = set(geometry.region_indices('left', rest_positions))
    bottom_nodes = set(geometry.region_indices('bottom', rest_positions))
    corner = sorted(left_nodes & bottom_nodes)
    right_nodes = geometry.region_indices('right', rest_positions)
    right_edges = [[right_nodes[k], right_nodes[k + 1]] for k in range(len(right_nodes) - 1)]

    with beam.mechanical_node() as mechanical:
        # Fix the axial displacement along x = 0, leaving it free to contract laterally. 
        # Pin the lateral displacement at the corner to remove the remaining rigid-body mode.
        mechanical.addObject(RegionClamp(geometry=geometry, dofs=mechanical.dofs, node=mechanical,
                                         prescribe_displacement_on={'left': [1, 0]},
                                         vec_type=VEC_DIM[geometry.spatial_dimensions], name='clampCtrl'))
        mechanical.addObject('PartialFixedProjectiveConstraint', name='pinCorner',
                             template=VEC_DIM[geometry.spatial_dimensions],
                             indices=corner, fixedDirections=[0, 1])

        with mechanical.addChild('LoadedEdge') as loaded_edge:
            loaded_edge.addObject('EdgeSetTopologyContainer', name='edges', edges=right_edges)
            loaded_edge.addObject('NodalSourceDensity', name='loadDensity', template='Vec2d',
                                  property=[[-case['load']['magnitude'], 0.0]])
            loaded_edge.addObject('VectorSourceTerm', name='traction', template='Vec2d,Edge',
                                  sourceDensity='@loadDensity')
            loaded_edge.addObject('FEMSourceTermIntegrator', name='load', template='Vec2d,Edge',
                                  topology='@edges', quadratureDegree=1, constantSources='@traction')
    return mechanical


def createScene(root):
    with open(os.path.join(CASE_DIR, 'case.json')) as f:
        case = json.load(f)

    if 'geometry' not in case:
        if len(sys.argv) != 2:
            sys.exit("sofa_scene.py under runSofa expects: --argv <element>")
        case = case[sys.argv[1]]

    build(root, case)
    root.addObject('RequiredPlugin', pluginName=["Sofa.Component.Visual"])
    root.addObject('VisualStyle', displayFlags=['showBehaviorModels', 'showForceFields'])
    return root


def solve(case):
    import Sofa.Core
    import Sofa.Simulation

    root = Sofa.Core.Node('root')
    mechanical = build(root, case)
    Sofa.Simulation.init(root)
    Sofa.Simulation.animate(root, root.dt.value)

    x0 = mechanical.dofs.rest_position.array()[:, :2].copy()
    u = mechanical.dofs.position.array()[:, :2] - x0
    return x0, u


def main(case):
    x, u = solve(case)
    print(f"{case['element']}: solved {len(x)} nodes")


if __name__ == '__main__':
    with open(os.path.join(CASE_DIR, 'case.json')) as f:
        whole_case = json.load(f)

    for sub_case in ([whole_case] if 'geometry' in whole_case else whole_case.values()):
        main(sub_case)
