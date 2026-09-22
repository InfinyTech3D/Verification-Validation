import json
import os
import sys

CASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(CASE_DIR, *[os.pardir] * 4))
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


def build(root, deck):
    root.addObject('RequiredPlugin', pluginName=PLUGINS)
    root.addObject('DefaultAnimationLoop')

    geometry = Beam2D(length=deck['geometry']['length'], width=deck['geometry']['width'])
    configs = {'material': deck['material'], 'forceField': deck['forceField'], 'solvers': deck['solvers']}
    beam = root.addChild(ElasticBeam(name='ElasticBeam', geometry=geometry, element=deck['element'],
                                     grid_resolution=deck['mesh']['resolution'], configs=configs))

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
                                  property=[[-deck['load']['magnitude'], 0.0]])
            loaded_edge.addObject('VectorSourceTerm', name='traction', template='Vec2d,Edge',
                                  sourceDensity='@loadDensity')
            loaded_edge.addObject('FEMSourceTermIntegrator', name='load', template='Vec2d,Edge',
                                  topology='@edges', quadratureDegree=1, constantSources='@traction')
    return mechanical


def createScene(root):
    with open(os.path.join(CASE_DIR, 'deck.json')) as f:
        deck = json.load(f)

    if 'geometry' not in deck:
        if len(sys.argv) != 2:
            sys.exit("sofa_scene.py under runSofa expects: --argv <element>")
        deck = deck[sys.argv[1]]

    build(root, deck)
    root.addObject('RequiredPlugin', pluginName=["Sofa.Component.Visual"])
    root.addObject('VisualStyle', displayFlags=['showBehaviorModels', 'showForceFields'])
    return root


def solve(deck):
    import Sofa.Core
    import Sofa.Simulation

    root = Sofa.Core.Node('root')
    mechanical = build(root, deck)
    Sofa.Simulation.init(root)
    Sofa.Simulation.animate(root, root.dt.value)

    x0 = mechanical.dofs.rest_position.array()[:, :2].copy()
    u = mechanical.dofs.position.array()[:, :2] - x0
    return x0, u


def main(deck):
    x, u = solve(deck)
    print(f"{deck['element']}: solved {len(x)} nodes")


if __name__ == '__main__':
    with open(os.path.join(CASE_DIR, 'deck.json')) as f:
        whole_deck = json.load(f)

    for sub_deck in ([whole_deck] if 'geometry' in whole_deck else whole_deck.values()):
        main(sub_deck)
