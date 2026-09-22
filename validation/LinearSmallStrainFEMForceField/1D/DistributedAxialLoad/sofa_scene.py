import json
import os
import sys

CASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(CASE_DIR, *[os.pardir] * 4))
for path in (CASE_DIR, REPO_ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)

from common.geometry import Bar1D
from common.sofa.conventions import VEC_DIM
from common.sofa.controllers import RegionClamp
from common.sofa.prefabs import ElasticBar

PLUGINS = [
    "Sofa.Component.Constraint.Projective",
    "Sofa.Component.LinearSolver.Direct",
    "Sofa.Component.ODESolver.Backward",
    "Sofa.Component.SolidMechanics.FEM.Elastic",
    "Sofa.Component.StateContainer",
    "Sofa.Component.Topology.Container.Dynamic",
    "Sofa.Component.Topology.Container.Grid",
]


def build(root, deck):
    root.addObject('RequiredPlugin', pluginName=PLUGINS)
    root.addObject('DefaultAnimationLoop')

    geometry = Bar1D(length=deck['geometry']['length'])
    configs = {'material': deck['material'], 'forceField': deck['forceField'], 'solvers': deck['solvers']}
    bar = root.addChild(ElasticBar(name='ElasticBar', geometry=geometry, element=deck['element'],
                                   grid_resolution=deck['mesh']['resolution'], configs=configs))

    # Initialize grid points here to work with BCs
    bar.Grid.grid.init()

    with bar.mechanical_node() as mechanical:
        # Apply Dirichlet BCs on x = 0
        mechanical.addObject(RegionClamp(geometry=geometry, dofs=mechanical.dofs, node=mechanical,
                                         prescribe_displacement_on={'left': [1]},
                                         vec_type=VEC_DIM[geometry.spatial_dimensions], name='clampCtrl'))

        mechanical.addObject('NodalSourceDensity', name='loadDensity', template='Vec1d',
                             property=[[deck['load']['magnitude']]])
        mechanical.addObject('VectorSourceTerm', name='bodyForce', template='Vec1d,Edge',
                             sourceDensity='@loadDensity')
        mechanical.addObject('FEMSourceTermIntegrator', name='distributedLoad', template='Vec1d,Edge',
                             topology='@topology', quadratureDegree=1, constantSources='@bodyForce')
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

    x = mechanical.dofs.rest_position.array()[:, 0].copy()
    u = mechanical.dofs.position.array()[:, 0] - x
    return x, u


def main(deck):
    x, u = solve(deck)
    print(f"{deck['element']}: solved {len(x)} nodes")


if __name__ == '__main__':
    with open(os.path.join(CASE_DIR, 'deck.json')) as f:
        whole_deck = json.load(f)

    for sub_deck in ([whole_deck] if 'geometry' in whole_deck else whole_deck.values()):
        main(sub_deck)
