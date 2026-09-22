import json
import os
import sys

CASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(CASE_DIR, *[os.pardir] * 4))
for path in (CASE_DIR, REPO_ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)

import numpy as np

from common.geometry import Beam2D
from common.sofa.prefabs import ElasticBeam

PLUGINS = [
    "Sofa.Component.Constraint.Projective",
    "Sofa.Component.LinearSolver.Direct",
    "Sofa.Component.MechanicalLoad",
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

    def node_at(x, y):
        return int(np.where(np.isclose(rest_positions[:, 0], x) & np.isclose(rest_positions[:, 1], y))[0][0])

    pin_index = node_at(0.0, 0.0)
    roller_index = node_at(geometry.length, 0.0)
    load_index = node_at(geometry.length / 2, geometry.width)

    with beam.mechanical_node() as mechanical:
        mechanical.addObject('FixedProjectiveConstraint', name='pin', indices=[pin_index])
        mechanical.addObject('PartialFixedProjectiveConstraint', name='roller', indices=[roller_index],
                             fixedDirections=[0, 1])
        mechanical.addObject('ConstantForceField', name='pointLoad', indices=[load_index],
                             forces=[[0.0, -deck['load']['magnitude']]])
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
