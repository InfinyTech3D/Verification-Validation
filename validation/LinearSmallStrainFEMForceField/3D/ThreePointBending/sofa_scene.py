import json
import os
import sys

CASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(CASE_DIR, *[os.pardir] * 4))
for path in (CASE_DIR, REPO_ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)

import numpy as np

from common.geometry import Beam3D
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

    geometry = Beam3D(length=deck['geometry']['length'], width=deck['geometry']['width'],
                      height=deck['geometry']['height'])
    configs = {'material': deck['material'], 'forceField': deck['forceField'], 'solvers': deck['solvers'],
              'gridMapping': deck.get('gridMapping', {})}
    beam = root.addChild(ElasticBeam(name='ElasticBeam', geometry=geometry, element=deck['element'],
                                     grid_resolution=deck['mesh']['resolution'], configs=configs))

    beam.Grid.grid.init()
    rest_positions = beam.Grid.grid.position.array()

    def nodes_at(x=None, y=None, z=None):
        mask = np.ones(len(rest_positions), dtype=bool)
        for axis, value in ((0, x), (1, y), (2, z)):
            if value is not None:
                mask &= np.isclose(rest_positions[:, axis], value)
        return np.where(mask)[0].tolist()

    pin_line = nodes_at(x=0.0, y=0.0)
    roller_line = nodes_at(x=geometry.length, y=0.0)
    z_pin = nodes_at(x=0.0, y=0.0, z=0.0)
    load_line = nodes_at(x=geometry.length / 2, y=geometry.width)

    with beam.mechanical_node() as mechanical:
        mechanical.addObject('PartialFixedProjectiveConstraint', name='pin', indices=pin_line,
                             fixedDirections=[1, 1, 0])
        mechanical.addObject('PartialFixedProjectiveConstraint', name='roller', indices=roller_line,
                             fixedDirections=[0, 1, 0])
        # Both support lines run the whole depth (z): nothing pins uz, leaving one rigid-body mode
        # (translation along z) free. Pin it at a single node, off the load line.
        mechanical.addObject('PartialFixedProjectiveConstraint', name='pinZ', indices=z_pin,
                             fixedDirections=[0, 0, 1])
        mechanical.addObject('ConstantForceField', name='pointLoad', indices=load_line,
                             forces=[[0.0, -deck['load']['magnitude'], 0.0]] * len(load_line))
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

    x0 = mechanical.dofs.rest_position.array()[:, :3].copy()
    u = mechanical.dofs.position.array()[:, :3] - x0
    return x0, u


def main(deck):
    x, u = solve(deck)
    print(f"{deck['element']}: solved {len(x)} nodes")


if __name__ == '__main__':
    with open(os.path.join(CASE_DIR, 'deck.json')) as f:
        whole_deck = json.load(f)

    for sub_deck in ([whole_deck] if 'geometry' in whole_deck else whole_deck.values()):
        main(sub_deck)
