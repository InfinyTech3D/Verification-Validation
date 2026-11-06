import json
import os
import sys

CASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(CASE_DIR, *[os.pardir] * 5))
for path in (CASE_DIR, REPO_ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)

import numpy as np

from common.materials import MATERIALS
from common.geometry import Beam2D
from common.sofa.conventions import ELEMENTS
from common.sofa.controllers import LoadRamp
from common.sofa.prefabs import ElasticBeam

PLUGINS = [
    "Sofa.Component.Constraint.Projective",
    "Sofa.Component.LinearSolver.Direct",
    "Sofa.Component.MechanicalLoad",
    "Sofa.Component.IntegrationScheme.Backward",
    "Sofa.Component.SolidMechanics.FEM.Elastic",
    "Sofa.Component.SolidMechanics.FEM.HyperElastic",
    "Sofa.Component.StateContainer",
    "Sofa.Component.Topology.Container.Dynamic",
    "Sofa.Component.Topology.Container.Grid",
    "Sofa.Component.Topology.Mapping",
]


def build(root, case):
    root.addObject('RequiredPlugin', pluginName=PLUGINS)
    root.addObject('DefaultAnimationLoop')

    geometry = Beam2D(length=case['geometry']['length'], width=case['geometry']['width'])
    material = case['material'] | {'sofa_component_name': MATERIALS[case['material']['type']].sofa_component_name}
    configs = {'material': material, 'forceField': case['forceField'], 'solvers': case['solvers']}
    beam = root.addChild(ElasticBeam(name='ElasticBeam', geometry=geometry, element=case['element'],
                                     grid_resolution=case['mesh']['resolution'], configs=configs))

    beam.Grid.grid.init()
    rest_positions = beam.Grid.grid.position.array()[:, :2]

    def node_at(x, y):
        return int(np.where(np.isclose(rest_positions[:, 0], x) & np.isclose(rest_positions[:, 1], y))[0][0])

    roller_indices = [node_at(0.0, 0.0), node_at(geometry.length, 0.0)]
    anchor_index = node_at(geometry.length / 2, 0.0)
    load_index = node_at(geometry.length / 2, geometry.width)

    with beam.mechanical_node() as mechanical:
        mechanical.addObject('PartialFixedProjectiveConstraint', name='rollers', indices=roller_indices,
                             fixedDirections=[0, 1])
        # Removes the rigid translation along x while keeping the solution symmetric about the midspan
        mechanical.addObject('PartialFixedProjectiveConstraint', name='anchor', indices=[anchor_index],
                             fixedDirections=[1, 0])
        point_load = mechanical.addObject('ConstantForceField', name='pointLoad', indices=[load_index],
                                          forces=[[0.0, -case['load']['magnitude']]])
        mechanical.addObject(LoadRamp(point_load, 'forces', case['load']['steps'], name='loadRamp'))
    return mechanical


def createScene(root):
    with open(os.path.join(CASE_DIR, 'case.json')) as f:
        case = json.load(f)

    if len(sys.argv) != 2:
        sys.exit("sofa_scene.py under runSofa expects: --argv <law>_<element>, e.g. NeoHookean_tet")

    build(root, case[sys.argv[1]])
    root.addObject('RequiredPlugin', pluginName=["Sofa.Component.Visual"])
    root.addObject('VisualStyle', displayFlags=['showBehaviorModels', 'showForceFields'])
    return root


def solve(case):
    import Sofa.Core
    import Sofa.Simulation

    root = Sofa.Core.Node('root')
    mechanical = build(root, case)
    Sofa.Simulation.init(root)
    for _ in range(case['load']['steps']):
        Sofa.Simulation.animate(root, root.dt.value)

    x0 = mechanical.dofs.rest_position.array()[:, :2].copy()
    u = mechanical.dofs.position.array()[:, :2] - x0
    cells = getattr(mechanical.topology, ELEMENTS[case['element']].data_name).array().copy()
    return x0, u, cells


def main(case):
    x, u, cells = solve(case)
    print(f"{case['element']}: solved {len(x)} nodes")


if __name__ == '__main__':
    with open(os.path.join(CASE_DIR, 'case.json')) as f:
        for sub_case in json.load(f).values():
            main(sub_case)
