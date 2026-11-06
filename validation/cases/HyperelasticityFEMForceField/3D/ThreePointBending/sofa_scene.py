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
from common.geometry import Beam3D
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

    geometry = Beam3D(length=case['geometry']['length'], width=case['geometry']['width'],
                      height=case['geometry']['height'])
    material = case['material'] | {'sofa_component_name': MATERIALS[case['material']['type']].sofa_component_name}
    configs = {'material': material, 'forceField': case['forceField'], 'solvers': case['solvers'],
              'gridMapping': case.get('gridMapping', {})}
    beam = root.addChild(ElasticBeam(name='ElasticBeam', geometry=geometry, element=case['element'],
                                     grid_resolution=case['mesh']['resolution'], configs=configs))

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
        point_load = mechanical.addObject('ConstantForceField', name='pointLoad', indices=load_line,
                                          forces=[[0.0, -case['load']['magnitude'], 0.0]] * len(load_line))
        mechanical.addObject(LoadRamp(point_load, 'forces', case['load']['steps'], name='loadRamp'))

        boundary_kind_name, mapping = ELEMENTS[case['element']].boundary_mappings[0]
        with mechanical.addChild(boundary_kind_name) as boundary:
            boundary.addObject(ELEMENTS[boundary_kind_name].container, name='topology')
            boundary.addObject(ELEMENTS[boundary_kind_name].container.replace('Container', 'Modifier'))
            boundary.addObject(mapping, input='@../topology', output='@topology')
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

    x0 = mechanical.dofs.rest_position.array()[:, :3].copy()
    u = mechanical.dofs.position.array()[:, :3] - x0
    boundary_kind_name, _ = ELEMENTS[case['element']].boundary_mappings[0]
    boundary = mechanical.getChild(boundary_kind_name).topology
    faces = getattr(boundary, ELEMENTS[boundary_kind_name].data_name).array().copy()
    return x0, u, faces


def main(case):
    x, u, faces = solve(case)
    print(f"{case['element']}: solved {len(x)} nodes")


if __name__ == '__main__':
    with open(os.path.join(CASE_DIR, 'case.json')) as f:
        for sub_case in json.load(f).values():
            main(sub_case)
