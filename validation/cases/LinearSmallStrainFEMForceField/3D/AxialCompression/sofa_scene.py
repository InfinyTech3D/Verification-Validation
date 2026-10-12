import json
import os
import sys

CASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(CASE_DIR, *[os.pardir] * 5))
for path in (CASE_DIR, REPO_ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)

from common.geometry import Beam3D
from common.sofa.conventions import ELEMENTS, VEC_DIM
from common.sofa.controllers import RegionClamp
from common.sofa.prefabs import ElasticBeam

PLUGINS = [
    "Sofa.Component.Constraint.Projective",
    "Sofa.Component.Engine.Select",
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

    geometry = Beam3D(length=case['geometry']['length'], width=case['geometry']['width'],
                      height=case['geometry']['height'])
    configs = {'material': case['material'], 'forceField': case['forceField'], 'solvers': case['solvers'],
              'gridMapping': case.get('gridMapping', {})}
    beam = root.addChild(ElasticBeam(name='ElasticBeam', geometry=geometry, element=case['element'],
                                     grid_resolution=case['mesh']['resolution'], configs=configs))

    beam.Grid.grid.init()
    rest_positions = beam.Grid.grid.position.array()
    left_nodes = set(geometry.region_indices('left', rest_positions))
    bottom_nodes = set(geometry.region_indices('bottom', rest_positions))
    front_nodes = set(geometry.region_indices('front', rest_positions))
    top_nodes = set(geometry.region_indices('top', rest_positions))
    corner = sorted(left_nodes & bottom_nodes & front_nodes)
    second_corner = sorted(left_nodes & top_nodes & front_nodes)

    boundary_kind_name, mapping = ELEMENTS[case['element']].boundary_mappings[0]
    boundary_kind = ELEMENTS[boundary_kind_name]
    vec_type = VEC_DIM[geometry.spatial_dimensions]

    length = geometry.length
    eps = min(case['mesh']['resolution'][1:]) * 1e-6
    right_box = [length - eps, -eps, -eps, length + eps, geometry.width + eps, geometry.height + eps]

    with beam.mechanical_node() as mechanical:
        # Fix the axial displacement along x = 0, leaving it free to contract laterally.
        # Pin one corner in y and z to remove translation; pin a second corner off that axis in
        # z only, to remove the remaining rigid-body mode: rotation about x.
        mechanical.addObject(RegionClamp(geometry=geometry, dofs=mechanical.dofs, node=mechanical,
                                         prescribe_displacement_on={'left': [1, 0, 0]},
                                         vec_type=vec_type, name='clampCtrl'))
        mechanical.addObject('PartialFixedProjectiveConstraint', name='pinCorner',
                             template=vec_type, indices=corner, fixedDirections=[0, 1, 1])
        mechanical.addObject('PartialFixedProjectiveConstraint', name='pinSecondCorner',
                             template=vec_type, indices=second_corner, fixedDirections=[0, 0, 1])

        with mechanical.addChild(boundary_kind_name) as boundary:
            boundary.addObject(boundary_kind.container, name='topology')
            boundary.addObject(boundary_kind.container.replace('Container', 'Modifier'))
            boundary.addObject(mapping, input='@../topology', output='@topology')

            boundary.addObject('BoxROI', name='rightFace', box=right_box, position='@../dofs.rest_position',
                               **{boundary_kind.data_name: f'@topology.{boundary_kind.data_name}'})

            with boundary.addChild('LoadedFace') as loaded_face:
                loaded_face.addObject(boundary_kind.container, name='face', position='@../../dofs.rest_position',
                                      **{boundary_kind.data_name:
                                         f'@../rightFace.{boundary_kind.data_name}InROI'})
                loaded_face.addObject('NodalSourceDensity', name='loadDensity', template=vec_type,
                                      property=[[-case['load']['magnitude'], 0.0, 0.0]])
                loaded_face.addObject('VectorSourceTerm', name='traction', template=f'{vec_type},{boundary_kind.cpp}',
                                      sourceDensity='@loadDensity')
                loaded_face.addObject('FEMSourceTermIntegrator', name='load', template=f'{vec_type},{boundary_kind.cpp}',
                                      topology='@face', quadratureDegree=1, constantSources='@traction')
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

    x0 = mechanical.dofs.rest_position.array()[:, :3].copy()
    u = mechanical.dofs.position.array()[:, :3] - x0
    return x0, u


def main(case):
    x, u = solve(case)
    print(f"{case['element']}: solved {len(x)} nodes")


if __name__ == '__main__':
    with open(os.path.join(CASE_DIR, 'case.json')) as f:
        whole_case = json.load(f)

    for sub_case in ([whole_case] if 'geometry' in whole_case else whole_case.values()):
        main(sub_case)
