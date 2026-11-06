import json
import os
import sys

import numpy as np

CASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(CASE_DIR, *[os.pardir] * 5))
for path in (CASE_DIR, REPO_ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)

from common.materials import MATERIALS
from common.sofa.controllers import LoadRamp
from common.sofa.conventions import ELEMENTS
from common.sofa.prefabs.base import ScenePrefab

PLUGINS = [
    "Sofa.Component.Constraint.Projective",
    "Sofa.Component.Engine.Select",
    "Sofa.Component.IO.Mesh",
    "Sofa.Component.LinearSolver.Direct",
    "Sofa.Component.IntegrationScheme.Backward",
    "Sofa.Component.SolidMechanics.FEM.Elastic",
    "Sofa.Component.SolidMechanics.FEM.HyperElastic",
    "Sofa.Component.StateContainer",
    "Sofa.Component.Topology.Container.Dynamic",
    "Sofa.Component.Topology.Mapping",
]


def build(root, case):
    root.addObject('RequiredPlugin', pluginName=PLUGINS)
    root.addObject('DefaultAnimationLoop')

    element = ELEMENTS[case['element']]
    boundary_kind_name, mapping = element.boundary_mappings[0]
    boundary_kind = ELEMENTS[boundary_kind_name]

    half_span, half_arm_width = case['geometry']['halfSpan'], case['geometry']['halfArmWidth']
    half_thickness = case['geometry']['thickness'] / 2
    eps = 1e-6 * half_span
    arm_ends = [[half_span - eps, -half_arm_width - eps, -half_thickness - eps,
                 half_span + eps, half_arm_width + eps, half_thickness + eps],
                [-half_span - eps, -half_arm_width - eps, -half_thickness - eps,
                 -half_span + eps, half_arm_width + eps, half_thickness + eps],
                [-half_arm_width - eps, half_span - eps, -half_thickness - eps,
                 half_arm_width + eps, half_span + eps, half_thickness + eps],
                [-half_arm_width - eps, -half_span - eps, -half_thickness - eps,
                 half_arm_width + eps, -half_span + eps, half_thickness + eps]]

    with root.addChild('Cruciform') as mechanical:
        loader = mechanical.addObject('MeshGmshLoader', name='loader',
                                      filename=os.path.join(CASE_DIR, case['mesh']['file']))
        mechanical.addObject(element.container, name='topology', src='@loader')
        mechanical.addObject('MechanicalObject', name='dofs', template='Vec3d', src='@loader')
        material = case['material'] | {'sofa_component_name': MATERIALS[case['material']['type']].sofa_component_name}
        ScenePrefab.add_force_field(mechanical, case['forceField'], material, template=f'Vec3d,{element.cpp}',
                                    vec_template='Vec3d')
        ScenePrefab.add_solvers(mechanical, case['solvers'])

        loader.init()
        x = loader.position.array()

        # The tractions balance, leaving the 6 rigid-body modes free. They are pinned where the equibiaxial
        # symmetry already sets the displacement to zero: the centre, and the x and y axes on the mid-plane.
        centre = np.where(np.isclose(x, 0.0).all(axis=1))[0].tolist()
        x_axis = np.where(np.isclose(x[:, 1], 0.0) & np.isclose(x[:, 2], 0.0))[0].tolist()
        y_axis = np.where(np.isclose(x[:, 0], 0.0) & np.isclose(x[:, 2], 0.0))[0].tolist()
        mechanical.addObject('PartialFixedProjectiveConstraint', name='centre', indices=centre,
                             fixedDirections=[1, 1, 1])
        mechanical.addObject('PartialFixedProjectiveConstraint', name='xAxis', indices=x_axis,
                             fixedDirections=[0, 1, 1])
        mechanical.addObject('PartialFixedProjectiveConstraint', name='yAxis', indices=y_axis,
                             fixedDirections=[1, 0, 1])

        # The traction points outwards on each arm end: one value per node
        traction = case['load']['magnitude'] * np.column_stack([
            np.sign(x[:, 0]) * np.isclose(np.abs(x[:, 0]), half_span),
            np.sign(x[:, 1]) * np.isclose(np.abs(x[:, 1]), half_span),
            np.zeros(len(x))])

        with mechanical.addChild(boundary_kind_name) as boundary:
            boundary.addObject(boundary_kind.container, name='topology')
            boundary.addObject(boundary_kind.container.replace('Container', 'Modifier'))
            boundary.addObject(mapping, input='@../topology', output='@topology')

            boundary.addObject('BoxROI', name='armEnds', box=arm_ends, position='@../dofs.rest_position',
                               **{boundary_kind.data_name: f'@topology.{boundary_kind.data_name}'})

            with boundary.addChild('LoadedFace') as loaded_face:
                loaded_face.addObject(boundary_kind.container, name='face', position='@../../dofs.rest_position',
                                      **{boundary_kind.data_name:
                                         f'@../armEnds.{boundary_kind.data_name}InROI'})
                load_density = loaded_face.addObject('NodalSourceDensity', name='loadDensity', template='Vec3d',
                                                     property=traction.tolist())
                loaded_face.addObject(LoadRamp(load_density, 'property', case['load']['steps'], name='loadRamp'))
                loaded_face.addObject('VectorSourceTerm', name='traction', template=f'Vec3d,{boundary_kind.cpp}',
                                      sourceDensity='@loadDensity')
                loaded_face.addObject('FEMSourceTermIntegrator', name='load', template=f'Vec3d,{boundary_kind.cpp}',
                                      topology='@face', quadratureDegree=1, constantSources='@traction')
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
