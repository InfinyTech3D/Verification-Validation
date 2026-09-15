"""SOFA prefab: a dimension/element-agnostic elastic beam."""

from .base import ScenePrefab
from ..conventions import VEC_DIM, ELEMENTS
from ...geometry import Beam2D, Beam3D


def validate_parameters(config):
    """Required ElasticBeam parameters, and that the embedding space is at least 2D.

    A beam is a 2D or 3D continuum domain (plane or solid elasticity).
    """
    required = ['geometry', 'resolution', 'element']
    missing = [p for p in required if p not in config]
    if missing:
        raise ValueError(f"ElasticBeam: missing required parameters {missing}")
    if not isinstance(config['geometry'], (Beam2D, Beam3D)):
        raise ValueError(f"ElasticBeam: geometry must be a Beam2D or Beam3D, "
                         f"got {type(config['geometry']).__name__}")
    if config['geometry'].spatial_dimensions < 2:
        raise ValueError(f"ElasticBeam: spatialDimensions must be at least 2, "
                         f"got {config['geometry'].spatial_dimensions}")


class ElasticBeam(ScenePrefab):
    """Prefab for an elastic beam model in SOFA."""

    prefabParameters = [
        {'name': 'resolution', 'type': 'Vec3d',  'help': 'nodes per axis [nx, ny, nz]'},
        {'name': 'element',    'type': 'string', 'help': 'element kind (edge/tri/quad/tet/hexa)'},
    ]

    def __init__(self, *args, **kwargs):
        validate_parameters(kwargs)
        super().__init__(*args, **kwargs)

    def init(self):
        extents = list(self.geometry.parameters) + [0.0] * (3 - len(self.geometry.parameters))
        VecType = VEC_DIM[self.geometry.spatial_dimensions]
        element_kind = ELEMENTS[self.element.value]
        res = self.resolution.value

        # Grid Topology Node
        with self.addChild('Grid') as grid_node:
            grid_node.addObject('RegularGridTopology', name='grid',
                                nx=int(res[0]), ny=int(res[1]), nz=int(res[2]),
                                min=[0.0, 0.0, 0.0], max=extents)

        # Node containing Beam components
        with self.addChild('Beam') as beam:
            self.beam = beam

            # Topology
            if element_kind.grid_mapping is None:
                beam.addObject(element_kind.container, name='topology', position='@../Grid/grid.position',
                               **{element_kind.data_name: f'@../Grid/grid.{element_kind.data_name}'})
            else:
                beam.addObject(element_kind.container, name='topology', position='@../Grid/grid.position')
                beam.addObject(element_kind.grid_mapping, input='@../Grid/grid', output='@topology')
                beam.addObject(element_kind.container.replace('Container', 'Modifier'))
            # DOFs
            beam.addObject('MechanicalObject', name='dofs', template=VecType)
            # ForceField to test
            self.add_force_field(beam, self.spec['forceField'], self.spec['material'],
                                 template=f"{VecType},{element_kind.cpp}")
            # ODE & Linear Solvers
            self.add_solvers(beam, self.spec['solvers'])
