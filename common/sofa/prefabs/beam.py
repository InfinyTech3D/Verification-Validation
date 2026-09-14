"""SOFA prefab: a dimension/element-agnostic elastic beam."""

from .base import ScenePrefab
from ..conventions import VEC_DIM, ELEMENTS


def validate_parameters(config):
    """Required ElasticBeam parameters, and that the embedding space is at least 2D.

    A beam is a 2D or 3D continuum domain (plane or solid elasticity)
    """
    required = ['extents', 'resolution', 'spatialDimensions', 'element']
    missing = [p for p in required if p not in config]
    if missing:
        raise ValueError(f"ElasticBeam: missing required parameters {missing}")
    if config['spatialDimensions'] < 2:
        raise ValueError(f"ElasticBeam: spatialDimensions must be at least 2, "
                         f"got {config['spatialDimensions']}")


class ElasticBeam(ScenePrefab):
    """Prefab for an elastic beam model in SOFA."""

    prefabParameters = [
        {'name': 'extents',        'type': 'Vec3d',  'help': 'box max corner [Lx, Ly, Lz]'},
        {'name': 'resolution',     'type': 'Vec3d',  'help': 'nodes per axis [nx, ny, nz]'},
        {'name': 'spatialDimensions', 'type': 'int', 'help': 'dimension of the embedding space'},
        {'name': 'element',        'type': 'string', 'help': 'element element_kind (edge/tri/quad/tet/hexa)'},
    ]

    def __init__(self, *args, **kwargs):
        validate_parameters(kwargs)
        super().__init__(*args, **kwargs)

    def init(self):
        VecType = VEC_DIM[self.spatialDimensions.value]
        element_kind = ELEMENTS[self.element.value]
        res = self.resolution.value

        # Grid Topology Node
        with self.addChild('Grid') as grid_node:
            grid_node.addObject('RegularGridTopology', name='grid',
                                nx=int(res[0]), ny=int(res[1]), nz=int(res[2]),
                                min=[0.0, 0.0, 0.0], max=list(self.extents.value))

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
