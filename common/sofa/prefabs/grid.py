"""SOFA prefabs: an elastic rectilinear structured grid in 1 to 3 dimensions, as a bar or a beam."""

from .base import ScenePrefab, params
from ..conventions import VEC_DIM, ELEMENTS
from ...geometry import Grid

# Element kinds this prefab meshes, by the topological dimension of the geometry.
ELEMENT_KINDS = {1: ('edge',), 2: ('quad', 'tri'), 3: ('hexa', 'tet')}


def validate_parameters(name, config):
    """Required ElasticGrid parameters, and an element kind its geometry can be meshed with."""
    required = ['geometry', 'grid_resolution', 'element']
    missing = [p for p in required if p not in config]
    if missing:
        raise ValueError(f"{name}: missing required parameters {missing}")
    geometry = config['geometry']
    if not isinstance(geometry, Grid):
        raise ValueError(f"{name}: geometry must be a Grid, "
                         f"got {type(geometry).__name__}")
    kinds = ELEMENT_KINDS[geometry.dim]
    if config['element'] not in kinds:
        raise ValueError(f"{name}: element on a {geometry.dim}D {type(geometry).__name__} must be "
                         f"one of {list(kinds)}, got {config['element']!r}")


class ElasticGrid(ScenePrefab):
    """Prefab for an elastic rectilinear structured grid model in SOFA."""

    # The mechanical node's name in the scene graph; a named grid calls it what it is.
    BODY = 'Body'

    prefabParameters = [
        {'name': 'grid_resolution', 'type': 'Vec3d',  'help': 'grid points per axis [nx, ny, nz]'},
        {'name': 'element',    'type': 'string', 'help': 'element kind (edge/tri/quad/tet/hexa)'},
    ]

    def __init__(self, *args, **kwargs):
        validate_parameters(type(self).__name__, kwargs)
        super().__init__(*args, **kwargs)

    def init(self):
        dim = self.geometry.dim
        extents = list(self.geometry.parameters) + [0.0] * (3 - dim)
        VecType = VEC_DIM[self.geometry.spatial_dimensions]
        element_kind = ELEMENTS[self.element.value]
        grid_resolution = list(self.grid_resolution.value)[:dim] + [1] * (3 - dim)

        # Grid Topology Node
        with self.addChild('Grid') as grid_node:
            grid_node.addObject('RegularGridTopology', name='grid',
                                nx=int(grid_resolution[0]), ny=int(grid_resolution[1]),
                                nz=int(grid_resolution[2]), min=[0.0, 0.0, 0.0], max=extents)

        # Node containing the body's components
        with self.addChild(self.BODY) as body:
            self.body = body

            # Topology
            if element_kind.grid_mapping is None:
                body.addObject(element_kind.container, name='topology', position='@../Grid/grid.position',
                               **{element_kind.data_name: f'@../Grid/grid.{element_kind.data_name}'})
            else:
                body.addObject(element_kind.container, name='topology', position='@../Grid/grid.position')
                body.addObject(element_kind.grid_mapping, input='@../Grid/grid', output='@topology',
                               **params(self.configs.get('gridMapping', {})))
                body.addObject(element_kind.container.replace('Container', 'Modifier'))
            # DOFs
            body.addObject('MechanicalObject', name='dofs', template=VecType)
            # The force field under test.
            self.add_force_field(body, self.configs['forceField'], self.configs['material'],
                                 template=f"{VecType},{element_kind.cpp}")
            # ODE & Linear Solvers
            self.add_solvers(body, self.configs['solvers'])

    def mechanical_node(self):
        return self.body


class ElasticBar(ElasticGrid):
    """Prefab for an elastic bar model in SOFA."""

    BODY = 'Bar'


class ElasticBeam(ElasticGrid):
    """Prefab for an elastic beam model in SOFA."""

    BODY = 'Beam'
