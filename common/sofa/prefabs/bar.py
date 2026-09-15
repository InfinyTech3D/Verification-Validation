"""SOFA prefab: a 1D elastic bar, embeddable in 1D to 3D space."""

from .base import ScenePrefab
from ..conventions import VEC_DIM, ELEMENTS


def validate_parameters(config):
    """Required ElasticBar parameters."""
    required = ['geometry', 'resolution']
    missing = [p for p in required if p not in config]
    if missing:
        raise ValueError(f"ElasticBar: missing required parameters {missing}")


class ElasticBar(ScenePrefab):
    """Prefab for an elastic bar model in SOFA."""

    prefabParameters = [
        {'name': 'resolution', 'type': 'int', 'help': 'nodes along the bar'},
    ]

    def __init__(self, *args, **kwargs):
        validate_parameters(kwargs)
        super().__init__(*args, **kwargs)

    def init(self):
        VecType = VEC_DIM[self.geometry.spatial_dimensions]
        element_kind = ELEMENTS["edge"]

        # Grid Topology Node
        with self.addChild('Grid') as grid_node:
            grid_node.addObject('RegularGridTopology', name='grid',
                                nx=int(self.resolution.value), ny=1, nz=1,
                                min=[0.0, 0.0, 0.0], max=[self.geometry.length, 0.0, 0.0])

        # Node containing Bar components
        with self.addChild('Bar') as bar:
            self.bar = bar

            # Topology: edges are grid-native, so no topological mapping is ever needed.
            bar.addObject(element_kind.container, name='topology', position='@../Grid/grid.position',
                          **{element_kind.data_name: f'@../Grid/grid.{element_kind.data_name}'})
            # DOFs
            bar.addObject('MechanicalObject', name='dofs', template=VecType)
            # ForceField to test
            self.add_force_field(bar, self.spec['forceField'], self.spec['material'],
                                 template=f"{VecType},{element_kind.cpp}")
            # ODE & Linear Solvers
            self.add_solvers(bar, self.spec['solvers'])
