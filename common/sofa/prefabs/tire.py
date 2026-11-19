"""SOFA prefab: an elastic tire -- an annulus, extruded along z when it is 3D."""

import itertools

import numpy as np

from .base import ScenePrefab
from ..conventions import VEC_DIM, ELEMENTS
from ...geometry import Tire

# Element kinds this prefab meshes, by the topological dimension of the geometry.
ELEMENT_KINDS = {2: ('quad', 'tri'), 3: ('hexa', 'tet')}

# The two triangles one quadrilateral splits into, on one diagonal and on the other.
_TRIANGLE_PATTERNS = (((1, 2, 0), (3, 0, 2)),
                      ((0, 1, 3), (2, 3, 1)))

# Swapping the two nodes of every hexahedron edge along one axis, as the permutation it applies to
# the eight corner slots. sofa::geometry::Hexahedron::{x,y,z}Edges.
_EDGE_SWAPS = ((1, 0, 3, 2, 5, 4, 7, 6),
               (3, 2, 1, 0, 7, 6, 5, 4),
               (4, 5, 6, 7, 0, 1, 2, 3))

# The six tetrahedra one hexahedron splits into, before and after an odd number of those swaps.
_TETRA_PATTERNS = {
    False: ((0, 5, 1, 6), (0, 1, 3, 6), (1, 3, 6, 2), (6, 3, 0, 7), (6, 7, 0, 5), (7, 5, 4, 0)),
    True:  ((0, 5, 6, 1), (0, 1, 6, 3), (1, 3, 2, 6), (6, 3, 7, 0), (6, 7, 5, 0), (7, 5, 0, 4)),
}


def validate_parameters(config):
    """Required ElasticTire parameters, and an element kind its geometry can be meshed with."""
    required = ['geometry', 'grid_resolution', 'element']
    missing = [p for p in required if p not in config]
    if missing:
        raise ValueError(f"ElasticTire: missing required parameters {missing}")
    geometry = config['geometry']
    if not isinstance(geometry, Tire):
        raise ValueError(f"ElasticTire: geometry must be a Tire2D or a Tire3D, "
                         f"got {type(geometry).__name__}")
    kinds = ELEMENT_KINDS[geometry.dim]
    if config['element'] not in kinds:
        raise ValueError(f"ElasticTire: element on a {type(geometry).__name__} must be one of "
                         f"{list(kinds)}, got {config['element']!r}")


def annulus(geometry, cells):
    """The annulus in the xy-plane: its nodes, and the quadrilaterals over them.

    `cells` counts cells radially and circumferentially. The ring closes on itself, so the
    circumferential direction holds exactly that many nodes, not one more.
    """
    radial, circumferential = cells

    radii = np.linspace(geometry.inner_radius, geometry.outer_radius, radial + 1)
    angles = np.linspace(0.0, 2.0 * np.pi, circumferential, endpoint=False)

    r, theta = np.meshgrid(radii, angles, indexing='ij')
    nodes = np.stack([r * np.cos(theta), r * np.sin(theta), np.zeros_like(r)],
                     axis=-1).reshape(-1, 3)

    node = np.arange(nodes.shape[0]).reshape(r.shape)
    i, j = np.mgrid[0:radial, 0:circumferential]
    ring = (j + 1) % circumferential   # the last cell of the ring closes back onto j = 0

    # (radial, circumferential) turns counter-clockwise, the winding GridTopology gives a cell.
    corners = (node[i, j], node[i + 1, j], node[i + 1, ring], node[i, ring])
    return nodes, np.stack(corners, axis=-1).reshape(-1, 4)


def extrude(nodes, quads, length, axial):
    """The annulus swept along z into `axial` layers: the swept nodes, and the hexahedra over them.

    Each layer repeats the annulus, so a hexahedron is its quadrilateral followed by the same one
    a layer up -- again the winding GridTopology gives a cell.
    """
    layer = nodes.shape[0]
    swept = np.tile(nodes, (axial + 1, 1))
    swept[:, 2] = np.repeat(np.linspace(0.0, length, axial + 1), layer)

    base = np.arange(axial)[:, None, None] * layer
    hexahedra = np.concatenate([quads[None] + base, quads[None] + base + layer], axis=-1)
    return swept, hexahedra.reshape(-1, 8)


def triangulate(quads, cells):
    """Two triangles per quadrilateral, split the way Quad2TriangleTopologicalMapping splits one.

    Neighbours share a whole edge, so any choice of diagonal conforms; alternating it cell by cell
    only keeps the mesh from leaning the same way everywhere.
    """
    i, j = np.mgrid[0:cells[0], 0:cells[1]]
    diagonal = ((i ^ j) & 1).reshape(-1)

    triangles = np.empty((quads.shape[0], 2, 3), dtype=np.int64)
    for choice, pattern in enumerate(_TRIANGLE_PATTERNS):
        selected = diagonal == choice
        triangles[selected] = quads[selected][:, pattern]

    return triangles.reshape(-1, 3)


def tetrahedralize(hexahedra, cells):
    """Six tetrahedra per hexahedron, split the way Hexa2TetraTopologicalMapping splits one.

    The node labelling alternates with the parity of the cell's index, so neighbours agree on the
    diagonal of every face they share. Across the seam that pairs the last circumferential cell
    with the first, which only agree when there is an even number of them.
    """
    radial, circumferential, axial = cells
    if circumferential % 2:
        raise ValueError(f"ElasticTire: an odd number of circumferential cells "
                         f"({circumferential}) leaves the tetrahedra of the closing cell "
                         f"disagreeing with the first on their shared diagonal")

    # Layer by layer, as `extrude` stacks them.
    k, i, j = np.mgrid[0:axial, 0:radial, 0:circumferential]
    parities = np.stack([i % 2 == 0, j % 2 == 1, k % 2 == 1], axis=-1).reshape(-1, 3)

    tetrahedra = np.empty((hexahedra.shape[0], 6, 4), dtype=np.int64)
    for signature in itertools.product((False, True), repeat=3):
        slots, swapped = list(range(8)), False
        for axis, swap in enumerate(signature):
            if swap:
                slots = [slots[p] for p in _EDGE_SWAPS[axis]]
                swapped = not swapped
        pattern = [[slots[corner] for corner in tetra] for tetra in _TETRA_PATTERNS[swapped]]

        selected = np.all(parities == signature, axis=-1)
        tetrahedra[selected] = hexahedra[selected][:, pattern]

    return tetrahedra.reshape(-1, 4)


def mesh(geometry, cells, element):
    """The nodes of `geometry` and its elements of the given kind."""
    nodes, elements = annulus(geometry, cells[:2])
    if geometry.dim == 3:
        nodes, elements = extrude(nodes, elements, geometry.extrusion_length, cells[2])

    if element == 'tri':
        elements = triangulate(elements, cells)
    elif element == 'tet':
        elements = tetrahedralize(elements, cells)
    return nodes, elements


class ElasticTire(ScenePrefab):
    """Prefab for an elastic tire model in SOFA."""

    prefabParameters = [
        {'name': 'grid_resolution', 'type': 'Vec3d',
         'help': 'nodes per axis [radial, circumferential] and [, axial] when extruded'},
        {'name': 'element',    'type': 'string', 'help': 'element kind (quad/tri/hexa/tet)'},
    ]

    def __init__(self, *args, **kwargs):
        validate_parameters(kwargs)
        super().__init__(*args, **kwargs)

    def init(self):
        VecType = VEC_DIM[self.geometry.spatial_dimensions]
        element_kind = ELEMENTS[self.element.value]
        # A study states nodes per axis, the mesh is laid out in cells.
        cells = [int(count) - 1 for count in
                 list(self.grid_resolution.value)[:self.geometry.dim]]

        nodes, elements = mesh(self.geometry, cells, self.element.value)

        # Node containing Tire components
        with self.addChild('Tire') as tire:
            self.tire = tire

            # Topology: meshed directly, so no grid and no topological mapping to reach it.
            tire.addObject(element_kind.container, name='topology', position=nodes.tolist(),
                           **{element_kind.data_name: elements.tolist()})
            # DOFs
            tire.addObject('MechanicalObject', name='dofs', template=VecType)
            # The force field under test.
            self.add_force_field(tire, self.configs['forceField'], self.configs['material'],
                                 template=f"{VecType},{element_kind.cpp}")
            # ODE & Linear Solvers
            self.add_solvers(tire, self.configs['solvers'])

    def mechanical_node(self):
        return self.tire
