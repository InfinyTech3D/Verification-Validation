"""FEniCS (dolfinx) meshes built from tool-agnostic Geometry."""

import basix.ufl
import numpy as np
import ufl
from dolfinx import mesh as dmesh
from mpi4py import MPI


class GridMesh:
    """A dolfinx mesh built from a Grid geometry, matching its extents and cell resolution.

    dolfinx renumbers a mesh's nodes on construction, so a direct comparison with a SOFA solution
    is not readily available. `inverse` undoes this remap allowing the comparison to happen. 
    The inversion only holds for a degree-1 Lagrange space, where each DOF is one mesh node.
    """

    def __init__(self, geometry, resolution):
        self.geometry = geometry
        self.resolution = resolution
        self.inverse = None

    def build(self, comm=MPI.COMM_WORLD):
        raise NotImplementedError


class BarMesh(GridMesh):
    """A 1D interval mesh matching a Bar1D geometry."""

    def build(self, comm=MPI.COMM_WORLD):
        cells = self.resolution[0] - 1
        mesh = dmesh.create_interval(comm, cells, [0.0, self.geometry.length])
        self.inverse = np.argsort(mesh.geometry.input_global_indices)
        return mesh


class BeamMesh(GridMesh):
    """A dolfinx mesh matching a Beam2D or Beam3D geometry, as SOFA would mesh it: quadrilaterals
    or triangles in 2D, hexahedra or tetrahedra in 3D.

    The nodes are ordered i + nx*(j + ny*k) like SOFA's RegularGridTopology (2D: i + nx*j). In 2D,
    each quad splits in two triangles along a diagonal; with `swapping`, this diagonal alternates
    per quad in a checkerboard pattern, else every quad splits the same way. In 3D, each
    hexahedron splits into 6 tets, following SOFA's Hexa2TetraTopologicalMapping: with `swapping`,
    a cell's corners are mirrored along X/Y/Z depending on its grid position before the tet
    pattern is applied; without it, every hexahedron splits the same way.
    """

    # SOFA's Hexahedron corner convention (n0..n7) and edge/tet patterns, from
    # sofa::geometry::Hexahedron and Hexa2TetraTopologicalMapping.cpp.
    SOFA_CORNERS = ((0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0), (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1))
    X_EDGES = ((0, 1), (4, 5), (3, 2), (7, 6))
    Y_EDGES = ((4, 7), (5, 6), (1, 2), (0, 3))
    Z_EDGES = ((4, 0), (5, 1), (6, 2), (7, 3))
    NON_SWAPPED = ((0, 5, 1, 6), (0, 1, 3, 6), (1, 3, 6, 2), (6, 3, 0, 7), (6, 7, 0, 5), (7, 5, 4, 0))
    SWAPPED = ((0, 5, 6, 1), (0, 1, 6, 3), (1, 3, 2, 6), (6, 3, 7, 0), (6, 7, 5, 0), (7, 5, 0, 4))
    # Hex cells need corners in basix's (0,0,0)..(1,1,1) lexicographic order, not SOFA's.
    BASIX_ORDER = (0, 1, 3, 2, 4, 5, 7, 6)

    def __init__(self, geometry, resolution, element='quad', swapping=True):
        super().__init__(geometry, resolution)
        self.element = element
        self.swapping = swapping

    def build(self, comm=MPI.COMM_WORLD):
        if self.geometry.dim == 2:
            nx, ny = self.resolution[0], self.resolution[1]
            dx, dy = self.geometry.length / (nx - 1), self.geometry.width / (ny - 1)

            def node(i, j):
                return i + nx * j

            points = np.array([[i * dx, j * dy] for j in range(ny) for i in range(nx)])

            cells = []
            for j in range(ny - 1):
                for i in range(nx - 1):
                    p0, p1, p2, p3 = node(i, j), node(i + 1, j), node(i + 1, j + 1), node(i, j + 1)
                    if self.element == 'quad':
                        cells.append([p0, p1, p3, p2])
                    else:
                        if self.swapping and (i ^ j) & 1:
                            cells += [[p0, p1, p3], [p2, p3, p1]]
                        else:
                            cells += [[p1, p2, p0], [p3, p0, p2]]

            cell_type = "quadrilateral" if self.element == 'quad' else "triangle"
            cell_element = basix.ufl.element("Lagrange", cell_type, 1, shape=(2,))
            mesh = dmesh.create_mesh(comm, np.array(cells), points, ufl.Mesh(cell_element))
            self.inverse = np.argsort(mesh.geometry.input_global_indices)
            return mesh

        nx, ny, nz = self.resolution[0], self.resolution[1], self.resolution[2]
        dx, dy, dz = (self.geometry.length / (nx - 1), self.geometry.width / (ny - 1),
                     self.geometry.height / (nz - 1))

        def node(i, j, k):
            return i + nx * (j + ny * k)

        points = np.array([[i * dx, j * dy, k * dz]
                           for k in range(nz) for j in range(ny) for i in range(nx)])

        cells = []
        for k in range(nz - 1):
            for j in range(ny - 1):
                for i in range(nx - 1):
                    corners = [node(i + di, j + dj, k + dk) for di, dj, dk in self.SOFA_CORNERS]
                    if self.element == 'hexa':
                        cells.append([corners[c] for c in self.BASIX_ORDER])
                        continue

                    swapped = False
                    if self.swapping:
                        if i % 2 == 0:
                            for v0, v1 in self.X_EDGES:
                                corners[v0], corners[v1] = corners[v1], corners[v0]
                            swapped = not swapped
                        if j % 2 == 1:
                            for v0, v1 in self.Y_EDGES:
                                corners[v0], corners[v1] = corners[v1], corners[v0]
                            swapped = not swapped
                        if k % 2 == 1:
                            for v0, v1 in self.Z_EDGES:
                                corners[v0], corners[v1] = corners[v1], corners[v0]
                            swapped = not swapped
                    pattern = self.SWAPPED if swapped else self.NON_SWAPPED
                    cells += [[corners[c] for c in tet] for tet in pattern]

        cell_type = "hexahedron" if self.element == 'hexa' else "tetrahedron"
        cell_element = basix.ufl.element("Lagrange", cell_type, 1, shape=(3,))
        mesh = dmesh.create_mesh(comm, np.array(cells), points, ufl.Mesh(cell_element))
        self.inverse = np.argsort(mesh.geometry.input_global_indices)
        return mesh
