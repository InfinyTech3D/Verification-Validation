"""Central map from tool-agnostic vocabulary to SOFA components."""

from dataclasses import dataclass

# Spatial dimension (DOF/embedding space) -> MechanicalObject / force-field vector template.
VEC_DIM = {1: "Vec1d", 2: "Vec2d", 3: "Vec3d"}

@dataclass(frozen=True)
class ElementKind:
    """Everything the scene-building layer needs to know about one type of finite element."""

    container: str              # topology container class
    data_name: str              # Data field this kind's own container addresses its elements by
    cpp: str                    # SOFA geometry name, for compound templates ("Vec3d,Hexahedron")
    facet_kind: str | None      # the kind of this element's own facets, None if it has none
    grid_mapping: str | None    # topological mapping generating it from the grid, None if grid-native
    boundary_mappings: tuple    # (element kind, mapping) steps from this element to its boundary

ELEMENTS = {
    "edge": ElementKind(container="EdgeSetTopologyContainer",        data_name="edges",
                        cpp="Edge",        facet_kind=None,   grid_mapping=None,
                        boundary_mappings=()),
    "tri":  ElementKind(container="TriangleSetTopologyContainer",    data_name="triangles",
                        cpp="Triangle",    facet_kind="edge", grid_mapping="Quad2TriangleTopologicalMapping",
                        boundary_mappings=(("edge", "Triangle2EdgeTopologicalMapping"),)),
    "quad": ElementKind(container="QuadSetTopologyContainer",        data_name="quads",
                        cpp="Quad",        facet_kind="edge", grid_mapping=None,
                        # No Quad2Edge mapping exists, so the boundary is reached through a
                        # triangulation; the diagonals it adds are interior and drop out again.
                        boundary_mappings=(("tri", "Quad2TriangleTopologicalMapping"),
                                           ("edge", "Triangle2EdgeTopologicalMapping"))),
    "tet":  ElementKind(container="TetrahedronSetTopologyContainer", data_name="tetrahedra",
                        cpp="Tetrahedron", facet_kind="tri",  grid_mapping="Hexa2TetraTopologicalMapping",
                        boundary_mappings=(("tri", "Tetra2TriangleTopologicalMapping"),)),
    "hexa": ElementKind(container="HexahedronSetTopologyContainer",  data_name="hexahedra",
                        cpp="Hexahedron",  facet_kind="quad", grid_mapping=None,
                        boundary_mappings=(("quad", "Hexa2QuadTopologicalMapping"),)),
}
