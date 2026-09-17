"""Central map from tool-agnostic vocabulary to SOFA components."""

from dataclasses import dataclass

# Spatial dimension (DOF/embedding space) -> MechanicalObject / force-field vector template.
VEC_DIM = {1: "Vec1d", 2: "Vec2d", 3: "Vec3d"}

@dataclass(frozen=True)
class ElementKind:
    """Everything the scene-building layer needs to know about one type of finite element."""

    dim: int                    # topological dimension of this element
    container: str              # topology container class
    data_name: str              # Data field this kind's own container addresses its elements by
    cpp: str                    # SOFA geometry name, for compound templates ("Vec3d,Hexahedron")
    facet_kind: str | None      # the kind of this element's own facets, None if it has none
    grid_mapping: str | None    # topological mapping generating it from the grid, None if grid-native
    boundary_mappings: tuple    # (element kind, mapping, Data) steps from this element to its boundary

ELEMENTS = {
    "edge": ElementKind(dim=1, container="EdgeSetTopologyContainer",        data_name="edges",
                        cpp="Edge",        facet_kind=None,   grid_mapping=None,
                        boundary_mappings=()),
    "tri":  ElementKind(dim=2, container="TriangleSetTopologyContainer",    data_name="triangles",
                        cpp="Triangle",    facet_kind="edge", grid_mapping="Quad2TriangleTopologicalMapping",
                        boundary_mappings=(("edge", "Triangle2EdgeTopologicalMapping", {}),)),
    "quad": ElementKind(dim=2, container="QuadSetTopologyContainer",        data_name="quads",
                        cpp="Quad",        facet_kind="edge", grid_mapping=None,
                        # To arrive at a Quad2Edge map -> triangulate quads and then Tri2Edge
                        boundary_mappings=(("tri", "Quad2TriangleTopologicalMapping", {}),
                                           ("edge", "Triangle2EdgeTopologicalMapping", {}))),
    "tet":  ElementKind(dim=3, container="TetrahedronSetTopologyContainer", data_name="tetrahedra",
                        cpp="Tetrahedron", facet_kind="tri",  grid_mapping="Hexa2TetraTopologicalMapping",
                        boundary_mappings=(("tri", "Tetra2TriangleTopologicalMapping",
                                            {"flipNormals": True}),)), # TODO Verify flipNormals is needed
    "hexa": ElementKind(dim=3, container="HexahedronSetTopologyContainer",  data_name="hexahedra",
                        cpp="Hexahedron",  facet_kind="quad", grid_mapping=None,
                        boundary_mappings=(("quad", "Hexa2QuadTopologicalMapping",
                                            {"flipNormals": True}),)), # TODO Verify flipNormals is needed
}
