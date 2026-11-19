"""Tool-agnostic geometry descriptions for the VnV suite."""

from abc import ABC, abstractmethod


def _region_indices_from_mesh(mesh_path):
    """Node indices tagged by each physical group of a Gmsh .msh file, keyed by group name."""
    import meshio

    mesh = meshio.read(mesh_path)
    tag_names = {tag: name for name, (tag, _dim) in mesh.field_data.items()}
    groups = {}
    for block, tags in zip(mesh.cells, mesh.cell_data.get("gmsh:physical", [])):
        for element, tag in zip(block.data, tags):
            name = tag_names.get(tag)
            if name is not None:
                groups.setdefault(name, set()).update(element.tolist())
    return {name: sorted(indices) for name, indices in groups.items()}


class Geometry(ABC):
    """A shape parameterization plus its named boundary regions."""

    dim: int  # topological dimension of the shape (and of the elements meshing it)
    PARAMETER_ORDER: tuple  # the family's parameters, ordered by the dimension that introduces each

    def __init__(self, *, dim=None, spatialDimensions=None, meshPath=None, **parameters):
        """Build this shape, taking each name in its `PARAMETER_NAMES` as a keyword.

        e.g. ``Beam3D(length=2.0, width=1.0, height=1.0)``. Pass `dim` only for a family base,
        which names no dimension of its own: ``Grid(dim=3, length=2.0, width=1.0, height=1.0)``
        builds that same box.
        """
        stated = getattr(type(self), "dim", None)
        if stated is None and dim is None:
            raise TypeError(f"{type(self).__name__} states no dim: pass dim= to build one, or "
                            f"name the geometry that states it")
        if stated is not None and dim is not None and dim != stated:
            raise TypeError(f"{type(self).__name__} is {stated}D and cannot be built with dim={dim}")
        self.dim = stated if dim is None else dim

        if set(parameters) != set(self.PARAMETER_NAMES):
            raise TypeError(f"{type(self).__name__} takes {list(self.PARAMETER_NAMES)}, "
                            f"got {sorted(parameters)}")

        for name in self.PARAMETER_NAMES:
            setattr(self, name, parameters[name])
        self._set_spatial_dimensions(spatialDimensions)

        if meshPath is not None:
            self._regions = {name: (lambda nodes, indices=indices: indices)
                             for name, indices in _region_indices_from_mesh(meshPath).items()}
        else:
            self._regions = {name: (lambda nodes, predicate=predicate:
                                    [i for i, p in enumerate(nodes) if predicate(p)])
                             for name, predicate in self._analytic_predicates.items()}

    @property
    def PARAMETER_NAMES(self) -> tuple:
        """The parameters this shape takes: the first `dim` of its family's.

        Each dimension adds one, so a member takes as many as it has dimensions: of Grid's
        ("length", "width", "height"), Bar1D takes ("length",) and Beam2D ("length", "width").
        A family that does not grow this way sets `PARAMETER_NAMES` itself instead.
        """
        return self.PARAMETER_ORDER[:self.dim]

    @property
    def parameters(self) -> list:
        """This shape's defining parameters, read off the attributes named in `PARAMETER_NAMES`.

        e.g. Beam2D(length=2.0, width=0.5).parameters == [2.0, 0.5]
        """
        return [getattr(self, name) for name in self.PARAMETER_NAMES]

    @property
    def named_parameters(self) -> dict:
        """Parameters keyed by name: how a manufactured solution asks for one of them.

        e.g. Beam2D(length=2.0, width=0.5).named_parameters == {"length": 2.0, "width": 0.5}
        """
        return dict(zip(self.PARAMETER_NAMES, self.parameters))

    def _set_spatial_dimensions(self, spatial_dimensions):
        """Set how many dimensions of space this shape is embedded in; defaults to its own."""
        if spatial_dimensions is None:
            spatial_dimensions = self.dim

        if spatial_dimensions < self.dim:
            raise ValueError(
                f"{type(self).__name__} is {self.dim}D and cannot be embedded in a "
                f"{spatial_dimensions}D space: the embedding space must be at least as "
                f"large as the shape."
            )
        if spatial_dimensions > 3:
            raise ValueError(
                f"{type(self).__name__}: spatial_dimensions cannot exceed 3, "
                f"got {spatial_dimensions}."
            )

        self.spatial_dimensions = spatial_dimensions

    @property
    def region_names(self):
        """This geometry's named boundary regions."""
        return self._regions.keys()

    def region_indices(self, region, nodes) -> list:
        """Indices into `nodes` belonging to `region`."""
        return self._regions[region](nodes)

    @property
    @abstractmethod
    def _analytic_predicates(self) -> dict:
        """Named boundary regions as ``{name: predicate(point) -> bool}``, from this geometry's
        own shape parameters."""
