"""Tool-agnostic geometry descriptions for the VnV suite."""

from abc import ABC, abstractmethod


class Geometry(ABC):
    """A shape parameterization plus its named boundary regions."""

    dim: int  # topological dimension of the shape (and of the elements meshing it)

    
    PARAMETER_NAMES: tuple #Named shape parameters e.g. Beam2D.PARAMETER_NAMES = ("length", "width")

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
    @abstractmethod
    def boundary_regions(self) -> dict:
        """Named boundary regions as ``{name: predicate(point) -> bool}``."""
