"""Rectilinear structured grid geometry in 1 to 3 dimensions: bars and beams."""

import numpy as np

from .geometry import Geometry


class Grid(Geometry):
    """A rectilinear structured grid, axis i spanning 0..parameters[i]. Always starts at the origin."""

    PARAMETER_ORDER = ("length", "width", "height")
    AXIS_NAMES = (("left", "right"), ("bottom", "top"), ("front", "back"))

    @property
    def _analytic_predicates(self) -> dict:
        predicates = {}
        for axis, (low, high) in enumerate(self.AXIS_NAMES[:self.dim]):
            extent = self.parameters[axis]
            predicates[low] = lambda p, axis=axis: np.isclose(p[axis], 0.0)
            predicates[high] = lambda p, axis=axis, extent=extent: np.isclose(p[axis], extent)
        return predicates


class Bar1D(Grid):
    """A 1D bar occupying ``[0, length]`` along x."""
    dim = 1


class Beam2D(Grid):
    """A 2D rectangle occupying ``[0, length] x [0, width]`` in the xy-plane."""
    dim = 2


class Beam3D(Grid):
    """A 3D box occupying ``[0, length] x [0, width] x [0, height]``."""
    dim = 3
