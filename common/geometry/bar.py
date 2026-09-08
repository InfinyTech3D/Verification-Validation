"""1D bar geometry."""

import numpy as np

from .geometry import Geometry


class Bar1D(Geometry):
    """A 1D bar occupying ``[0, length]`` along x. Always starts at the origin."""

    dim = 1
    PARAMETER_NAMES = ("length",)

    def __init__(self, length, spatialDimensions=None):
        self._set_spatial_dimensions(spatialDimensions)
        self.length = length

    @property
    def boundary_regions(self) -> dict:
        L = self.length
        return {
            "left":  lambda p: np.isclose(p[0], 0.0),
            "right": lambda p: np.isclose(p[0], L),
        }
