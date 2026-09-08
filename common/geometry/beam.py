"""2D and 3D beam geometry."""

import numpy as np

from .geometry import Geometry


class Beam2D(Geometry):
    """A 2D rectangle occupying ``[0, length] x [0, width]`` in the xy-plane. Always starts at the origin."""

    dim = 2
    PARAMETER_NAMES = ("length", "width")

    def __init__(self, length, width, spatialDimensions=None):
        self._set_spatial_dimensions(spatialDimensions)
        self.length = length
        self.width = width

    @property
    def boundary_regions(self) -> dict:
        L, W = self.length, self.width
        return {
            "left":   lambda p: np.isclose(p[0], 0.0),
            "right":  lambda p: np.isclose(p[0], L),
            "bottom": lambda p: np.isclose(p[1], 0.0),
            "top":    lambda p: np.isclose(p[1], W),
        }


class Beam3D(Geometry):
    """A 3D box occupying ``[0, length] x [0, width] x [0, height]``. Always starts at the origin."""

    dim = 3
    PARAMETER_NAMES = ("length", "width", "height")

    def __init__(self, length, width, height, spatialDimensions=None):
        self._set_spatial_dimensions(spatialDimensions)
        self.length = length
        self.width = width
        self.height = height

    @property
    def boundary_regions(self) -> dict:
        L, W, H = self.length, self.width, self.height
        return {
            "left":   lambda p: np.isclose(p[0], 0.0),
            "right":  lambda p: np.isclose(p[0], L),
            "bottom": lambda p: np.isclose(p[1], 0.0),
            "top":    lambda p: np.isclose(p[1], W),
            "front":  lambda p: np.isclose(p[2], 0.0),
            "back":   lambda p: np.isclose(p[2], H),
        }
