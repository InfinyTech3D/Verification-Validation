"""Tire geometry (Chamberland et al., 2010, Fig. 1)."""

import numpy as np

from .geometry import Geometry


class Tire(Geometry):
    """An annulus in the xy-plane, extruded along z from 0 to extrusion_length.

    Centered on the z-axis. Dirichlet BCs go on the inner/outer curved (cylindrical)
    surfaces; Neumann BCs go on the two flat faces at z=0 and z=extrusion_length.
    """

    dim = 3
    PARAMETER_NAMES = ("inner_radius", "outer_radius", "extrusion_length")

    def __init__(self, inner_radius, outer_radius, extrusion_length, spatialDimensions=None):
        self._set_spatial_dimensions(spatialDimensions)
        self.inner_radius = inner_radius
        self.outer_radius = outer_radius
        self.extrusion_length = extrusion_length

    @property
    def boundary_regions(self) -> dict:
        def radius(p):
            return np.hypot(p[0], p[1])

        return {
            "inner": lambda p: np.isclose(radius(p), self.inner_radius),
            "outer": lambda p: np.isclose(radius(p), self.outer_radius),
            "front": lambda p: np.isclose(p[2], 0.0),
            "back":  lambda p: np.isclose(p[2], self.extrusion_length),
        }
