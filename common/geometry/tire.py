"""Tire geometry (Chamberland et al., 2010, Fig. 1)."""

import numpy as np

from .geometry import Geometry


class Tire(Geometry):
    """An annulus in the xy-plane, centered on the origin, extruded along z when it is 3D."""

    PARAMETER_ORDER = ("inner_radius", "outer_radius", "extrusion_length")

    @property
    def _analytic_predicates(self) -> dict:
        predicates = {
            "inner": lambda p: np.isclose(np.hypot(p[0], p[1]), self.inner_radius),
            "outer": lambda p: np.isclose(np.hypot(p[0], p[1]), self.outer_radius),
        }
        if self.dim == 3:
            predicates["front"] = lambda p: np.isclose(p[2], 0.0)
            predicates["back"] = lambda p: np.isclose(p[2], self.extrusion_length)
        return predicates


class Tire2D(Tire):
    """An annulus occupying ``inner_radius <= r <= outer_radius`` of the xy-plane."""
    dim = 2


class Tire3D(Tire):
    """A Tire2D extruded along z from 0 to extrusion_length."""
    dim = 3
