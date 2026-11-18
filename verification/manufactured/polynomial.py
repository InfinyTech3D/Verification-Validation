"""Manufactured solutions stated on any geometry."""

from common.geometry import Geometry

from .base import ManufacturedSolution


class Cubic2D(ManufacturedSolution):
    """u(x, y) = [A (x^3 + x y^2), A (y^3 + x^2 y)]"""

    geometry = Geometry
    dim = 2

    def displacement(self, coordinates):
        x, y = coordinates[0], coordinates[1]
        return [self.amplitude * (x**3 + x * y**2), self.amplitude * (y**3 + x**2 * y)]


class Cubic3D(ManufacturedSolution):
    """u(x, y, z) = [A (x^3 + y^2 z), A (y^3 + z^2 x), A (z^3 + x^2 y)]"""

    geometry = Geometry
    dim = 3

    def displacement(self, coordinates):
        x, y, z = coordinates
        return [self.amplitude * (x**3 + y**2 * z),
                self.amplitude * (y**3 + z**2 * x),
                self.amplitude * (z**3 + x**2 * y)]
