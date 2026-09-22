"""Manufactured solutions stated on the grid geometry."""

import sympy as sp

from common.geometry import Grid

from .base import ManufacturedSolution, TrigonometricSolution


class Quadratic1D(ManufacturedSolution):
    """u(x) = [A x^2]: body force constant, so the nodal source is the source; clamped at 'left'."""

    geometry = Grid
    dim = 1

    def displacement(self, coordinates):
        return [self.amplitude * coordinates[0] ** 2]


class Quadratic2D(ManufacturedSolution):
    """u(x, y) = [A (x^2 + x y), A (y^2 + x y)]

    Body force constant and traction linear, so both are nodally exact.
    """

    geometry = Grid
    dim = 2

    def displacement(self, coordinates):
        x, y = coordinates[0], coordinates[1]
        return [self.amplitude * (x**2 + x * y), self.amplitude * (y**2 + x * y)]


class Quadratic3D(ManufacturedSolution):
    """u(x, y, z) = [A (x^2 + x y + x z), A (y^2 + y z + x y), A (z^2 + z x + y z)]

    Body force constant and traction linear, so both are nodally exact.
    """

    geometry = Grid
    dim = 3

    def displacement(self, coordinates):
        x, y, z = coordinates
        return [self.amplitude * (x**2 + x * y + x * z),
                self.amplitude * (y**2 + y * z + x * y),
                self.amplitude * (z**2 + z * x + y * z)]


class Trigonometric1D(TrigonometricSolution):
    """u(x) = [A sin(k x)], k = 2 pi / L: prescribed (clamped) at 'left', traction at 'right'."""

    geometry = Grid
    dim = 1

    def displacement(self, coordinates):
        return [self.amplitude * sp.sin(self.wavenumber("length") * coordinates[0])]


class Trigonometric2D(TrigonometricSolution):
    """u(x, y) = [A sin(kx x) cos(ky y),
                 A cos(kx x) sin(ky y)]
    kx = 2 pi / L, ky = 2 pi / W

    u_x fixed on the left face, u_y on the bottom face, traction elsewhere.
    """

    geometry = Grid
    dim = 2

    def displacement(self, coordinates):
        kx, ky = self.wavenumber("length"), self.wavenumber("width")
        x, y = coordinates[0], coordinates[1]
        return [self.amplitude * sp.sin(kx * x) * sp.cos(ky * y),
                self.amplitude * sp.cos(kx * x) * sp.sin(ky * y)]


class Trigonometric3D(TrigonometricSolution):
    """u(x, y, z) = [A sin(kx x) cos(ky y) cos(kz z),
                    A cos(kx x) sin(ky y) cos(kz z),
                    A cos(kx x) cos(ky y) sin(kz z)]
    kx = 2 pi / L, ky = 2 pi / W, kz = 2 pi / H

    Each component fixed on the low face of its own axis, traction elsewhere.
    """

    geometry = Grid
    dim = 3

    def displacement(self, coordinates):
        kx, ky, kz = (self.wavenumber("length"), self.wavenumber("width"),
                      self.wavenumber("height"))
        x, y, z = coordinates
        return [self.amplitude * sp.sin(kx * x) * sp.cos(ky * y) * sp.cos(kz * z),
                self.amplitude * sp.cos(kx * x) * sp.sin(ky * y) * sp.cos(kz * z),
                self.amplitude * sp.cos(kx * x) * sp.cos(ky * y) * sp.sin(kz * z)]

