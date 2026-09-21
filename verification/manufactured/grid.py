"""Manufactured solutions stated on the grid geometry."""

import sympy as sp

from common.geometry import Grid

from .base import ManufacturedSolution


class Quadratic1D(ManufacturedSolution):
    """u(x) = [A x^2]: body force constant, so the nodal source is the source; clamped at 'left'."""

    geometry = Grid
    dim = 1
    prescribe_displacement_on = {"left": [1]}
    traction_on = ("right",)

    def displacement(self, coordinates):
        return [self.amplitude * coordinates[0] ** 2]


class Quadratic2D(ManufacturedSolution):
    """u(x, y) = [A x^2, A y^2]

    Body force constant and traction linear, so both are nodally exact.
    """

    geometry = Grid
    dim = 2
    # Uncomment to prescribe Dirichlet BCs on all boundaries
    # prescribe_displacement_on = {"left": [1, 0], "right": [1, 0], "bottom": [0, 1], "top": [0, 1]}
    prescribe_displacement_on = {"left": [1, 0], "bottom": [0, 1]}
    traction_on = ("left", "right", "bottom", "top")

    def displacement(self, coordinates):
        x, y = coordinates[0], coordinates[1]
        return [self.amplitude * x**2, self.amplitude * y**2]


class Trigonometric1D(ManufacturedSolution):
    """u(x) = [A sin(k x)], k = 2 pi / L: prescribed (clamped) at 'left', traction at 'right'."""

    geometry = Grid
    dim = 1
    prescribe_displacement_on = {"left": [1]}
    traction_on = ("right",)

    def displacement(self, coordinates):
        return [self.amplitude * sp.sin(self.wavenumber("length") * coordinates[0])]


class Trigonometric2D(ManufacturedSolution):
    """u(x, y) = [A sin(kx x) cos(ky y),
                 A cos(kx x) sin(ky y)]
    kx = 2 pi / L, ky = 2 pi / W

    u_x fixed on the left face, u_y on the bottom face, traction elsewhere.
    """

    geometry = Grid
    dim = 2
    # Uncomment to prescribe Dirichlet BCs on all boundaries
    # prescribe_displacement_on = {"left": [1, 0], "right": [1, 0], "bottom": [0, 1], "top": [0, 1]}
    prescribe_displacement_on = {"left": [1, 0], "bottom": [0, 1]}
    traction_on = ("left", "right", "bottom", "top")

    def displacement(self, coordinates):
        kx, ky = self.wavenumber("length"), self.wavenumber("width")
        x, y = coordinates[0], coordinates[1]
        return [self.amplitude * sp.sin(kx * x) * sp.cos(ky * y),
                self.amplitude * sp.cos(kx * x) * sp.sin(ky * y)]


class Trigonometric3D(ManufacturedSolution):
    """u(x, y, z) = [A sin(kx x) cos(ky y) cos(kz z),
                    A cos(kx x) sin(ky y) cos(kz z),
                    A cos(kx x) cos(ky y) sin(kz z)]
    kx = 2 pi / L, ky = 2 pi / W, kz = 2 pi / H

    Each component fixed on the low face of its own axis, traction elsewhere.
    """

    geometry = Grid
    dim = 3
    # Uncomment to prescribe Dirichlet BCs on all boundaries
    # prescribe_displacement_on = {"left": [1, 0, 0], "right": [1, 0, 0],
    #                              "bottom": [0, 1, 0], "top": [0, 1, 0],
    #                              "front": [0, 0, 1], "back": [0, 0, 1]}
    prescribe_displacement_on = {"left": [1, 0, 0], "bottom": [0, 1, 0], "front": [0, 0, 1]}
    traction_on = ("left", "right", "bottom", "top", "front", "back")

    def displacement(self, coordinates):
        kx, ky, kz = (self.wavenumber("length"), self.wavenumber("width"),
                      self.wavenumber("height"))
        x, y, z = coordinates
        return [self.amplitude * sp.sin(kx * x) * sp.cos(ky * y) * sp.cos(kz * z),
                self.amplitude * sp.cos(kx * x) * sp.sin(ky * y) * sp.cos(kz * z),
                self.amplitude * sp.cos(kx * x) * sp.cos(ky * y) * sp.sin(kz * z)]

