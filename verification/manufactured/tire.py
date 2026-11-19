"""Manufactured solutions stated on the tire geometry (Chamberland et al., 2010, Sec. 5.2)."""

import sympy as sp

from common.geometry import Tire

from .base import ManufacturedSolution


class Quartic2D(ManufacturedSolution):
    """u(x, y) = [A (x^4 + 2 x y / 5),
                  A (y^4 / 10 - 2 x y / 5)]

    u prescribed on the inner edge, traction on the outer.
    """

    geometry = Tire
    dim = 2

    def displacement(self, coordinates):
        x, y = coordinates[0], coordinates[1]
        return [self.amplitude * (x**4 + sp.Rational(2, 5) * x * y),
                self.amplitude * (sp.Rational(1, 10) * y**4 - sp.Rational(2, 5) * x * y)]


class Quartic3D(ManufacturedSolution):
    """u(X) = [X1^4 + 2 X2 X3 / 5,
               X2^4 + 2 X1 X3 / 5,
               X3^4 / 10 - 2 X1 X2 X3 / 5]

    u prescribed on the inner and outer curved surfaces, traction on the two flat ones.
    """

    geometry = Tire
    dim = 3

    def displacement(self, coordinates):
        x, y, z = coordinates
        X1, X2, X3 = x, z, -y

        u1 = X1**4 + sp.Rational(2, 5) * X2 * X3
        u2 = X2**4 + sp.Rational(2, 5) * X1 * X3
        u3 = sp.Rational(1, 10) * X3**4 - sp.Rational(2, 5) * X1 * X2 * X3

        return [self.amplitude * u1, -self.amplitude * u3, self.amplitude * u2]
