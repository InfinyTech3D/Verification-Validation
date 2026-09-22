"""Analytic solution for AxialCompression: a beam under a uniform compressive edge traction.

u_x(x) = stress*x/E, u_y(x, y) = -nu*stress*y/E, for a beam clamped at x=0, under a uniform
compressive traction on x=length, plane stress.
"""

import numpy as np


def displacement(x, deck):
    young_modulus = deck['material']['youngModulus']
    poisson_ratio = deck['material']['poissonRatio']
    stress = -deck['load']['magnitude']
    ux = stress * x[:, 0] / young_modulus
    uy = -poisson_ratio * stress * x[:, 1] / young_modulus
    return np.column_stack([ux, uy])
