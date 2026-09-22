"""Analytic solution for DistributedAxialLoad: a bar under a uniform distributed axial load.

u(x) = q/E * (length*x - x^2/2), for a bar clamped at x=0, free at x=length, under a uniform
axial body force q per unit length.
"""


def displacement(x, deck):
    length = deck['geometry']['length']
    young_modulus = deck['material']['youngModulus']
    load = deck['load']['magnitude']
    return load / young_modulus * (length * x - x ** 2 / 2)
