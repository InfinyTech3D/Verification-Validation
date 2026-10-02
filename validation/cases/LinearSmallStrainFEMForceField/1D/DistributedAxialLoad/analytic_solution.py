"""Analytic solution for DistributedAxialLoad: a bar under a uniform distributed axial load.

u(x) = q/E * (length*x - x^2/2), for a bar clamped at x=0, free at x=length, under a uniform
axial body force q per unit length.
"""


def displacement(x, case):
    length = case['geometry']['length']
    young_modulus = case['material']['youngModulus']
    load = case['load']['magnitude']
    return load / young_modulus * (length * x - x ** 2 / 2)
