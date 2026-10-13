"""Converts the sympy material laws of common/materials.py into UFL."""

import sympy as sp
import ufl

from ..materials import MATERIALS


def strain_energy_density(material, gradient):
    """Converts the strain energy density of `material` from sympy to UFL, in the displacement gradient `gradient`."""
    dim = gradient.ufl_shape[0]
    law = MATERIALS[material['type']]
    symbols = sp.Matrix(dim, dim, lambda i, j: sp.Symbol(f"G_{i}_{j}"))
    energy = law(*law.get_material_parameters(material, dim)).energy_density(symbols)
    return sp.lambdify(list(symbols), energy, modules=[{'log': ufl.ln}])(
        *[gradient[i, j] for i in range(dim) for j in range(dim)])
