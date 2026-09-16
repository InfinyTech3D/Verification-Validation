"""Constitutive laws: a material declares psi(gradient); its stress and tangent are differentiated from that."""

from abc import ABC, abstractmethod

import numpy as np
import sympy as sp


def _tensor(dimensions, label):
    """A d x d matrix of independent symbols, named "<label>_i_j", one per entry.

    Stand-in for a real tensor (a gradient, a direction, ...) so an expression can be differentiated
    symbolically before any actual numbers exist.
    """
    return sp.Matrix(dimensions, dimensions, lambda i, j: sp.Symbol(f"{label}_{i}_{j}"))


class Material(ABC):
    """A material as one strain energy density psi(gradient); stress and tangent come from it.

    psi is the elastic energy stored per unit volume.
    It is defined for a given displacement gradient (a d x d matrix, d = spatial dimension).
    Everything else is a derivative of psi:
      stress(gradient)                        = d(psi)/d(gradient)                     -- a d x d tensor
      directional_tangent(gradient, direction) = d(stress)/d(gradient), applied to `direction` -- also d x d

    Preferably, for any given material, only psi is declared. stress and tangent are derived by 
    symbolic differentiation.
    """

    @abstractmethod
    def energy_density(self, gradient):
        """psi(gradient): the strain energy density, an expression in `gradient`'s symbolic entries."""

    def stress(self, gradient):
        """d(psi)/d(gradient), evaluated at the given `gradient`."""
        gradient_symbols = _tensor(gradient.rows, "G")
        return self._stress(gradient_symbols).subs(dict(zip(gradient_symbols, gradient)))

    def directional_tangent(self, gradient, direction):
        """d(stress)/d(gradient) applied to `direction`, evaluated at `gradient`."""
        gradient_symbols = _tensor(gradient.rows, "G")
        symbolic_stress = self._stress(gradient_symbols)
        rows, columns = gradient_symbols.rows, gradient_symbols.cols
        result = sp.Matrix(rows, columns, lambda i, j: sum(
            sp.diff(symbolic_stress[i, j], gradient_symbols[k, l]) * direction[k, l]
            for k in range(rows) for l in range(columns)))
        return result.subs(dict(zip(gradient_symbols, gradient)))

    def _stress(self, gradient_symbols):
        """The stress formula in terms of `gradient_symbols`"""
        psi = self.energy_density(gradient_symbols)
        return sp.Matrix(gradient_symbols.rows, gradient_symbols.cols,
                         lambda i, j: sp.diff(psi, gradient_symbols[i, j]))


def compile_laws(material, dimensions):
    """Compile and hand back a material's psi and tangent into plain numpy functions over batches of gradients."""
    gradient, direction = _tensor(dimensions, "G"), _tensor(dimensions, "D")
    arguments = list(gradient) + list(direction)
    energy_law = sp.lambdify(list(gradient), material.energy_density(gradient), "numpy")
    directional_tangent_law = [sp.lambdify(arguments, entry, "numpy")
                               for entry in material.directional_tangent(gradient, direction)]

    def columns_of(values):
        values = np.asarray(values, dtype=float)
        return values, [values[..., i, j]
                        for i in range(dimensions) for j in range(dimensions)]

    def compiled_energy_density(gradients):
        values, columns = columns_of(gradients)
        return np.broadcast_to(np.asarray(energy_law(*columns), dtype=float), values.shape[:-2])

    def compiled_directional_tangent(at, applied_to):
        base, base_columns = columns_of(at)
        applied, applied_columns = columns_of(applied_to)
        batch = np.broadcast_shapes(base.shape[:-2], applied.shape[:-2])
        result = np.empty(batch + (dimensions, dimensions), dtype=float)
        for (i, j), component in zip(np.ndindex(dimensions, dimensions), directional_tangent_law):
            result[..., i, j] = np.broadcast_to(
                np.asarray(component(*base_columns, *applied_columns), dtype=float), batch)
        return result

    return compiled_energy_density, compiled_directional_tangent


# --- The materials themselves, one class per type of material. ---


class LinearElastic(Material):
    """Hooke: psi = 1/2 lambda tr(eps)^2 + mu eps:eps on the symmetric gradient eps."""

    def __init__(self, mu, lam):
        self.mu = mu
        self.lam = lam

    def energy_density(self, gradient):
        strain = (gradient + gradient.T) / 2
        return self.lam * strain.trace() ** 2 / 2 + self.mu * sum(e ** 2 for e in strain)
