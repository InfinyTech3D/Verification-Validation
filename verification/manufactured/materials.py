"""Constitutive laws: a material declares psi(gradient); its stress and tangent are differentiated from that."""

from abc import ABC, abstractmethod

import sympy as sp


def _toLameParameters1D(youngModulus, poissonRatio):
    """(mu, lambda) at d = 1, where lambda + 2 mu = E collapses Hooke to sigma = E eps."""
    return 0.5 * youngModulus, 0.0


def _toLameParameters2D(youngModulus, poissonRatio):
    """(mu, lambda) at d = 2, in plane stress: the convention SOFA's force fields assume."""
    return (youngModulus / (2 * (1 + poissonRatio)),
            youngModulus * poissonRatio / (1 - poissonRatio ** 2))


def _toLameParameters3D(youngModulus, poissonRatio):
    """(mu, lambda) at d = 3."""
    return (youngModulus / (2 * (1 + poissonRatio)),
            youngModulus * poissonRatio / ((1 + poissonRatio) * (1 - 2 * poissonRatio)))


# dim -> Young/Poisson -> (mu, lambda)
_toLame = {1: _toLameParameters1D,
         2: _toLameParameters2D,
         3: _toLameParameters3D}


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
      fpk_stress(gradient)                    = d(psi)/d(gradient)                     -- a d x d tensor
      directional_tangent(gradient, direction) = d(fpk_stress)/d(gradient), applied to `direction` -- also d x d

    Preferably, for any given material, only psi is declared. fpk_stress and tangent are derived by 
    symbolic differentiation.
    """

    # True when the stress is symmetric for every displacement gradient
    symmetric_stress = False

    @classmethod
    @abstractmethod
    def get_material_parameters(cls, parameters, spatial_dimensions):
        """Return the constant parameters defining the material law from a dictionary."""

    @abstractmethod
    def energy_density(self, gradient):
        """psi(gradient): the strain energy density, an expression in `gradient`'s symbolic entries."""

    def fpk_stress(self, gradient):
        """d(psi)/d(gradient), evaluated at the given `gradient`."""
        gradient_symbols = _tensor(gradient.rows, "G")
        return self._fpk_stress(gradient_symbols).subs(dict(zip(gradient_symbols, gradient)))

    def cauchy_stress(self, gradient):
        """The Cauchy stress, evaluated at the given `gradient`."""
        return self.fpk_stress(gradient)

    def directional_tangent(self, gradient, direction):
        """d(stress)/d(gradient) applied to `direction`, evaluated at `gradient`."""
        gradient_symbols = _tensor(gradient.rows, "G")
        symbolic_stress = self._fpk_stress(gradient_symbols)
        rows, columns = gradient_symbols.rows, gradient_symbols.cols
        result = sp.Matrix(rows, columns, lambda i, j: sum(
            sp.diff(symbolic_stress[i, j], gradient_symbols[k, l]) * direction[k, l]
            for k in range(rows) for l in range(columns)))
        return result.subs(dict(zip(gradient_symbols, gradient)))

    def _fpk_stress(self, gradient_symbols):
        """The stress formula in terms of `gradient_symbols`"""
        psi = self.energy_density(gradient_symbols)
        return sp.Matrix(gradient_symbols.rows, gradient_symbols.cols,
                         lambda i, j: sp.diff(psi, gradient_symbols[i, j]))


# --- The materials themselves, one class per type of material. ---


class LinearElastic(Material):
    """Hooke: psi = 1/2 lambda tr(eps)^2 + mu eps:eps on the symmetric gradient eps."""

    symmetric_stress = True

    @classmethod
    def get_material_parameters(cls, parameters, spatial_dimensions):
        return _toLame[spatial_dimensions](parameters["youngModulus"], parameters["poissonRatio"])

    def __init__(self, mu, lam):
        self.mu = mu
        self.lam = lam

    def energy_density(self, gradient):
        strain = (gradient + gradient.T) / 2
        return self.lam * strain.trace() ** 2 / 2 + self.mu * sum(e ** 2 for e in strain)
