"""Constitutive laws: a material declares psi(gradient); its stress and tangent are differentiated from that."""

from abc import ABC, abstractmethod

import numpy as np
import sympy as sp

import Sofa.SofaDeformable


def _toLameParameters1D(youngModulus, poissonRatio):
    """(mu, lambda) at d = 1, where lambda + 2 mu = E collapses Hooke to sigma = E eps."""
    return 0.5 * youngModulus, 0.0


# dim -> Young/Poisson -> (mu, lambda)
_toLame = {1: _toLameParameters1D,
         2: Sofa.SofaDeformable.toLameParameters2D,
         3: Sofa.SofaDeformable.toLameParameters3D}


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

    # The name of the SOFA component implementing the material law
    # None when no material component is used e.g. LinearSmallStrainFEMForceField
    sofa_component_name = None

    @classmethod
    @abstractmethod
    def get_material_parameters(cls, parameters, spatial_dimensions):
        """Return the constant parameters defining the material law from a dictionary."""

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

    @classmethod
    def get_material_parameters(cls, parameters, spatial_dimensions):
        return _toLame[spatial_dimensions](parameters["youngModulus"], parameters["poissonRatio"])

    def __init__(self, mu, lam):
        self.mu = mu
        self.lam = lam

    def energy_density(self, gradient):
        strain = (gradient + gradient.T) / 2
        return self.lam * strain.trace() ** 2 / 2 + self.mu * sum(e ** 2 for e in strain)


class MooneyRivlin(Material):
    """psi = mu10 (J^(-2/d) I1 - d) + mu01 (J^(-4/d) I2 - d(d-1)/2) + kappa/2 (ln J)^2.

    I1 = tr(C) and I2 = (I1^2 - tr(C^2)) / 2 on C = F^T F, J = det F, F = I + G.
    """

    sofa_component_name = "MooneyRivlinMaterial"

    @classmethod
    def get_material_parameters(cls, parameters, spatial_dimensions):
        return parameters["mu10"], parameters["mu01"], parameters["bulkModulus"]

    def __init__(self, mu10, mu01, bulk_modulus):
        self.mu10 = mu10
        self.mu01 = mu01
        self.bulk_modulus = bulk_modulus

    def energy_density(self, gradient):
        dimensions = gradient.rows
        deformation = sp.eye(dimensions) + gradient
        cauchy_green = deformation.T * deformation
        jacobian = deformation.det()
        first = cauchy_green.trace()
        second = (first ** 2 - (cauchy_green * cauchy_green).trace()) / 2
        return (self.mu10 * (jacobian ** sp.Rational(-2, dimensions) * first - dimensions)
                + self.mu01 * (jacobian ** sp.Rational(-4, dimensions) * second
                               - dimensions * (dimensions - 1) / 2)
                + self.bulk_modulus * sp.log(jacobian) ** 2 / 2)


class NeoHookean(Material):
    """psi = mu/2 (tr(C) - d - 2 ln J) + lambda/2 (ln J)^2 on C = F^T F, J = det F, F = I + G."""

    sofa_component_name = "NeoHookeanMaterial"

    @classmethod
    def get_material_parameters(cls, parameters, spatial_dimensions):
        return _toLame[spatial_dimensions](parameters["youngModulus"], parameters["poissonRatio"])

    def __init__(self, mu, lam):
        self.mu = mu
        self.lam = lam

    def energy_density(self, gradient):
        deformation = sp.eye(gradient.rows) + gradient
        log_jacobian = sp.log(deformation.det())
        return (self.mu * ((deformation.T * deformation).trace() - gradient.rows
                           - 2 * log_jacobian) / 2 + self.lam * log_jacobian ** 2 / 2)


class Ogden(Material):
    """psi = mu/alpha^2 (J^(-alpha/d) tr(C^(alpha/2)) - d) + kappa/2 (ln J)^2 

    with C = F^T F, J = det F, F = I + G.

    Formally, alpha can take any real value; zero alone is excluded, since it divides. However,
    there is a limitation in terms of symbolic differentiation here: tr(C^(alpha/2)) sums the
    eigenvalues of C, and beyond a 2x2 those have no closed form that survives differentiation. So
    2D takes any non-zero alpha, while 3D takes even alpha, where C^(alpha/2) is a plain matrix
    power and no eigenvalue is needed.
    """

    sofa_component_name = "OgdenMaterial"

    @classmethod
    def get_material_parameters(cls, parameters, spatial_dimensions):
        return parameters["mu"], parameters["alpha"], parameters["kappa"]

    def __init__(self, mu, alpha, kappa):
        if alpha == 0:
            raise ValueError("alpha must be non-zero")
        self.mu = mu
        self.alpha = sp.Rational(str(alpha))
        self.kappa = kappa

    def energy_density(self, gradient):
        dimensions = gradient.rows
        deformation = sp.eye(dimensions) + gradient
        cauchy_green = deformation.T * deformation
        jacobian = deformation.det()

        exponent = self.alpha / 2
        if exponent.is_integer:
            # A whole alpha/2 is a matrix power, keeping tr(C^(alpha/2)) polynomial at any dimension.
            stretch_sum = (cauchy_green ** int(exponent)).trace()
        elif dimensions == 2:
            # Otherwise the eigenvalues of C are needed. In 2D they are the roots of
            #   lambda^2 - tr(C) lambda + det(C),  i.e.  lambda_+- = (tr(C) +- sqrt(D)) / 2
            # with D = tr(C)^2 - 4 det(C) = (lambda_1 - lambda_2)^2, which vanishes under an
            # isotropic stretch. psi stays finite there, but D is a difference of nearly equal
            # numbers: once the two eigenvalues are within sqrt(eps) of each other it cancels to a
            # negative value and sqrt(D) turns NaN.
            discriminant = cauchy_green.trace() ** 2 - 4 * cauchy_green.det()
            stretch_sum = sum(((cauchy_green.trace() + sign * sp.sqrt(discriminant)) / 2) ** exponent
                              for sign in (1, -1))
        else:
            raise ValueError(f"alpha = {self.alpha} needs the eigenvalues of C, which beyond a 2x2 "
                             f"have no closed form that survives differentiation; at {dimensions}D "
                             f"alpha must be even")

        return (self.mu / self.alpha ** 2
                * (jacobian ** (-self.alpha / dimensions) * stretch_sum - dimensions)
                + self.kappa * sp.log(jacobian) ** 2 / 2)


class SaintVenantKirchhoff(Material):
    """psi = 1/2 lambda tr(E)^2 + mu E:E on the Green-Lagrange strain E = (F^T F - I) / 2, F = I + G.

    Stores nothing under a rigid rotation, and tends to LinearElastic as the gradient goes to zero.
    """

    sofa_component_name = "StVenantKirchhoffMaterial"

    @classmethod
    def get_material_parameters(cls, parameters, spatial_dimensions):
        return _toLame[spatial_dimensions](parameters["youngModulus"], parameters["poissonRatio"])

    def __init__(self, mu, lam):
        self.mu = mu
        self.lam = lam

    def energy_density(self, gradient):
        identity = sp.eye(gradient.rows)
        deformation = identity + gradient
        strain = (deformation.T * deformation - identity) / 2
        return self.lam * strain.trace() ** 2 / 2 + self.mu * sum(e ** 2 for e in strain)
