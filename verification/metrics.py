"""The quantities a level measures: one class per metric, so a metric is defined in one place."""

from abc import ABC, abstractmethod

import numpy as np


class Measurement:
    """One solved mesh refinement level, sampled at its quadrature points."""

    def __init__(self, quadrature, displacement, manufactured_problem):
        # MeshQuadrature Class
        self.quadrature = quadrature

        # ManufacturedProblem Class
        self.manufactured_problem = manufactured_problem

        # The FEM displacement and its gradient, interpolated from the field SOFA solved for to the
        # quadrature points.
        self.u_h = quadrature.values(displacement)
        self.grad_h = quadrature.grads(displacement)

        # The manufactured solution's analytical displacement and gradient, evaluated at those same
        # points.
        self.u_exact = quadrature.sample(manufactured_problem.u)
        self.grad_exact = quadrature.sample(manufactured_problem.grad_u)

        # u_error/grad_error are what a metric measures. 
        # u_exact/grad_exact (not the error) set its noise floor, computed through the same norm
        self.u_error = self.u_h - self.u_exact
        self.grad_error = self.grad_h - self.grad_exact

        self.energy_exact = quadrature.energy(self.grad_exact, manufactured_problem.energy_density)


class Metric(ABC):
    """A quantity measured at every refinement level: how to measure it, and what it is small relative to."""

    name = ""

    @abstractmethod
    def measure(self, measurement):
        """The metric's value on one level."""

    @abstractmethod
    def scale(self, measurement):
        """Reference magnitude the noise floor (noiseFloorFraction * this) is taken against:
        the norms scale with the field, the energy metrics with the energy."""


class L2(Metric):
    """Root of the integrated squared error, e = u_h - u."""

    name = "L2"

    def measure(self, measurement):
        return measurement.quadrature.l2(measurement.u_error)

    def scale(self, measurement):
        return measurement.quadrature.l2(measurement.u_exact)


class H1(Metric):
    """H1 semi-norm of the error: ‖∇e‖_F (Frobenius norm of the gradient), e = u_h - u."""

    name = "H1"

    def measure(self, measurement):
        return measurement.quadrature.frobenius(measurement.grad_error)

    def scale(self, measurement):
        return measurement.quadrature.frobenius(measurement.grad_exact)


class EnergyNorm(Metric):
    """The material-weighted H1 semi-norm the Galerkin solution minimizes in:

        ‖e‖_E = √(2 ∫ ψ(∇e) dΩ),   e = u_h - u

    where ψ is the material's strain energy density -- equal to the bilinear form a(e, e) for a
    linear material, since ψ is then exactly quadratic in the gradient.
    """

    name = "Enorm"

    def measure(self, measurement):
        return float(np.sqrt(2.0 * measurement.quadrature.energy(
            measurement.grad_error, measurement.manufactured_problem.energy_density)))

    def scale(self, measurement):
        return np.sqrt(2.0 * measurement.energy_exact)


# METRICS is the fixed list of metric instances every level is measured against. Order matters
# because nothing else fixes it: a study iterates over this list to set the column order of every
# printed table and the key order of every JSON record, so reordering it here reorders both.
#
# Expected orders are the deck's `expectedOrder`, not a constant here: they follow from the element,
# not the metric, so a P2 deck states different numbers for the same three metrics (P1: L2 -> 2,
# H1 -> 1, Enorm -> 1).
METRICS = [L2(), H1(), EnergyNorm()]
