"""Shared definition of the traction-bar case: the one place SOFA and FEniCS agree on.

A bar of unit cross-section occupying [0, length] along x, clamped at x=0 and pulled at
x=length by an axial point load. Small strain, linear elastic, P1 elements.

SOFA reduces isotropic elasticity to 1D as lambda + 2*mu with lambda = E*nu/(1+nu) and
mu = E/(2*(1+nu)), which collapses to E: the axial stiffness carries no Poisson effect, so
poissonRatio is inert here and FEniCS must use the same reduction to be comparable.
"""

import os

LENGTH = 1.0
CELLS = 9
YOUNG_MODULUS = 1000.0
POISSON_RATIO = 0.3
TRACTION = 1000.0

FENICS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fenics.json')
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')


def analytic_displacement(x):
    """u(x) = F*x/(E*A) for a bar of unit cross-section under an end load F."""
    return TRACTION * x / YOUNG_MODULUS
