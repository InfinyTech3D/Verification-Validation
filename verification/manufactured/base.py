"""Manufactured solutions: exact displacements declared symbolically, and the regions where
their BCs apply.
"""

from abc import ABC, abstractmethod

import sympy as sp


class ManufacturedSolution(ABC):
    """An exact displacement, declared symbolically, and the regions where its BCs apply.

    Stated without reference to a material; `ManufacturedProblem` pairs the two.
    """

    geometry: type                    # the geometry family this solution is stated on
    dim: int                          # the PDE's own dimension, not the space it is solved in

    def __init__(self, config, geometry, traction_on=None):
        if not isinstance(geometry, self.geometry):
            raise ValueError(f"{type(self).__name__} is stated on a {self.geometry.__name__}, "
                             f"got a {type(geometry).__name__}")
        if geometry.dim != self.dim:
            raise ValueError(f"{type(self).__name__} is {self.dim}D, "
                             f"got a {geometry.dim}D {type(geometry).__name__}")

        self.config = config                        # the case's "solution" block; a subclass may read more from it
        self.amplitude = config["amplitude"]        # too large relative to the geometry inverts elements
        self.parameters = geometry.named_parameters
        self.regions = list(geometry.region_names)
        self.traction_on = traction_on or {}        # {region: directions} the case hands to traction

    @property
    def prescribe_displacement_on(self):
        """{region: fixed_directions}: every direction of every region, less the ones the case
        handed to the traction. A region left with nothing fixed is dropped.
        """
        prescribed = {}
        for region in self.regions:
            free = self.traction_on.get(region, [0] * self.dim)
            fixed_directions = [1 - direction for direction in free]
            if any(fixed_directions):
                prescribed[region] = fixed_directions
        return prescribed

    @abstractmethod
    def displacement(self, coordinates):
        """Exact displacement as sympy expressions, one per component of this solution's own dimension."""


class TrigonometricSolution(ManufacturedSolution):
    """A solution built from sines and cosines fitted to the geometry's extent."""

    def __init__(self, config, geometry, traction_on=None):
        super().__init__(config, geometry, traction_on)
        self.periods = config.get("periods", 1)     # more periods need a finer mesh to resolve

    def wavenumber(self, parameter):
        """The wavenumber fitting `self.periods` full periods across one of the geometry's
        characteristic parameters, e.g. 2 pi / length for a single period.
        """
        return 2 * sp.pi * sp.Rational(str(self.periods)) / sp.Rational(str(self.parameters[parameter]))
