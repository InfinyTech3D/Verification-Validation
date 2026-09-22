"""Manufactured solutions: exact displacements declared symbolically, and the problems a material
derives from one.
"""

from abc import ABC, abstractmethod

import numpy as np
import sympy as sp

from ..materials import compile_laws

_COORDINATES = sp.symbols("x y z")

_COMPONENT_NAMES = ("u_x", "u_y", "u_z")


class ManufacturedSolution(ABC):
    """An exact displacement, declared symbolically, and the regions where its BCs apply.

    Stated without reference to a material; `ManufacturedProblem` pairs the two.
    """

    geometry: type                    # the geometry family this solution is stated on
    dim: int                          # the PDE's own dimension, not the space it is solved in
    prescribe_displacement_on = {}    # {region: fixed_directions} where u is prescribed
    traction_on = ()                  # regions where the derived traction is applied

    def __init__(self, config, geometry):
        if not isinstance(geometry, self.geometry):
            raise ValueError(f"{type(self).__name__} is stated on a {self.geometry.__name__}, "
                             f"got a {type(geometry).__name__}")
        if geometry.dim != self.dim:
            raise ValueError(f"{type(self).__name__} is {self.dim}D, "
                             f"got a {geometry.dim}D {type(geometry).__name__}")

        self.config = config                        # the deck's "solution" block; a subclass may read more from it
        self.amplitude = config["amplitude"]        # too large relative to the geometry inverts elements
        self.parameters = geometry.named_parameters

    @abstractmethod
    def displacement(self, coordinates):
        """Exact displacement as sympy expressions, one per component of this solution's own dimension."""


class TrigonometricSolution(ManufacturedSolution):
    """A solution built from sines and cosines fitted to the geometry's extent."""

    def __init__(self, config, geometry):
        super().__init__(config, geometry)
        self.periods = config.get("periods", 1)     # more periods need a finer mesh to resolve

    def wavenumber(self, parameter):
        """The wavenumber fitting `self.periods` full periods across one of the geometry's
        characteristic parameters, e.g. 2 pi / length for a single period.
        """
        return 2 * sp.pi * sp.Rational(str(self.periods)) / sp.Rational(str(self.parameters[parameter]))


class ManufacturedProblem:
    """A manufactured solution paired with a material. This pairing determines displacement
    gradient, stress, body-force source, and energy density.
    """

    def __init__(self, solution, material, spatial_dimensions):
        self.solution = solution
        self.material = material
        dimensions = spatial_dimensions

        self.coordinates = sp.Matrix(_COORDINATES[:dimensions])

        # The solution states its own dim components; embedding into higher spatial dimensions pads
        # the rest with zero.
        components = list(self.solution.displacement(_COORDINATES[:self.solution.dim]))
        if len(components) != self.solution.dim:
            raise ValueError(f"{type(self.solution).__name__}: displacement must state "
                             f"{self.solution.dim} components, one per dimension of the solution, "
                             f"got {len(components)}")
        components += [0] * (dimensions - self.solution.dim)

        # Keep a symbolic representation of the displacement: `equation` reads off this directly.
        self.displacement_expression = displacement = sp.Matrix(components)
        gradient = displacement.jacobian(self.coordinates)
        stress = material.stress(gradient)
        source = -sp.Matrix([sum(sp.diff(stress[i, j], self.coordinates[j])
                                 for j in range(dimensions))
                             for i in range(dimensions)])

        self.u = self._compile_field(displacement, (dimensions,))
        self.grad_u = self._compile_field(gradient, (dimensions, dimensions))
        self.stress = self._compile_field(stress, (dimensions, dimensions))
        self.source = self._compile_field(source, (dimensions,))
        self.energy_density, self.tangent = compile_laws(material, dimensions)

    # The regions to apply BCs are further stated by its solution object.
    @property
    def prescribe_displacement_on(self):
        return self.solution.prescribe_displacement_on

    @property
    def traction_on(self):
        return self.solution.traction_on

    @property
    def equation(self):
        """The manufactured displacement in LaTeX form, one line per component."""
        names = ("u",) if len(self.displacement_expression) == 1 else _COMPONENT_NAMES
        return "\n".join(f"${name} = {sp.latex(component)}$"
                         for name, component in zip(names, self.displacement_expression))

    def _compile_field(self, expression, shape):
        """Compile a symbolic expression of the coordinates into a numpy field function.

        The result, `evaluate(points)`, takes one point or a whole batch of points (shape
        (..., spatial_dimensions)) and returns the expression's value at each, shaped (*batch, *shape).
        """
        components = [sp.lambdify(list(self.coordinates), entry, "numpy") for entry in expression]

        def compiled_field(points):
            points = np.asarray(points, dtype=float)
            batch = points.shape[:-1]
            columns = [points[..., d] for d in range(len(self.coordinates))]
            values = np.empty(batch + shape, dtype=float)
            for index, component in zip(np.ndindex(*shape), components):
                values[(...,) + index] = np.broadcast_to(
                    np.asarray(component(*columns), dtype=float), batch)
            return values

        return compiled_field

