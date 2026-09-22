"""Manufactured solutions: exact displacements declared symbolically, and the problems a material
derives from one.
"""

from abc import ABC, abstractmethod

import numpy as np
import sympy as sp

from ..materials import compile_laws

_COORDINATES = sp.symbols("x y z")

_COMPONENT_NAMES = ("u_x", "u_y", "u_z")

# Rotation axis -> the two axes it turns, ordered to yield positive angle in right-handed system.
_ROTATION_PLANE = {"x": (1, 2), "y": (2, 0), "z": (0, 1)}


def rotation_matrix(config, dimensions):
    """A constant rotation from a deck's `rotation` block."""
    angle = sp.rad(sp.Rational(str(config["angleDegrees"])))
    cosine, sine = sp.cos(angle), sp.sin(angle)

    matrix = sp.eye(dimensions)
    first, second = _ROTATION_PLANE[config["axis"]] if dimensions == 3 else (0, 1)
    matrix[first, first] = cosine
    matrix[first, second] = -sine
    matrix[second, first] = sine
    matrix[second, second] = cosine
    return matrix


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

        self.config = config                        # the deck's "solution" block; a subclass may read more from it
        self.amplitude = config["amplitude"]        # too large relative to the geometry inverts elements
        self.parameters = geometry.named_parameters
        self.regions = list(geometry.region_names)
        self.traction_on = traction_on or {}        # {region: directions} the deck hands to traction

    @property
    def prescribe_displacement_on(self):
        """{region: fixed_directions}: every direction of every region, less the ones the deck
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


class ManufacturedProblem:
    """A manufactured solution paired with a material. This pairing determines displacement
    gradient, stress, body-force source, and energy density.
    """

    def __init__(self, solution, material, spatial_dimensions, rotation=None):
        self.solution = solution
        self.material = material
        self.rotation = rotation
        # Numeric once: the pullback runs on every quadrature point of every refinement level.
        self._rotation_transposed = (None if rotation is None
                                     else np.array(rotation.T.evalf(), dtype=float))
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

        displacement = sp.Matrix(components)
        # Build stress using the unrotated gradient
        stress = material.stress(displacement.jacobian(self.coordinates))

        if rotation is not None:
            displacement = rotation * (self.coordinates + displacement) - self.coordinates
            stress = rotation * stress

        # Keep a symbolic representation of the displacement: `equation` reads off this directly.
        self.displacement_expression = displacement

        gradient = displacement.jacobian(self.coordinates)
        source = -sp.Matrix([sum(sp.diff(stress[i, j], self.coordinates[j])
                                 for j in range(dimensions))
                             for i in range(dimensions)])

        self.u = self._compile_field(displacement, (dimensions,))
        self.grad_u = self._compile_field(gradient, (dimensions, dimensions))
        self.stress = self._compile_field(stress, (dimensions, dimensions))
        self.source = self._compile_field(source, (dimensions,))
        self.energy_density, self.tangent = compile_laws(material, dimensions)

    def rigid_positions(self, rest):
        """R X: the rest configuration turned by this problem's rotation, itself where there is none.

        The configuration a corotational force field reads as undeformed, so the solve starts there.
        """
        if self.rotation is None:
            return rest
        return np.asarray(rest) @ self._rotation_transposed

    def local_gradient(self, gradients):
        """Displacement gradients in the frame the material law is applied in: R^T (I + G) - I.

        Recovers the unrotated gradient, so the norms measure strain rather than the rigid rotation.
        """
        if self.rotation is None:
            return gradients
        identity = np.eye(len(self.coordinates))
        return np.einsum('ij,...jk->...ik', self._rotation_transposed,
                         identity + np.asarray(gradients)) - identity

    # The regions to apply BCs are further stated by its solution object.
    @property
    def prescribe_displacement_on(self):
        return self.solution.prescribe_displacement_on

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

