"""Manufactured solutions: exact displacements declared symbolically, and the problems a material
derives from one.
"""

from abc import ABC, abstractmethod

import numpy as np
import sympy as sp

from .materials import compile_laws

_COORDINATES = sp.symbols("x y z")

_COMPONENT_NAMES = ("u_x", "u_y", "u_z")


class ManufacturedSolution(ABC):
    """An exact displacement, declared symbolically, and the regions where its BCs apply.

    Stated without reference to a material; `ManufacturedProblem` pairs the two.
    """

    prescribe_displacement_on = {}    # {region: fixed_directions} where u is prescribed
    traction_on = ()                  # regions where the derived traction is applied

    def __init__(self, spec, geometry):
        self.spec = spec                            # the deck's "function" block; a subclass may read more from it
        self.amplitude = spec["amplitude"]          # too large relative to the geometry inverts elements
        self.parameters = geometry.named_parameters
        self.dim = geometry.dim

    @abstractmethod
    def displacement(self, coordinates):
        """Exact displacement as sympy expressions, one per component of this solution's own dimension."""

    def wavenumber(self, parameter):
        """The wavenumber of one full period across one of the geometry's characteristic parameters.
        e.g. 2 pi / length"""
        return 2 * sp.pi / sp.Rational(str(self.parameters[parameter]))


class ManufacturedProblem:
    """A manufactured solution paired with a material. This pairing determines displacement
    gradient, stress, body-force source, and energy density.
    """

    def __init__(self, manufactured_solution, material, spatial_dimensions):
        self.solution = manufactured_solution
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
        # Each component is lambdified on its own.
        components = [sp.lambdify(list(self.coordinates), entry, "numpy") for entry in expression]

        # The compiled field itself, returned to the caller as a plain callable.
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


# --- The solutions themselves, one class per deck "function"; registry.py keys them by (dim, name). ---

class Quadratic1D(ManufacturedSolution):
    """u(x) = [A x^2]: body force constant, so the nodal source is the source; clamped at 'left'."""

    prescribe_displacement_on = {"left": [1]}
    traction_on = ("right",)

    def displacement(self, coordinates):
        return [self.amplitude * coordinates[0] ** 2]


class Quadratic2D(ManufacturedSolution):
    """u(x, y) = [A x^2, A y^2]

    Body force constant and traction linear, so both are nodally exact.
    """

    # Each face prescribes only the component that is constant on it -- u_x on the x-faces is
    # A x^2 at fixed x -- so the nodal values carry the Dirichlet data without error either.
    prescribe_displacement_on = {"left": [1, 0], "right": [1, 0], "bottom": [0, 1], "top": [0, 1]}
    traction_on = ("left", "right", "bottom", "top")

    def displacement(self, coordinates):
        x, y = coordinates[0], coordinates[1]
        return [self.amplitude * x**2, self.amplitude * y**2]


class Trigonometric1D(ManufacturedSolution):
    """u(x) = [A sin(k x)], k = 2 pi / L: prescribed (clamped) at 'left', traction at 'right'."""

    prescribe_displacement_on = {"left": [1]}
    traction_on = ("right",)

    def displacement(self, coordinates):
        return [self.amplitude * sp.sin(self.wavenumber("length") * coordinates[0])]


class Trigonometric2D(ManufacturedSolution):
    """u(x, y) = [A sin(kx x) cos(ky y),
                 A cos(kx x) sin(ky y)]
    kx = 2 pi / L, ky = 2 pi / W

    u_x fixed on x-faces, u_y fixed on y-faces, traction elsewhere.
    """

    prescribe_displacement_on = {"left": [1, 0], "right": [1, 0], "bottom": [0, 1], "top": [0, 1]}
    traction_on = ("left", "right", "bottom", "top")

    def displacement(self, coordinates):
        kx, ky = self.wavenumber("length"), self.wavenumber("width")
        x, y = coordinates[0], coordinates[1]
        return [self.amplitude * sp.sin(kx * x) * sp.cos(ky * y),
                self.amplitude * sp.cos(kx * x) * sp.sin(ky * y)]


class Trigonometric3D(ManufacturedSolution):
    """u(x, y, z) = [A sin(kx x) cos(ky y) cos(kz z),
                    A cos(kx x) sin(ky y) cos(kz z),
                    A cos(kx x) cos(ky y) sin(kz z)]
    kx = 2 pi / L, ky = 2 pi / W, kz = 2 pi / H

    Each component fixed on its own pair of faces, traction elsewhere.
    """

    prescribe_displacement_on = {"left": [1, 0, 0], "right": [1, 0, 0],
                                 "bottom": [0, 1, 0], "top": [0, 1, 0],
                                 "front": [0, 0, 1], "back": [0, 0, 1]}
    traction_on = ("left", "right", "bottom", "top", "front", "back")

    def displacement(self, coordinates):
        kx, ky, kz = (self.wavenumber("length"), self.wavenumber("width"),
                      self.wavenumber("height"))
        x, y, z = coordinates
        return [self.amplitude * sp.sin(kx * x) * sp.cos(ky * y) * sp.cos(kz * z),
                self.amplitude * sp.cos(kx * x) * sp.sin(ky * y) * sp.cos(kz * z),
                self.amplitude * sp.cos(kx * x) * sp.cos(ky * y) * sp.sin(kz * z)]
