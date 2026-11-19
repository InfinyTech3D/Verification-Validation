"""Manufactured problems: a manufactured solution paired with a material, and everything derived
from the pair.
"""

import numpy as np
import sympy as sp

from .materials import compile_laws

_COORDINATES = sp.symbols("x y z")

_COMPONENT_NAMES = ("u_x", "u_y", "u_z")


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

        displacement = sp.Matrix(components)
        stress = material.stress(displacement.jacobian(self.coordinates))

        # Keep a symbolic representation of the displacement: `equation` reads off this directly.
        self.displacement_expression = displacement
        self.stress_expression = stress

        gradient = displacement.jacobian(self.coordinates)
        source = -sp.Matrix([sum(sp.diff(stress[i, j], self.coordinates[j])
                                 for j in range(dimensions))
                             for i in range(dimensions)])
        self.source_expression = source

        self.u =self._compile_field(displacement, (dimensions,))
        self.grad_u = self._compile_field(gradient, (dimensions, dimensions))
        self.stress = self._compile_field(stress, (dimensions, dimensions))
        self.source = self._compile_field(source, (dimensions,))
        self.energy_density, self.tangent = compile_laws(material, dimensions)

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

