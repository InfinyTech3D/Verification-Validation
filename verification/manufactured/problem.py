"""Manufactured problems: a manufactured solution paired with a material, and everything derived
from the pair.
"""

import numpy as np
import sympy as sp

_COORDINATES = sp.symbols("x y z")

_COMPONENT_NAMES = ("u_x", "u_y", "u_z")

# Rotation axis -> the two axes it turns, ordered to yield positive angle in right-handed system.
_ROTATION_PLANE = {"x": (1, 2), "y": (2, 0), "z": (0, 1)}


def rotation_matrix(config, dimensions):
    """The rotation of a case's `rotation` block: constant by `angleDegrees`, or a field whose angle
    grows linearly, by `angleDegreesPerLength[axis]` along each axis named there.
    """
    if "angleDegreesPerLength" in config:
        # theta(x) = sum_k alpha_k x_k
        angle = sum(sp.rad(sp.Rational(str(rate))) * _COORDINATES["xyz".index(axis)]
                    for axis, rate in config["angleDegreesPerLength"].items())
    else:
        # theta constant
        angle = sp.rad(sp.Rational(str(config["angleDegrees"])))
    cosine, sine = sp.cos(angle), sp.sin(angle)

    # R = [[cos theta, -sin theta], [sin theta, cos theta]] in the plane the axis turns, I elsewhere
    matrix = sp.eye(dimensions)
    first, second = _ROTATION_PLANE[config["axis"]] if dimensions == 3 else (0, 1)
    matrix[first, first] = cosine
    matrix[first, second] = -sine
    matrix[second, first] = sine
    matrix[second, second] = cosine
    return matrix


class ManufacturedProblem:
    """A manufactured solution paired with a material. This pairing determines displacement
    gradient, stress, body-force source, and energy density.

    The fields are numpy functions of the rest position: u, grad_u, fpk_stress, cauchy_stress, source,
    rotation_field, the rotation the case imposes, and frame_rotation, the rotation of the frame the
    material law is applied in. Both rotations are None without a rotation. fpk_stress_expression and
    source_expression are the symbolic forms of their fields, None where those are derived numerically.
    """

    def __init__(self, solution, material, spatial_dimensions, rotation=None):
        self.solution = solution
        self.material = material
        self.rotation = rotation
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
        self.parameter_values = solution.parameter_values | material.parameter_values

        # The sympy variables that become the inputs of the compiled numpy functions.
        # x, for the fields
        position = list(self.coordinates)
        # G, for the material laws
        gradient_symbols = sp.Matrix(dimensions, dimensions, lambda i, j: sp.Symbol(f"G_{i}_{j}"))
        # D, the direction the tangent is applied to
        direction_symbols = sp.Matrix(dimensions, dimensions, lambda i, j: sp.Symbol(f"D_{i}_{j}"))
        matrix = (dimensions, dimensions)

        # psi(G)
        self.energy_density = self._compile([material.energy_density(gradient_symbols)], [gradient_symbols], ())
        # dP/dG : D
        self.tangent = self._compile(material.directional_tangent(gradient_symbols, direction_symbols),
                                     [gradient_symbols, direction_symbols], matrix)

        # u_hat
        unrotated_displacement = sp.Matrix(components)
        displacement = unrotated_displacement

        self.symmetric_stress = material.symmetric_stress
        self.rotation_field = None
        constant_rotation = False
        if rotation is not None:
            # u = R (X + u_hat) - X
            displacement = rotation * (self.coordinates + unrotated_displacement) - self.coordinates
            # R(x)
            self.rotation_field = self._compile(rotation, [position], matrix)
            # R is the same at every x
            constant_rotation = not any(rotation.has(axis) for axis in self.coordinates)
            # If a rotation is prescribed, it applies to the stress breaking its symmetry
            self.symmetric_stress = False

        # Keep a symbolic representation of the displacement: `equation` reads off this directly.
        self.displacement_expression = displacement
        # grad u
        gradient = displacement.jacobian(self.coordinates)
        # u(x)
        self.u = self._compile(displacement, [position], (dimensions,))
        # grad u(x)
        self.grad_u = self._compile(gradient, [position], matrix)

        if rotation is None:
            # P = P_law(grad u),  sigma = sigma_law(grad u)
            self._derive_fields_symbolically(material.fpk_stress(gradient), material.cauchy_stress(gradient),
                                             frame_rotation=None)
        elif constant_rotation:
            # The stresses of the unrotated problem, turned by R: F = I + grad u = R (I + grad u_hat), so
            # R^T F - I = grad u_hat.
            # grad u_hat
            unrotated_gradient = unrotated_displacement.jacobian(self.coordinates)
            # P = R P_law(grad u_hat),  sigma = R sigma_law(grad u_hat) R^T
            self._derive_fields_symbolically(
                rotation * material.fpk_stress(unrotated_gradient),
                rotation * material.cauchy_stress(unrotated_gradient) * rotation.T,
                frame_rotation=rotation)
        else:
            # A varying rotation strains the body:
            #   F = I + grad u = R (I + grad u_hat) + sum_k (dR/dx_k (X + u_hat)) e_k^T
            # and R^T F - I is no longer grad u_hat. The frame of the material law is the polar rotation
            # of F:
            #   F = R_p U,  R_p a rotation,  U symmetric
            if dimensions == 2:
                # In 2D the polar rotation is a ratio of the entries of F.
                identity = sp.eye(dimensions)
                # F = I + grad u
                deformation = identity + gradient
                # R_p = [[a, -b], [b, a]] / sqrt(a^2 + b^2),  a = F_11 + F_22,  b = F_21 - F_12
                trace, skew = deformation[0, 0] + deformation[1, 1], deformation[1, 0] - deformation[0, 1]
                polar_rotation = sp.Matrix([[trace, -skew], [skew, trace]]) / sp.sqrt(trace ** 2 + skew ** 2)
                # R_p^T F - I
                local_gradient = polar_rotation.T * deformation - identity
                # P = R_p P_law(R_p^T F - I),  sigma = R_p sigma_law(R_p^T F - I) R_p^T
                self._derive_fields_symbolically(
                    polar_rotation * material.fpk_stress(local_gradient),
                    polar_rotation * material.cauchy_stress(local_gradient) * polar_rotation.T,
                    frame_rotation=polar_rotation)
            else:
                # Beyond 2D the polar rotation has no closed form to differentiate: R_p is computed at each
                # point from the SVD of F, so the stress laws are compiled as functions of G and evaluated
                # at R_p^T F - I.
                # P_law(G), sigma_law(G), turned into P = R_p P_law(R_p^T F - I) and
                # sigma = R_p sigma_law(R_p^T F - I) R_p^T inside _derive_fields_numerically
                self._derive_fields_numerically(
                    gradient,
                    self._compile(material.fpk_stress(gradient_symbols), [gradient_symbols], matrix),
                    self._compile(material.cauchy_stress(gradient_symbols), [gradient_symbols], matrix))

    def rigid_positions(self, rest):
        """R X: the rest configuration turned by this problem's rotation, itself where there is none.

        The configuration a corotational force field reads as undeformed, so the solve starts there.
        """
        if self.rotation is None:
            return rest
        rest = np.asarray(rest)
        # R(X) X
        return np.einsum('...ij,...j->...i', self.rotation_field(rest), rest)

    def local_gradient(self, gradients, points):
        """Displacement gradients at `points` in the frame the material law is applied in:
        R^T (I + G) - I, R the frame rotation there.

        Recovers the unrotated gradient, so the norms measure strain rather than the rotation.
        """
        if self.frame_rotation is None:
            return gradients
        identity = np.eye(len(self.coordinates))
        # R^T (I + G) - I
        return np.einsum('...ji,...jk->...ik', self.frame_rotation(points),
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

    def _derive_fields_symbolically(self, fpk_stress, cauchy_stress, frame_rotation):
        """Computes the source term symbolically, as minus the divergence of the stress P. Compiles it, the
        stresses P and sigma, and the frame rotation R, all sympy expressions of x, into numpy fields.
        """
        dimensions = len(self.coordinates)
        position, matrix = list(self.coordinates), (dimensions, dimensions)
        # P
        self.fpk_stress_expression = fpk_stress
        # s_i = -sum_j dP_ij/dx_j
        self.source_expression = -sp.Matrix([sum(sp.diff(fpk_stress[i, j], self.coordinates[j])
                                                  for j in range(dimensions))
                                              for i in range(dimensions)])

        # P(x)
        self.fpk_stress = self._compile(fpk_stress, [position], matrix)
        # sigma(x)
        self.cauchy_stress = self._compile(cauchy_stress, [position], matrix)
        # s(x)
        self.source = self._compile(self.source_expression, [position], (dimensions,))
        if frame_rotation is not None:
            # R(x)
            self.frame_rotation = self._compile(frame_rotation, [position], matrix)
        else:
            self.frame_rotation = None

    def _derive_fields_numerically(self, gradient, fpk_stress_law, cauchy_stress_law):
        """Computes the frame rotation, the stresses and the source term numerically, at each point x: the
        frame rotation R_p is the polar rotation of F = I + grad u, from its SVD; the stresses P and sigma
        apply the stress laws to R_p^T F - I; the source term is minus the divergence of P, by fourth-order
        central differences.
        """
        dimensions = len(self.coordinates)
        identity = np.eye(dimensions)
        # F(x) = I + grad u
        deformation = self._compile(sp.eye(dimensions) + gradient, [list(self.coordinates)], identity.shape)
        self.fpk_stress_expression = None
        self.source_expression = None

        def rotation_and_local_gradient(points):
            deformation_values = deformation(points)
            # F = W S V^T,  R_p = W V^T
            left, _, right = np.linalg.svd(deformation_values)
            rotation = left @ right
            # R_p,  R_p^T F - I
            return rotation, np.swapaxes(rotation, -1, -2) @ deformation_values - identity

        def frame_rotation(points):
            # R_p(x)
            return rotation_and_local_gradient(points)[0]

        def fpk_stress(points):
            rotation, local_gradient = rotation_and_local_gradient(points)
            # P = R_p P_law(R_p^T F - I)
            return rotation @ fpk_stress_law(local_gradient)

        def cauchy_stress(points):
            rotation, local_gradient = rotation_and_local_gradient(points)
            # sigma = R_p sigma_law(R_p^T F - I) R_p^T
            return rotation @ cauchy_stress_law(local_gradient) @ np.swapaxes(rotation, -1, -2)

        def source(points):
            # h: balances truncation, O(h^4), against round-off, O(eps / h), on fields of unit length scale.
            step_size = 3e-4
            points = np.asarray(points, dtype=float)
            divergence = np.zeros(points.shape)
            for j, step in enumerate(step_size * identity):
                # dP/dx_j = [8 (P(x + h e_j) - P(x - h e_j)) - (P(x + 2h e_j) - P(x - 2h e_j))] / 12h
                derivative = (8 * (fpk_stress(points + step) - fpk_stress(points - step))
                              - (fpk_stress(points + 2 * step) - fpk_stress(points - 2 * step))
                              ) / (12 * step_size)
                # sum_j dP_ij/dx_j
                divergence += derivative[..., :, j]
            # s_i = -sum_j dP_ij/dx_j
            return -divergence

        # P(x), sigma(x), s(x)
        self.fpk_stress, self.cauchy_stress, self.source = fpk_stress, cauchy_stress, source
        # R_p(x)
        self.frame_rotation = frame_rotation

    def _compile(self, expression, arguments, shape):
        """Compile `expression`, entries in the symbols of `arguments`, into a numpy function.

        `arguments` are tensors of symbols: the position for a field, the gradient for a material law.
        The result takes one array per argument, shaped (..., *argument shape), a single value or a
        whole batch, and returns the expression's value for each, shaped (*batch, *shape).
        """
        arguments = [np.array(argument, dtype=object) for argument in arguments]
        symbols = [symbol for argument in arguments for symbol in argument.flat]
        entries = [sp.lambdify(symbols, entry.xreplace(self.parameter_values), "numpy") for entry in expression]

        def compiled(*values):
            values = [np.asarray(value, dtype=float) for value in values]
            batch = np.broadcast_shapes(*(value.shape[:value.ndim - argument.ndim]
                                          for value, argument in zip(values, arguments)))
            columns = [value[(...,) + index]
                       for value, argument in zip(values, arguments) for index in np.ndindex(argument.shape)]
            result = np.empty(batch + shape, dtype=float)
            for index, entry in zip(np.ndindex(*shape), entries):
                result[(...,) + index] = entry(*columns)
            return result

        return compiled

