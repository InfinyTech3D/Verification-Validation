"""Dim-generic mesh quadrature: the fields sampled on it, and the norms integrated over them."""

import numpy as np

import Sofa.SofaFEM


class MeshQuadrature:
    """The reference->physical mapping of a whole mesh, evaluated once and shared by every norm.

    Array layout throughout: `e` element, `q` quadrature point, `a` node within an element,
    `c` field component, `d` spatial direction.
    """

    def __init__(self, nodes, node_indices, element, degree):
        self.nodes = np.asarray(nodes)
        # node_indices[e] = the mesh-node indices that form element e.
        self.node_indices = np.asarray(node_indices)
        dim = self.nodes.shape[1]

        # Reference-space data is identical for every element of this type, so fetch it once.
        reference_weights, self.shape_functions, reference_gradients = \
            Sofa.SofaFEM.quadrature_data(element, dim, degree)
        self.shape_function_gradients, measures = Sofa.SofaFEM.element_mapping_batch(
            element, self.nodes, self.node_indices, reference_gradients)

        self.points = np.einsum('qa,ead->eqd', self.shape_functions, self.nodes[self.node_indices])
        # The dOmega of every integral: the reference weight carried into physical space.
        self.weights = reference_weights * measures

    def integrate(self, values):
        """Integrate a scalar given at every (element, quadrature point)."""
        return float(np.sum(values * self.weights))

    def values(self, field):
        """A nodal field at every quadrature point: (e, q, c)."""
        return np.einsum('qa,eac->eqc', self.shape_functions, np.asarray(field)[self.node_indices])

    def grads(self, field):
        """Gradient of a nodal field: (e, q, c, d) = d(u_c)/dx_d."""
        return np.einsum('eac,eqad->eqcd',
                         np.asarray(field)[self.node_indices], self.shape_function_gradients)

    def sample(self, field):
        """A manufactured field at every quadrature point: the one seam where it meets this mesh."""
        return field(self.points)

    def L2Norm(self, values):
        """L2 norm of a vector field given at every quadrature point: (e, q, c) -> scalar."""
        return float(np.sqrt(self.integrate(np.sum(values * values, axis=-1))))

    def FrobeniusNorm(self, tensors):
        """L2 norm of a tensor field, Frobenius over its last two axes: (e, q, c, d) -> scalar."""
        return float(np.sqrt(self.integrate(np.sum(tensors * tensors, axis=(-2, -1)))))

    def energy(self, gradients, energy_density):
        """Elastic energy of a displacement gradient field: integral over the mesh of psi."""
        return self.integrate(energy_density(gradients))
