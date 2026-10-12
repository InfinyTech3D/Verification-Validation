"""Post-init SOFA controllers loading a mesh with a manufactured problem."""

import numpy as np

import Sofa
import Sofa.Core

from common.sofa.conventions import ELEMENTS


class ApplyManufacturedSourceTerm(Sofa.Core.Controller):
    """Loads a mesh with the body force a manufactured solution puts on the right-hand side."""

    def __init__(self, node, dofs, source, vec_type, element, source_quadrature_degree,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.node = node
        self.dofs = dofs
        self.source = source
        self.vec_type = vec_type
        self.element = element
        self.source_quadrature_degree = source_quadrature_degree

    def init(self):
        template = f'{self.vec_type},{ELEMENTS[self.element].cpp}'

        self.density = self.node.addObject('NodalSourceDensity', name='sourceDensity',
                                           template=self.vec_type)
        # One call for the whole mesh: the manufactured fields evaluate over an array of points.
        self.density.property.value = self.source(self.dofs.rest_position.array())

        self.node.addObject('VectorSourceTerm', name='bodyForce', template=template,
                            sourceDensity='@sourceDensity')
        self.node.addObject('FEMSourceTermIntegrator', name='bodySource', template=template,
                            topology='@topology', quadratureDegree=self.source_quadrature_degree,
                            constantSources='@bodyForce')


class ApplyManufacturedTraction(Sofa.Core.Controller):
    """Loads the whole boundary of a mesh with the traction a manufactured solution exerts on it."""

    def __init__(self, geometry, node, dofs, stress, vec_type, element, spatial_dimensions,
                 source_quadrature_degree, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.geometry = geometry
        self.node = node
        self.dofs = dofs
        self.stress = stress
        self.vec_type = vec_type
        self.element = element
        self.spatial_dimensions = spatial_dimensions
        self.source_quadrature_degree = source_quadrature_degree
        self.nodal_stress = None
        # Point load is used for the particular case of Traction BC in 1D
        self.point_load = None

    def init(self):
        rest_positions = self.dofs.rest_position.array()
        mappings = ELEMENTS[self.element].boundary_mappings

        # Only Edges have no Volume2Boundary mapping. Simply prescribe a nodal force.
        if not mappings:
            self.add_point_load(rest_positions)
            return

        boundary_kind = ELEMENTS[mappings[-1][0]]
        if boundary_kind.dim + 1 != self.spatial_dimensions:
            # TODO StressSourceTerm does not support elements of codimension 2. So we cannot apply
            # traction BCs on Quad/Triangular meshes in 3D. The Edge normals are arbitrary.
            return

        # Goes from the volume to the boundary elements, applying one topological mapping at a time.
        boundary = self.node
        for kind, mapping in mappings:
            element_kind = ELEMENTS[kind]
            boundary = boundary.addChild(kind)
            boundary.addObject(element_kind.container, name='topology')
            boundary.addObject(element_kind.container.replace('Container', 'Modifier'))
            boundary.addObject(mapping, input='@../topology', output='@topology')

        # Apply the nodal stress load on the boundary.
        template = f'{self.vec_type},{boundary_kind.cpp}'
        # The manufactured stress at every node, converted to the layout NodalStress takes.
        stress_values = self.stress(rest_positions)
        rows, columns = np.tril_indices(stress_values.shape[-1])
        self.nodal_stress = boundary.addObject('NodalStress', name='stress', template=self.vec_type)
        self.nodal_stress.property.value = np.ascontiguousarray(stress_values[:, rows, columns])

        boundary.addObject('StressSourceTerm', name='traction', template=template, stress='@stress')
        boundary.addObject('FEMSourceTermIntegrator', name='tractionSource', template=template,
                           topology='@topology', quadratureDegree=self.source_quadrature_degree,
                           constantSources='@traction')

    def add_point_load(self, rest_positions):
        """The nodal traction on every boundary region, its normal pointing away from the mesh."""
        centre = rest_positions.mean(axis=0)
        indices, tractions = [], []
        for region in self.geometry.region_names:
            nodes = self.geometry.region_indices(region, rest_positions)
            outward = rest_positions[nodes] - centre
            outward /= np.linalg.norm(outward, axis=-1, keepdims=True)

            indices += nodes
            tractions.append(np.einsum('nij,nj->ni',
                                       self.stress(rest_positions[nodes]), outward))

        self.point_load = self.node.addObject('ConstantForceField', name='traction',
                                              template=self.vec_type, indices=indices,
                                              forces=np.concatenate(tractions))
