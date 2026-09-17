"""Post-init SOFA controllers for the VnV suite."""

import numpy as np

import Sofa
import Sofa.Core

from .conventions import ELEMENTS


def region_indices(geometry, region, nodes):
    """Node indices whose coordinate satisfies the geometry's named-region predicate."""
    predicate = geometry.boundary_regions[region]
    indices = []
    for index, point in enumerate(nodes):
        if predicate(point):
            indices.append(index)
    return indices


class ApplyManufacturedSourceTerm(Sofa.Core.Controller):
    """Loads a mesh with the body force a manufactured solution puts on the right-hand side."""

    def __init__(self, node, dofs, source, vec_type, element, quadrature_degree,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.node = node
        self.dofs = dofs
        self.source = source
        self.vec_type = vec_type
        self.element = element
        self.quadrature_degree = quadrature_degree

    def init(self):
        template = f'{self.vec_type},{ELEMENTS[self.element].cpp}'

        density = self.node.addObject('NodalSourceDensity', name='sourceDensity',
                                      template=self.vec_type)
        # One call for the whole mesh: the manufactured fields evaluate over an array of points.
        density.property.value = self.source(self.dofs.rest_position.array())

        self.node.addObject('VectorSourceTerm', name='bodyForce', template=template,
                            sourceDensity='@sourceDensity')
        self.node.addObject('FEMSourceTermIntegrator', name='bodySource', template=template,
                            topology='@topology', quadratureDegree=self.quadrature_degree,
                            constantSources='@bodyForce')


class ApplyManufacturedTraction(Sofa.Core.Controller):
    """Loads the whole boundary of a mesh with the traction a manufactured solution exerts on it."""

    def __init__(self, geometry, node, dofs, stress, vec_type, element, spatial_dimensions,
                 quadrature_degree, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.geometry = geometry
        self.node = node
        self.dofs = dofs
        self.stress = stress
        self.vec_type = vec_type
        self.element = element
        self.spatial_dimensions = spatial_dimensions
        self.quadrature_degree = quadrature_degree

    def init(self):
        rest_positions = self.dofs.rest_position.array()
        mappings = ELEMENTS[self.element].boundary_mappings

        # Only Edges have no Volume2Boundary mapping. Simply prescribe a nodal force.
        if not mappings:
            self.add_point_load(rest_positions)
            return

        boundary_kind = ELEMENTS[mappings[-1][0]]
        if boundary_kind.dim + 1 != self.spatial_dimensions:
            # TODO StressSourceTerm does not support elements of codimention 2. So we cannot apply
            # traction BCs on Quad/Triangluar meshes in 3D. The Edge normals are arbitrary.
            return

        # Goes from the volume to the boundary elements, applying one topological mapping at a time.
        boundary = self.node
        for kind, mapping, data in mappings:
            element_kind = ELEMENTS[kind]
            boundary = boundary.addChild(kind)
            boundary.addObject(element_kind.container, name='topology')
            boundary.addObject(element_kind.container.replace('Container', 'Modifier'))
            boundary.addObject(mapping, input='@../topology', output='@topology', **data)

        # Apply the nodal stress load on the boundary.
        template = f'{self.vec_type},{boundary_kind.cpp}'
        # The manufactured stress at every node, converted to the layout NodalStress takes.
        stress = self.stress(rest_positions)
        rows, columns = np.tril_indices(stress.shape[-1])
        nodal_stress = boundary.addObject('NodalStress', name='stress', template=self.vec_type)
        nodal_stress.property.value = np.ascontiguousarray(stress[:, rows, columns])

        boundary.addObject('StressSourceTerm', name='traction', template=template, stress='@stress')
        boundary.addObject('FEMSourceTermIntegrator', name='tractionSource', template=template,
                           topology='@topology', quadratureDegree=self.quadrature_degree,
                           constantSources='@traction')

    def add_point_load(self, rest_positions):
        """The nodal traction on every boundary region, its normal pointing away from the mesh."""
        centre = rest_positions.mean(axis=0)
        indices, tractions = [], []
        for region in self.geometry.boundary_regions:
            nodes = region_indices(self.geometry, region, rest_positions)
            outward = rest_positions[nodes] - centre
            outward /= np.linalg.norm(outward, axis=-1, keepdims=True)

            indices += nodes
            tractions.append(np.einsum('nij,nj->ni',
                                       self.stress(rest_positions[nodes]), outward))

        self.node.addObject('ConstantForceField', name='traction', template=self.vec_type,
                            indices=indices, forces=np.concatenate(tractions))


class RegionClamp(Sofa.Core.Controller):
    """Builds one PartialFixedProjectiveConstraint per boundary region, then applies Dirichlet
    boundary conditions once the simulation has initialized.

    For each region, the nodes belonging to it are found and handed to its constraint as the indices
    it fixes. If a prescribed displacement is given, those nodes are also moved from rest to rest +
    u in the constraint's fixed directions before the constraint clamps them there. rest_position
    itself is never changed.
    """

    def __init__(self, geometry, dofs, node, prescribe_displacement_on, vec_type,
                 displacement=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.geometry = geometry
        self.dofs = dofs
        self.displacement = displacement

        # Fix dofs partially as determined by MMS `prescribe_displacement_on`
        self.region_constraints = []
        for region, fixed_directions in prescribe_displacement_on.items():
            # Pad the array with zeroes for embedded dimensions: e.g. [1, 0] -> [1, 0, 0]
            padded = tuple(fixed_directions) + (0,) * (geometry.spatial_dimensions - len(fixed_directions))
            constraint = node.addObject('PartialFixedProjectiveConstraint', name='clamp' + region,
                                  template=vec_type, fixedDirections=list(padded))
            self.region_constraints.append((region, padded, constraint))

    def onSimulationInitDoneEvent(self, event):
        rest_positions = self.dofs.rest_position.array()

        # Apply non-zero Dirichlet BC: Move each fixed direction to rest + u; rest_position is untouched.
        with self.dofs.position.writeableArray() as pos:
            for region, fixed_directions, constraint in self.region_constraints:
                indices = region_indices(self.geometry, region, rest_positions)

                if self.displacement is not None and indices:
                    displacement = self.displacement(rest_positions[indices])

                    for axis, fixed in enumerate(fixed_directions):
                        if fixed:
                            pos[indices, axis] = rest_positions[indices, axis] + displacement[:, axis]

                constraint.indices.value = indices
