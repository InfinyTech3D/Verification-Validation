"""Post-init SOFA controllers for the VnV suite."""

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

    def __init__(self, node, dofs, stress, vec_type, element, quadrature_degree, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.node = node
        self.dofs = dofs
        self.stress = stress
        self.vec_type = vec_type
        self.element = element
        self.quadrature_degree = quadrature_degree

    def init(self):
        mappings = ELEMENTS[self.element].boundary_mappings
        if not mappings:
            return  # the facets of an edge are points, which no traction can be integrated over

        boundary = self.node.addChild('neumann')

        # Walk down to the boundary elements, one topological mapping at a time.
        topology = '@../topology'
        for kind, mapping in mappings:
            element_kind = ELEMENTS[kind]
            boundary.addObject(element_kind.container, name=kind)
            boundary.addObject(element_kind.container.replace('Container', 'Modifier'))
            boundary.addObject(mapping, input=topology, output=f'@{kind}')
            topology = f'@{kind}'

        # The boundary node carries no state of its own, so the load lands on the dofs above it and
        # the nodal stress is indexed by those same nodes.
        rest_positions = self.dofs.rest_position.array()
        stress = boundary.addObject('NodalStress', name='stress', template=self.vec_type)
        # reshape tensor to adjust to NodalStress input format.
        stress.property.value = self.stress(rest_positions).reshape(len(rest_positions), -1)

        template = f'{self.vec_type},{ELEMENTS[mappings[-1][0]].cpp}'
        boundary.addObject('StressSourceTerm', name='traction', template=template, stress='@stress')
        boundary.addObject('FEMSourceTermIntegrator', name='tractionSource', template=template,
                           topology=topology, quadratureDegree=self.quadrature_degree,
                           constantSources='@traction')


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
