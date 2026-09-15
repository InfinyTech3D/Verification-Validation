"""Post-init SOFA controllers for the VnV suite."""

import Sofa
import Sofa.Core


def region_indices(geometry, region, nodes):
    """Node indices whose coordinate satisfies the geometry's named-region predicate."""
    predicate = geometry.boundary_regions[region]
    indices = []
    for index, point in enumerate(nodes):
        if predicate(point):
            indices.append(index)
    return indices


class RegionClamp(Sofa.Core.Controller):
    """Builds one PartialFixedProjectiveConstraint per boundary region, then applies
    Dirichlet boundary conditions once the simulation has initialized.

    For each region, the nodes belonging to it are found and handed to its constraint as
    the indices it fixes. If a prescribed displacement is given, those nodes are also moved from
    rest to rest + u in the constraint's fixed directions before the constraint clamps them there.
    rest_position itself is never changed.
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
