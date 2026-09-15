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
    """Applies Dirichlet boundary conditions once the simulation has initialized.

    - `groups` is a list of (constraint, regions, mask) tuples.
    - `constraint` is a PartialFixedProjectiveConstraint already built elsewhere.
    - `regions` are the geometry's named boundary regions whose nodes belong to it.
    - `mask` says which spatial directions that constraint fixes, one entry per
      axis; 1 fixes that axis, 0 leaves it free. For example:

        (some_constraint, ("left", "bottom"), [1, 0])

    means every node in the "left" or "bottom" regions has its x-direction fixed by
    `some_constraint`, while its y-direction stays free.

    For each group, the nodes belonging to its regions are found and handed to the constraint as
    the indices it fixes. If a prescribed displacement is given, those nodes are also moved from
    rest to rest + u in the constraint's fixed directions before the constraint clamps them there.
    rest_position itself is never changed.
    """

    def __init__(self, geometry, dofs, groups, displacement=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.geometry = geometry
        self.dofs = dofs
        self.groups = groups          # list of (constraint, regions, mask)
        self.displacement = displacement

    def onSimulationInitDoneEvent(self, event):
        rest_positions = self.dofs.rest_position.array()

        with self.dofs.position.writeableArray() as pos:
            for constraint, regions, mask in self.groups:
                indices = set()
                for region in regions:
                    indices.update(region_indices(self.geometry, region, rest_positions))
                indices = sorted(indices)

                if self.displacement is not None and indices:
                    # Prescribed displacement: move each fixed direction to rest + u; rest_position is untouched.
                    displacement = self.displacement(rest_positions[indices])

                    for axis, fixed in enumerate(mask):
                        if fixed:
                            pos[indices, axis] = rest_positions[indices, axis] + displacement[:, axis]

                constraint.indices.value = indices
