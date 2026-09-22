"""Post-init SOFA controllers for the VnV suite."""

import numpy as np

import Sofa
import Sofa.Core

from .conventions import ELEMENTS


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

        self.density = self.node.addObject('NodalSourceDensity', name='sourceDensity',
                                           template=self.vec_type)
        # One call for the whole mesh: the manufactured fields evaluate over an array of points.
        self.density.property.value = self.source(self.dofs.rest_position.array())

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
                           topology='@topology', quadratureDegree=self.quadrature_degree,
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


class RegionClamp(Sofa.Core.Controller):
    """Builds one PartialFixedProjectiveConstraint per boundary region, then applies Dirichlet
    boundary conditions once the simulation has initialized.

    For each region, the nodes belonging to it are found and handed to its constraint as the indices
    it fixes. If a prescribed displacement is given, those nodes are also moved from rest to rest +
    u in the constraint's fixed directions before the constraint clamps them there. rest_position
    itself is never changed.
    """

    def __init__(self, geometry, dofs, node, prescribe_displacement_on, vec_type,
                 displacement=None, rigid_positions=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.geometry = geometry
        self.dofs = dofs
        self.displacement = displacement
        # rest -> the configuration the solve starts from, itself when the problem is not rotated.
        self.rigid_positions = rigid_positions or (lambda rest: rest)

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
            # Move the initial positions after applying a rigid frame rotation.
            # TODO Placing the whole mesh does not belong to a boundary clamp. Done here for now.
            pos[:] = self.rigid_positions(rest_positions)

            for region, fixed_directions, constraint in self.region_constraints:
                indices = self.geometry.region_indices(region, rest_positions)

                if self.displacement is not None and indices:
                    displacement = self.displacement(rest_positions[indices])

                    for axis, fixed in enumerate(fixed_directions):
                        if fixed:
                            pos[indices, axis] = rest_positions[indices, axis] + displacement[:, axis]

                constraint.indices.value = indices


class Excitation(Sofa.Core.Controller):
    """Brings a mesh's loads up to their stated values over `steps_count` equal increments."""

    def __init__(self, steps_count, source_term_controllers=(), traction_controllers=(),
                 clamp_controllers=(), *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.steps_count = steps_count
        self.step = 0
        self.source_term_controllers = source_term_controllers
        self.traction_controllers = traction_controllers
        self.clamp_controllers = clamp_controllers
        # Pairs of Data each load controller has, e.g.
        # - The body force's sourceDensity.property,
        # - The traction's stress.property or a point load's forces
        # and their values at full excitation.
        self.loads_at_full_excitation = []

    def onSimulationInitDoneEvent(self, event):
        if self.steps_count == 1:
            return

        load_data = [controller.density.property for controller in self.source_term_controllers]
        for controller in self.traction_controllers:
            if controller.nodal_stress is not None:
                load_data.append(controller.nodal_stress.property)
            elif controller.point_load is not None:
                load_data.append(controller.point_load.forces)

        # Read once the loads have written themselves: what full excitation means is their word.
        self.loads_at_full_excitation = [(data, np.array(data.value)) for data in load_data]

    def onAnimateBeginEvent(self, event):
        """Applies the ramp: every load is brought to this step's share of its full value."""
        # The loads already hold their full values; rewriting them would re-integrate for nothing.
        if self.steps_count == 1:
            return

        self.step += 1
        factor = min(1.0, self.step / self.steps_count)

        for data, at_full_excitation in self.loads_at_full_excitation:
            data.value = factor * at_full_excitation

        # A Dirichlet condition prescribes a position rather than a magnitude, so its nodes are
        # walked out from where the solve started instead of being scaled.
        for controller in self.clamp_controllers:
            if controller.displacement is None:
                continue

            rest_positions = controller.dofs.rest_position.array()
            start_positions = controller.rigid_positions(rest_positions)
            with controller.dofs.position.writeableArray() as pos:
                for _, fixed_directions, constraint in controller.region_constraints:
                    indices = constraint.indices.value
                    if len(indices) == 0:
                        continue

                    rest, start = rest_positions[indices], start_positions[indices]
                    target = rest + controller.displacement(rest)
                    for axis, fixed in enumerate(fixed_directions):
                        if fixed:
                            pos[indices, axis] = (start[:, axis]
                                                  + factor * (target[:, axis] - start[:, axis]))
