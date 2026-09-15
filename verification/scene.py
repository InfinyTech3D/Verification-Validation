"""Generic SOFA scene assembly."""

from common.sofa import SofaScene
from common.sofa.prefabs import ElasticBeam


class Scene(SofaScene):
    """Assembles a SOFA scene: the elastic beam, then boundary conditions, then solvers."""

    plugins = [
        "Sofa.Component.SolidMechanics.FEM.Elastic",
        "Sofa.Component.Constraint.Projective",
        "Sofa.Component.Engine.Select",
        "Sofa.Component.Mapping.Linear",
        "Sofa.Component.MechanicalLoad",
        "Sofa.Component.LinearSolver.Direct",
        "Sofa.Component.LinearSolver.Iterative",
        "Sofa.Component.LinearSolver.Preconditioner",
        "Sofa.Component.LinearSystem",
        "Sofa.Component.ODESolver.Backward",
        "Sofa.Component.StateContainer",
        "Sofa.Component.Topology.Container.Grid",
        "Sofa.Component.Topology.Container.Dynamic",
        "Sofa.Component.Topology.Mapping",
    ]

    def __init__(self, geometry, material, force_field, element, resolution, solvers):
        super().__init__()
        self.geometry = geometry
        self.material = material
        self.force_field = force_field
        self.element = element
        self.resolution = resolution
        self.solvers = solvers

    def apply_bcs(self, beam):
        """Boundary conditions — filled by the verification or validation suite."""

    def body(self, root):
        root.addObject('DefaultAnimationLoop')
        resolution = [self.resolution[i] if i < len(self.resolution) else 1 for i in range(3)]
        # The beam is complete when it is added: the deck blocks it needs go in with it, since a
        # prefab runs its own init() from inside its constructor.
        beam = root.addChild(ElasticBeam(name='beam',
                                         extents=self.geometry.extents,
                                         resolution=resolution,
                                         spatialDimensions=self.geometry.spatial_dimensions,
                                         element=self.element,
                                         spec={'material': self.material,
                                               'forceField': self.force_field,
                                               'solvers': self.solvers}))
        self.apply_bcs(beam)
        return beam
