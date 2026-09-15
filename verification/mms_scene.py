"""Verification suite: an MMS scene whose BCs come from a manufactured solution."""

from common.sofa import SofaScene
from common.sofa.conventions import VEC_DIM
from common.sofa.controllers import RegionClamp
from common.sofa.prefabs import PREFAB_BY_GEOMETRY


class MMSScene(SofaScene):
    """A scene set up to solve a modified PDE. Boundary conditions and a body-force load are
    chosen so that the solution of that modified problem is a pre-manufactured one."""

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

    def __init__(self, deck, resolution):
        super().__init__()
        self.geometry = deck.geometry
        self.material = deck.material
        self.force_field = deck.force_field
        self.element = deck.element
        self.resolution = resolution
        self.solvers = deck.solvers
        self.mms = deck.solution
        self.source_quadrature_degree = deck.source_quadrature_degree

    def body(self, root):
        root.addObject('DefaultAnimationLoop')

        prefab_cls = PREFAB_BY_GEOMETRY[type(self.geometry)]
        spec = {'material': self.material, 'forceField': self.force_field, 'solvers': self.solvers}
        prefab = root.addChild(prefab_cls(name=prefab_cls.__name__, geometry=self.geometry,
                                          resolution=self.resolution, element=self.element, spec=spec))
        mechanical = prefab.mechanical_node()

        dim, spatial_dimensions = self.geometry.dim, self.geometry.spatial_dimensions
        VEC = VEC_DIM[spatial_dimensions]

        # Clamp the mesh dofs by region as determined by the manufactured solution BCs
        mechanical.addObject(RegionClamp(geometry=self.geometry,
                                         dofs=mechanical.dofs, node=mechanical,
                                         prescribe_displacement_on=self.mms.prescribe_displacement_on,
                                         vec_type=VEC, displacement=self.mms.u,
                                         name='clampCtrl'))

        # For an embedded mesh (dim < spatial_dimensions) the out-of-plane dimensions are fixed.
        if dim < spatial_dimensions:
            mechanical.addObject('PartialFixedProjectiveConstraint', name='outOfPlane', template=VEC,
                           fixAll=True,
                           fixedDirections=[0] * dim + [1] * (spatial_dimensions - dim)) # e.g. [0, 0] + [1] -> [0, 0, 1]

        return prefab
