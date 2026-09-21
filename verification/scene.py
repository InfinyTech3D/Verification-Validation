"""Verification suite: an MMS scene whose BCs come from a manufactured solution."""

from common.sofa import SofaScene
from common.sofa.conventions import VEC_DIM
from common.sofa.controllers import (ApplyManufacturedSourceTerm, ApplyManufacturedTraction,
                                     RegionClamp)
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

    def __init__(self, deck, cells):
        """One mesh of the deck, `cells` counting its cells per axis."""
        super().__init__()
        self.cells = cells
        self.geometry = deck.geometry
        self.material = deck.material
        self.force_field = deck.force_field
        self.element = deck.element
        self.solvers = deck.solvers
        self.manufactured_problem = deck.manufactured_problem
        self.source_quadrature_degree = deck.source_quadrature_degree

    @property
    def grid_resolution(self):
        """The grid resolution of `RegularGridTopology`, which is its `n` Data."""
        return [count + 1 for count in self.cells]

    def body(self, root):
        root.addObject('DefaultAnimationLoop')

        prefab_cls = PREFAB_BY_GEOMETRY[type(self.geometry)]
        configs = {'material': self.material, 'forceField': self.force_field, 'solvers': self.solvers}
        prefab = root.addChild(prefab_cls(name=prefab_cls.__name__, geometry=self.geometry,
                                          grid_resolution=self.grid_resolution, element=self.element, configs=configs))
        mechanical = prefab.mechanical_node()

        dim, spatial_dimensions = self.geometry.dim, self.geometry.spatial_dimensions
        VEC = VEC_DIM[spatial_dimensions]

        # Load the mesh with the body force the manufactured solution puts on the rhs
        mechanical.addObject(ApplyManufacturedSourceTerm(
            node=mechanical, dofs=mechanical.dofs, source=self.manufactured_problem.source,
            vec_type=VEC, element=self.element,
            quadrature_degree=self.source_quadrature_degree, name='bodyForceCtrl'))

        # Load the whole boundary with the traction that solution exerts there
        mechanical.addObject(ApplyManufacturedTraction(
            geometry=self.geometry, node=mechanical, dofs=mechanical.dofs,
            stress=self.manufactured_problem.stress,
            vec_type=VEC, element=self.element, spatial_dimensions=spatial_dimensions,
            quadrature_degree=self.source_quadrature_degree, name='tractionCtrl'))

        # Clamp the mesh dofs by region as determined by the manufactured solution BCs
        mechanical.addObject(RegionClamp(geometry=self.geometry,
                                         dofs=mechanical.dofs, node=mechanical,
                                         prescribe_displacement_on=self.manufactured_problem.prescribe_displacement_on,
                                         vec_type=VEC, displacement=self.manufactured_problem.u,
                                         name='clampCtrl'))

        # For an embedded mesh (dim < spatial_dimensions) the out-of-plane dimensions are fixed.
        if dim < spatial_dimensions:
            mechanical.addObject('PartialFixedProjectiveConstraint', name='outOfPlane', template=VEC,
                           fixAll=True,
                           fixedDirections=[0] * dim + [1] * (spatial_dimensions - dim)) # e.g. [0, 0] + [1] -> [0, 0, 1]

        return prefab
