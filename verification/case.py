"""The case as one object: parsed, validated, and resolved into a config dictionary that a run needs."""

import json
import pathlib

from common.geometry import Grid, Bar1D, Beam2D, Beam3D, Tire, Tire2D, Tire3D
from common.sofa.conventions import ELEMENTS

from . import manufactured
from .manufactured import ManufacturedProblem, rotation_matrix
from .manufactured.materials import LinearElastic, MooneyRivlin, NeoHookean, Ogden, SaintVenantKirchhoff
from .study.metrics import METRICS

CASES_ROOT = pathlib.Path(__file__).parent / "cases"

# case "geometry.type" -> geometry class, keyed by the class's own name.
GEOMETRIES = {cls.__name__: cls for cls in (Grid, Bar1D, Beam2D, Beam3D,
                                            Tire, Tire2D, Tire3D)}

# case "material.type" -> material class, keyed by the class's own name.
MATERIALS = {cls.__name__: cls for cls in (LinearElastic, MooneyRivlin, NeoHookean, Ogden,
                                          SaintVenantKirchhoff)}


class _ManufacturedSolutionRegistry:
    """(topological dim of the geometry, case "solution.type") -> ManufacturedSolution class.

    Looked up by naming convention: a case's solution type "Quadratic" at dim 2 resolves to
    manufactured.Quadratic2D. The dimension is the PDE's own, not the space it is solved in.
    """

    def __getitem__(self, key):
        dim, type_name = key
        class_name = f"{type_name}{dim}D"
        try:
            return getattr(manufactured, class_name)
        except AttributeError:
            raise KeyError(f"no manufactured solution {class_name!r} for dim={dim}, "
                           f"solution type={type_name!r}")


MANUFACTURED_SOLUTIONS = _ManufacturedSolutionRegistry()

# Required keys for all cases. A missing key is an omission, an unknown key is a typo.
REQUIRED = {"geometry", "element", "solution", "material", "forceField",
            "quadratureDegree", "sourceQuadratureDegree", "solvers", "mesh",
            "asymptoticTolerance", "expectedOrder", "expectedOrderTolerance", "relativeNoiseFloor"}

OPTIONAL = {
        "excitationSteps" # equal increments the loads are applied in; absent means one.
        , "tractionOn"    # {region: directions} handed to the manufactured traction; the rest stay Dirichlet.
        , "rotation"      # a rigid rotation of the whole problem; its absence is the statement.
        , "gridMapping"   # Data for the mapping meshing the grid, where the element kind needs one.
        , "compareAgainst"      # case whose recorded errors this one must reproduce, level for level.
        , "agreementTolerance"  # the largest relative difference from that record still counted as agreement.
}

MESH_KEYS = {"cells", "levels", "refinementRatio"}

ROTATION_KEYS = {"angleDegrees", "axis"}

# Rotation axis -> the coordinate axis it leaves fixed.
ROTATION_AXES = {"x": 0, "y": 1, "z": 2}



def _validate(name, config):
    """Every required key present, nothing unknown, and one expectation per metric.

    `solvers` and `forceField` pass their own keys through to SOFA Data verbatim, so this module
    cannot know what's legal inside them and only checks their presence/shape, not their contents.
    Every problem is collected before raising, so a case is corrected in one pass, not one key per run.
    """
    problems = []

    # Top-level keys: exact match against REQUIRED.
    missing = REQUIRED - set(config)
    if missing:
        problems.append(f"missing {sorted(missing)}")
    unknown = set(config) - REQUIRED - OPTIONAL
    if unknown:
        problems.append(f"unknown {sorted(unknown)}")

    steps_count = config.get("excitationSteps", 1)
    if not isinstance(steps_count, int) or steps_count < 1:
        problems.append(f"excitationSteps must be a positive integer, got {steps_count!r}")

    # A grid-native element kind is meshed by no mapping, so its Data would go nowhere.
    element = config.get("element")
    if "gridMapping" in config and element in ELEMENTS and ELEMENTS[element].grid_mapping is None:
        problems.append(f"gridMapping is stated but {element!r} is meshed straight from the grid")

    # mesh: exact match against MESH_KEYS.
    if "mesh" in config and set(config["mesh"]) != MESH_KEYS:
        problems.append(f"mesh must state exactly {sorted(MESH_KEYS)}, got {sorted(config['mesh'])}")

    # solution: just needs a type and an amplitude; the rest is that solution's own business.
    solution_config = config.get("solution", {})
    absent = {"type", "amplitude"} - set(solution_config)
    if absent:
        problems.append(f"solution must state {sorted(absent)}")

    # material: must name a registered type.
    material = config.get("material", {})
    if "type" not in material:
        problems.append(f"material must state a type, one of {sorted(MATERIALS)}")
    elif material["type"] not in MATERIALS:
        problems.append(f"unknown material {material['type']!r}, expected one of {sorted(MATERIALS)}")

    # forceField: must be an object stating a type.
    force_field = config.get("forceField")
    if not isinstance(force_field, dict):
        problems.append(f"forceField must be an object stating a type, got {force_field!r}")
        force_field = {}
    elif "type" not in force_field:
        problems.append("forceField must state a type")
    # forceField must not restate what the prefab already owns: name/template/topology follow
    # from element and the mesh, so a case stating either could contradict what it already stated.
    owned = {"name", "template", "topology"} & set(force_field)
    if owned:
        problems.append(f"forceField must not state {sorted(owned)}: the prefab sets them")
    # forceField must not restate a constitutive parameter that belongs in material, so the
    # manufactured source and the forceField under test can't drift apart.
    clash = (set(force_field) & set(material)) - {"type"}
    if clash:
        problems.append(f"forceField and material both state {sorted(clash)}: state them in material")

    # Check that comparisonAgainst is paired with agreementTolerance
    comparison = {"compareAgainst", "agreementTolerance"} & set(config)
    if comparison and len(comparison) == 1:
        problems.append(f"{sorted(comparison)[0]} needs the other of compareAgainst/agreementTolerance")

    # expectedOrder: one entry per registered metric, exactly.
    names = {metric.name for metric in METRICS}
    if "expectedOrder" in config and set(config["expectedOrder"]) != names:
        problems.append(f"expectedOrder must state exactly {sorted(names)}, "
                        f"got {sorted(config['expectedOrder'])}")

    if problems:
        raise ValueError(f"{name}: " + "; ".join(problems))


def _validate_boundary_conditions(name, config, geometry):
    """The case's traction regions exist, and state one direction per dimension of the space."""
    traction_on = config.get("tractionOn", {})
    problems = []

    # TODO A rotated stress is not symmetric, and NodalStress only carries a symmetric tensor.
    if traction_on and "rotation" in config:
        problems.append("a rotated problem cannot state tractionOn: its stress is not symmetric")

    unknown = set(traction_on) - set(geometry.region_names)
    if unknown:
        problems.append(f"tractionOn names {sorted(unknown)}, not regions of "
                        f"{type(geometry).__name__}: {sorted(geometry.region_names)}")

    # One direction per dimension of the mesh; the clamp pads the ones it is embedded in.
    for region, directions in traction_on.items():
        if (len(directions) != geometry.dim
                or any(direction not in (0, 1) for direction in directions)):
            problems.append(f"tractionOn[{region!r}] must state {geometry.dim} directions, "
                            f"each 0 or 1, got {directions}")

    if problems:
        raise ValueError(f"{name}: " + "; ".join(problems))


def _validate_rotation(name, config, geometry):
    """Validate whether the case's rotation is well formed."""
    rotation = config.get("rotation")
    if rotation is None:
        return

    if set(rotation) != ROTATION_KEYS:
        raise ValueError(f"{name}: rotation must state exactly {sorted(ROTATION_KEYS)}, "
                         f"got {sorted(rotation)}")
    if rotation["axis"] not in ROTATION_AXES:
        raise ValueError(f"{name}: unknown rotation axis {rotation['axis']!r}, "
                         f"expected one of {sorted(ROTATION_AXES)}")

    fixed = ROTATION_AXES[rotation["axis"]]
    turned = sorted(axis for axis, index in ROTATION_AXES.items() if index != fixed)
    if max(ROTATION_AXES[axis] for axis in turned) >= geometry.dim:
        raise ValueError(f"{name}: a rotation about {rotation['axis']} turns the "
                         f"{' and '.join(turned)} axes, which a {geometry.dim}D mesh does not span")


class Case:
    """One case: the study it specifies, with the geometry and the manufactured problem built."""

    def __init__(self, path, config):
        path = pathlib.Path(path)

        # Extract root path and keep stem as the name of the case.
        stem = path.resolve().with_suffix("")
        try:
            self.name = stem.relative_to(CASES_ROOT.resolve()).as_posix()
        except ValueError:
            self.name = f"{path.parent.name}/{path.stem}"
        _validate(self.name, config)

        # Geometry Class
        geometry_config = dict(config["geometry"])
        self.geometry = GEOMETRIES[geometry_config.pop("type")](**geometry_config)

        _validate_boundary_conditions(self.name, config, self.geometry)
        _validate_rotation(self.name, config, self.geometry)

        # Keyed on the geometry's topological dimension.
        self.traction_on = config.get("tractionOn", {})
        solution = MANUFACTURED_SOLUTIONS[(self.geometry.dim, config["solution"]["type"])](
            config["solution"], self.geometry, self.traction_on)

        # Nothing fixed anywhere leaves the rigid modes in the system.
        if not solution.prescribe_displacement_on:
            raise ValueError(f"{self.name}: tractionOn frees every direction of every region, "
                             f"which leaves the problem singular")

        # ManufacturedProblem Class
        rotation = (rotation_matrix(config["rotation"], self.geometry.spatial_dimensions)
                    if "rotation" in config else None)
        material = MATERIALS[config["material"]["type"]]
        self.manufactured_problem = ManufacturedProblem(
            solution, material(*material.get_material_parameters(
                config["material"], self.geometry.spatial_dimensions)),
            self.geometry.spatial_dimensions, rotation)

        self.element = config["element"]
        component = material.sofa_component_name
        self.material = config["material"] | ({"sofa_component_name": component} if component else {})
        self.force_field = config["forceField"]
        self.solvers = config["solvers"]
        self.mesh = config["mesh"]
        self.quadrature_degree = config["quadratureDegree"]
        self.source_quadrature_degree = config["sourceQuadratureDegree"]
        self.asymptotic_tolerance = config["asymptoticTolerance"]
        self.expected_order = config["expectedOrder"]
        self.expected_order_tolerance = config["expectedOrderTolerance"]
        self.relative_noise_floor = config["relativeNoiseFloor"]
        self.excitation_steps_count = config.get("excitationSteps", 1)
        self.grid_mapping = config.get("gridMapping", {})
        self.compare_against = config.get("compareAgainst")
        self.agreement_tolerance = config.get("agreementTolerance")

    @classmethod
    def load(cls, path):
        with open(path) as f:
            return cls(path, json.load(f))

    @property
    def equation(self):
        """The manufactured solution in math form, for figures and reports."""
        return self.manufactured_problem.equation

    def levels(self):
        """Cells per axis at each refinement level, coarsest first."""
        ratio = self.mesh["refinementRatio"]
        return [[count * ratio ** level for count in self.mesh["cells"]]
                for level in range(self.mesh["levels"])]
