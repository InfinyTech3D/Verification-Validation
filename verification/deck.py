"""The deck as one object: parsed, validated, and resolved into what a run needs."""

import json
import pathlib

import Sofa.SofaDeformable

from .manufactured import ManufacturedProblem, rotation_matrix
from .metrics import METRICS
from common.sofa.conventions import ELEMENTS

from .registry import MANUFACTURED_SOLUTIONS, GEOMETRIES, MATERIALS

# Required keys for all decks. A missing key is an omission, an unknown key is a typo.
REQUIRED = {"geometry", "element", "solution", "material", "forceField",
            "quadratureDegree", "sourceQuadratureDegree", "solvers", "mesh",
            "asymptoticTolerance", "expectedOrder", "expectedOrderTolerance", "relativeNoiseFloor"}

OPTIONAL = {
        "excitationSteps" # equal increments the loads are applied in; absent means one.
        , "tractionOn"    # {region: directions} handed to the manufactured traction; the rest stay Dirichlet.
        , "rotation"      # a rigid rotation of the whole problem; its absence is the statement.
        , "gridMapping"   # Data for the mapping meshing the grid, where the element kind needs one.
}

MESH_KEYS = {"cells", "levels", "refinementRatio"}

ROTATION_KEYS = {"angleDegrees", "axis"}

# Rotation axis -> the coordinate axis it leaves fixed.
ROTATION_AXES = {"x": 0, "y": 1, "z": 2}


# dim -> Young/Poisson -> (mu, lambda)
def _toLameParameters1D(youngModulus, poissonRatio):
    """(mu, lambda) at d = 1, where lambda + 2 mu = E collapses Hooke to sigma = E eps."""
    return 0.5 * youngModulus, 0.0

_LAME = {1: _toLameParameters1D,
         2: Sofa.SofaDeformable.toLameParameters2D,
         3: Sofa.SofaDeformable.toLameParameters3D}


def _material(config, spatial_dimensions):
    """The deck's material, built from the parameters it states."""
    return MATERIALS[config["type"]](*_LAME[spatial_dimensions](config["youngModulus"],
                                                             config["poissonRatio"]))


def _validate(name, config):
    """Every required key present, nothing unknown, and one expectation per metric.

    `solvers` and `forceField` pass their own keys through to SOFA Data verbatim, so this module
    cannot know what's legal inside them and only checks their presence/shape, not their contents.
    Every problem is collected before raising, so a deck is corrected in one pass, not one key per run.
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
    # from element and the mesh, so a deck stating either could contradict what it already stated.
    owned = {"name", "template", "topology"} & set(force_field)
    if owned:
        problems.append(f"forceField must not state {sorted(owned)}: the prefab sets them")
    # forceField must not restate a constitutive parameter that belongs in material, so the
    # manufactured source and the forceField under test can't drift apart.
    clash = (set(force_field) & set(material)) - {"type"}
    if clash:
        problems.append(f"forceField and material both state {sorted(clash)}: state them in material")

    # expectedOrder: one entry per registered metric, exactly.
    names = {metric.name for metric in METRICS}
    if "expectedOrder" in config and set(config["expectedOrder"]) != names:
        problems.append(f"expectedOrder must state exactly {sorted(names)}, "
                        f"got {sorted(config['expectedOrder'])}")

    if problems:
        raise ValueError(f"{name}: " + "; ".join(problems))


def _validate_boundary_conditions(name, config, geometry):
    """The deck's traction regions exist, and state one direction per dimension of the space."""
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
    """Validate whether the deck's rotation is well formed."""
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


class Deck:
    """One deck: the study it specifies, with the geometry and the manufactured problem built."""

    def __init__(self, path, config):
        path = pathlib.Path(path)

        # TODO Rework ugly naming convention
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
        self.manufactured_problem = ManufacturedProblem(
            solution, _material(config["material"], self.geometry.spatial_dimensions),
            self.geometry.spatial_dimensions, rotation)

        self.element = config["element"]
        self.material = config["material"]
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
