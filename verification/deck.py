"""The deck as one object: parsed, validated, and resolved into what a run needs."""

import json
import pathlib

import Sofa.SofaDeformable

from .manufactured import ManufacturedProblem
from .metrics import METRICS
from .registry import MANUFACTURED_SOLUTIONS, GEOMETRIES, MATERIALS

# Required keyes for all decks. A missing key is an omission, an unknown key is a typo.
REQUIRED = {"geometry", "element", "solution", "material", "forceField",
            "quadratureDegree", "sourceQuadratureDegree", "solvers", "mesh",
            "asymptoticTolerance", "expectedOrder", "expectedOrderTolerance", "noiseFloorFraction"}

MESH_KEYS = {"cells", "levels", "refinementRatio"}


# dim -> Young/Poisson -> (mu, lambda)
def _toLameParameters1D(youngModulus, poissonRatio):
    """(mu, lambda) at d = 1, where lambda + 2 mu = E collapses Hooke to sigma = E eps."""
    return 0.5 * youngModulus, 0.0

_LAME = {1: _toLameParameters1D,
         2: Sofa.SofaDeformable.toLameParameters2D,
         3: Sofa.SofaDeformable.toLameParameters3D}


def _material(spec, spatial_dimensions):
    """The deck's material, built from the parameters it states."""
    return MATERIALS[spec["type"]](*_LAME[spatial_dimensions](spec["youngModulus"],
                                                             spec["poissonRatio"]))


def _validate(name, spec):
    """Every required key present, nothing unknown, and one expectation per metric.

    `solvers` and `forceField` pass their own keys through to SOFA Data verbatim, so this module
    cannot know what's legal inside them and only checks their presence/shape, not their contents.
    Every problem is collected before raising, so a deck is corrected in one pass, not one key per run.
    """
    problems = []

    # Top-level keys: exact match against REQUIRED.
    missing = REQUIRED - set(spec)
    if missing:
        problems.append(f"missing {sorted(missing)}")
    unknown = set(spec) - REQUIRED
    if unknown:
        problems.append(f"unknown {sorted(unknown)}")

    # mesh: exact match against MESH_KEYS.
    if "mesh" in spec and set(spec["mesh"]) != MESH_KEYS:
        problems.append(f"mesh must state exactly {sorted(MESH_KEYS)}, got {sorted(spec['mesh'])}")

    # solution: just needs a type and an amplitude; the rest is that solution's own business.
    solution_spec = spec.get("solution", {})
    absent = {"type", "amplitude"} - set(solution_spec)
    if absent:
        problems.append(f"solution must state {sorted(absent)}")

    # material: must name a registered type.
    material = spec.get("material", {})
    if "type" not in material:
        problems.append(f"material must state a type, one of {sorted(MATERIALS)}")
    elif material["type"] not in MATERIALS:
        problems.append(f"unknown material {material['type']!r}, expected one of {sorted(MATERIALS)}")

    # forceField: must be an object stating a type.
    force_field = spec.get("forceField")
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
    if "expectedOrder" in spec and set(spec["expectedOrder"]) != names:
        problems.append(f"expectedOrder must state exactly {sorted(names)}, "
                        f"got {sorted(spec['expectedOrder'])}")

    if problems:
        raise ValueError(f"{name}: " + "; ".join(problems))


class Deck:
    """One deck: the study it specifies, with the geometry and the manufactured problem built."""

    def __init__(self, path, spec):
        path = pathlib.Path(path)

        # TODO Rework ugly naming convention
        self.name = f"{path.parent.name}/{path.stem}"
        _validate(self.name, spec)

        # Geometry Class
        geometry_spec = dict(spec["geometry"])
        self.geometry = GEOMETRIES[geometry_spec.pop("type")](**geometry_spec)

        # ManufacturedSolution Class. Keyed on the geometry's topological dimension.
        solution = MANUFACTURED_SOLUTIONS[(self.geometry.dim, spec["solution"]["type"])](
            spec["solution"], self.geometry)

        # ManufacturedProblem Class
        self.manufactured_problem = ManufacturedProblem(
            solution, _material(spec["material"], self.geometry.spatial_dimensions),
            self.geometry.spatial_dimensions)

        self.element = spec["element"]
        self.material = spec["material"]
        self.force_field = spec["forceField"]
        self.solvers = spec["solvers"]
        self.mesh = spec["mesh"]
        self.quadrature_degree = spec["quadratureDegree"]
        self.source_quadrature_degree = spec["sourceQuadratureDegree"]
        self.asymptotic_tolerance = spec["asymptoticTolerance"]
        self.expected_order = spec["expectedOrder"]
        self.expected_order_tolerance = spec["expectedOrderTolerance"]
        self.noise_floor_fraction = spec["noiseFloorFraction"]

    @classmethod
    def load(cls, path):
        with open(path) as f:
            return cls(path, json.load(f))

    @property
    def equation(self):
        """The manufactured solution in math form, for figures and reports."""
        return self.manufactured_problem.equation

    def levels(self):
        """Grid cells per axis and mesh spacing per level, coarsest first.

        Refinement is at a fixed ratio on every axis, so there's no per-level list to drift.
        Cells, not elements: the deck controls what `RegularGridTopology` lays down, and the
        topology mappings split each cell by a fixed factor across levels, so `h` (the largest
        cell diameter) stays the honest mesh parameter for a convergence rate.
        """
        ratio = self.mesh["refinementRatio"]
        sweep = []
        for level in range(self.mesh["levels"]):
            cells = [count * ratio ** level for count in self.mesh["cells"]]
            spacing = max(extent / count
                          for extent, count in zip(self.geometry.parameters, cells))
            sweep.append((cells, spacing))
        return sweep
