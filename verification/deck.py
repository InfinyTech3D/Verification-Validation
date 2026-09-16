"""The deck as one object: parsed, checked, and resolved into what a run needs.

A deck is the whole specification of a study, and every parameter in it is explicit -- there are no
defaults to inherit and nothing is inferred. Checking that here, before the first scene is built, is
what turns a mistyped key from a KeyError six meshes into a sweep into one line before any SOFA work
happens. It is also the one place that reads a deck, so a change to the deck's vocabulary is a change
to this module rather than to the runner and the scene at once.
"""

import json
import pathlib

import Sofa.SofaDeformable

from .manufactured import ManufacturedProblem
from .metrics import METRICS
from .registry import FIELDS, GEOMETRIES, MATERIALS

# Every key a deck must state. Strict in both directions: a missing key is an omission and an unknown
# key is a typo, and neither should be discovered halfway through a sweep.
REQUIRED = {"geometry", "element", "function", "material", "forceField",
            "quadratureDegree", "sourceQuadratureDegree", "solvers", "mesh",
            "asymptoticTolerance", "expect", "expectTolerance", "noiseFloorRelative"}

MESH_KEYS = {"cells", "levels", "refinementRatio"}


def _toLameParameters1D(youngModulus, poissonRatio):
    """(mu, lambda) at d = 1, where lambda + 2 mu = E collapses Hooke to sigma = E eps."""
    return 0.5 * youngModulus, 0.0


# Spatial dimension -> SOFA's own Young/Poisson -> (mu, lambda) converter. Going through the bindings
# keeps the manufactured source term on exactly the constitutive branch the solver takes (plane stress
# at 2, plane strain at 3). d = 1 is ours: there is no SOFA branch to defer to.
_LAME = {1: _toLameParameters1D,
         2: Sofa.SofaDeformable.toLameParameters2D,
         3: Sofa.SofaDeformable.toLameParameters3D}


def _material(spec, spatial_dimensions):
    """The deck's material, built from the parameters it states.

    The conversion sits here rather than on the material because it is where SOFA is the authority:
    the symbolic laws take numbers and stay independent of a build. A material parameterized some
    other way than by Young and Poisson converts its own way, which is a branch for this function
    rather than a second constitutive law.
    """
    return MATERIALS[spec["type"]](*_LAME[spatial_dimensions](spec["youngModulus"],
                                                             spec["poissonRatio"]))


def _validate(name, spec):
    """Every required key present, nothing unknown, and one expectation per metric.

    `solvers` is checked only for its presence: DESIGN.md's contract is that its keys reach the
    component as Data verbatim, so a deck can set anything a component exposes without a Python
    change -- which means this module cannot know what is legal inside it. `geometry` is likewise left
    to the geometry class, whose keyword arguments differ per shape and already reject both a missing
    extent and an unknown one.

    Every problem is collected before raising, so a deck written from scratch is corrected in one pass
    rather than one key per run.
    """
    problems = []
    missing = REQUIRED - set(spec)
    if missing:
        problems.append(f"missing {sorted(missing)}")
    unknown = set(spec) - REQUIRED
    if unknown:
        problems.append(f"unknown {sorted(unknown)}")

    if "mesh" in spec and set(spec["mesh"]) != MESH_KEYS:
        problems.append(f"mesh must state exactly {sorted(MESH_KEYS)}, got {sorted(spec['mesh'])}")

    # The field's own block. A type and an amplitude are common to every field and anything else in
    # there is that field's business, so unknown keys are not an error the way they are at the top
    # level -- the same reasoning `solvers` gets, one axis down.
    function = spec.get("function", {})
    absent = {"type", "amplitude"} - set(function)
    if absent:
        problems.append(f"function must state {sorted(absent)}")

    # The material is named like everything else rather than assumed: which constitutive law a study
    # verified against is part of what it verified, not a default the code picks.
    material = spec.get("material", {})
    if "type" not in material:
        problems.append(f"material must state a type, one of {sorted(MATERIALS)}")
    elif material["type"] not in MATERIALS:
        problems.append(f"unknown material {material['type']!r}, expected one of {sorted(MATERIALS)}")

    # The component under test gets the same contract as `solvers`: a type plus whatever Data the
    # deck wants to set, so unknown keys are that component's business rather than this module's.
    # dict rather than truthiness: a bare component name is the pre-passthrough spelling, and `in`
    # on a string is a substring test, so it would pass the type check and fail later as a TypeError.
    force_field = spec.get("forceField")
    if not isinstance(force_field, dict):
        problems.append(f"forceField must be an object stating a type, got {force_field!r}")
        force_field = {}
    elif "type" not in force_field:
        problems.append("forceField must state a type")
    # `name` is what the runner reads the potential energy off, and `template` and `topology` follow
    # from `element` and the mesh -- a deck stating either could contradict what it already stated.
    owned = {"name", "template", "topology"} & set(force_field)
    if owned:
        problems.append(f"forceField must not state {sorted(owned)}: the prefab sets them")
    # One home for the constitutive parameters: the manufactured source reads `material` too, and two
    # sources is how a source term and the component it verifies drift apart silently.
    clash = (set(force_field) & set(material)) - {"type"}
    if clash:
        problems.append(f"forceField and material both state {sorted(clash)}: state them in material")

    # One expectation per metric, so adding a metric cannot leave the decks silently unjudged: the
    # study looks `expect` up by metric name, and a name it cannot find is a KeyError mid-sweep.
    names = {metric.name for metric in METRICS}
    if "expect" in spec and set(spec["expect"]) != names:
        problems.append(f"expect must state exactly {sorted(names)}, got {sorted(spec['expect'])}")

    if problems:
        raise ValueError(f"{name}: " + "; ".join(problems))


class Deck:
    """One deck: the study it specifies, with the geometry and the manufactured solution built."""

    def __init__(self, path, spec):
        path = pathlib.Path(path)
        # <dir>/<stem>. The dimension a run uses comes from geometry.dim, never from the folder name,
        # but the folder is what tells two decks of the same stem apart in a table.
        self.name = f"{path.parent.name}/{path.stem}"
        _validate(self.name, spec)

        geometry_spec = dict(spec["geometry"])
        self.geometry = GEOMETRIES[geometry_spec.pop("type")](**geometry_spec)
        # Keyed on the geometry's topological dimension, not the space it is embedded in: the same
        # field name means the same field in 1D, 2D and 3D.
        field = FIELDS[(self.geometry.dim, spec["function"]["type"])](
            spec["function"], self.geometry)
        # Field and material are independent axes of a study, so the deck names one of each and this
        # is where they are paired, in the space the mesh will be embedded in.
        self.solution = ManufacturedProblem(
            field, _material(spec["material"], self.geometry.spatial_dimensions),
            self.geometry.spatial_dimensions)

        self.element = spec["element"]
        self.material = spec["material"]
        self.force_field = spec["forceField"]
        self.solvers = spec["solvers"]
        self.mesh = spec["mesh"]
        self.quadrature_degree = spec["quadratureDegree"]
        self.source_quadrature_degree = spec["sourceQuadratureDegree"]
        self.asymptotic_tolerance = spec["asymptoticTolerance"]
        self.expect = spec["expect"]
        self.expect_tolerance = spec["expectTolerance"]
        self.noise_floor_relative = spec["noiseFloorRelative"]

    @classmethod
    def load(cls, path):
        with open(path) as f:
            return cls(path, json.load(f))

    @property
    def equation(self):
        """The manufactured field in math form, which titles the figures and heads the record.

        Delegated rather than reached for through `solution`, so a study reads one flat set of deck
        attributes and does not need to know a solution object exists.
        """
        return self.solution.equation

    def levels(self):
        """Grid cells per axis and mesh spacing per level, coarsest first.

        A deck states the coarsest cell count and how many times to refine it, so refinement at a fixed
        ratio on every axis is structural: there is no per-level list for a hand-written value to drift
        in, and the constant ratio the settling test assumes is the ratio the meshes have.

        Cells, not elements: what the deck controls is the grid `RegularGridTopology` lays down, and the
        topology mappings then split each cell -- 6 tetrahedra per hexahedron, 2 triangles per quad.
        Those elements are larger than the cell (the tetrahedra span its body diagonal), but by a factor
        fixed across levels, so it cancels in the ratio the pairwise order divides by and `h` stays the
        honest mesh parameter for a rate.
        """
        ratio = self.mesh["refinementRatio"]
        sweep = []
        for level in range(self.mesh["levels"]):
            cells = [count * ratio ** level for count in self.mesh["cells"]]
            # The classical mesh parameter is the largest cell diameter, which the coarsest direction sets.
            spacing = max(extent / count
                          for extent, count in zip(self.geometry.parameters, cells))
            sweep.append((cells, spacing))
        return sweep
