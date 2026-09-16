"""Name -> class registries for deck-driven verification tests."""

from ..common.geometry import Bar1D, Beam2D, Beam3D
from . import manufactured
from .materials import LinearElastic

# deck "geometry.type" -> geometry class, keyed by the class's own name.
GEOMETRIES = {cls.__name__: cls for cls in (Bar1D, Beam2D, Beam3D)}

# deck "material.type" -> material class, keyed by the class's own name.
MATERIALS = {cls.__name__: cls for cls in (LinearElastic,)}


class _ManufacturedSolutionRegistry:
    """(topological dim of the geometry, deck "solution.type") -> ManufacturedSolution class.

    Looked up by naming convention: a deck's solution type "Quadratic" at dim 2 resolves to
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
