from .base import ScenePrefab
from .bar import ElasticBar
from .beam import ElasticBeam
from ...geometry import Bar1D, Beam2D, Beam3D

PREFAB_BY_GEOMETRY = {Bar1D: ElasticBar, Beam2D: ElasticBeam, Beam3D: ElasticBeam}
