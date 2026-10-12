from .base import ScenePrefab
from .grid import ElasticGrid, ElasticBar, ElasticBeam
from .tire import ElasticTire
from ...geometry import Grid, Bar1D, Beam2D, Beam3D, Tire, Tire2D, Tire3D

PREFAB_BY_GEOMETRY = {Grid: ElasticGrid, Bar1D: ElasticBar,
                      Beam2D: ElasticBeam, Beam3D: ElasticBeam,
                      Tire: ElasticTire, Tire2D: ElasticTire, Tire3D: ElasticTire}
