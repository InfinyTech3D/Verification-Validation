from .base import ScenePrefab
from .grid import ElasticGrid, ElasticBar, ElasticBeam
from .tyre import ElasticTyre
from ...geometry import Grid, Bar1D, Beam2D, Beam3D, Tyre, Tyre2D, Tyre3D

PREFAB_BY_GEOMETRY = {Grid: ElasticGrid, Bar1D: ElasticBar,
                      Beam2D: ElasticBeam, Beam3D: ElasticBeam,
                      Tyre: ElasticTyre, Tyre2D: ElasticTyre, Tyre3D: ElasticTyre}
