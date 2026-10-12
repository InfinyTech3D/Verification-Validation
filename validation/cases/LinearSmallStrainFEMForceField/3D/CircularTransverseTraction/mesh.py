"""Writes the cylinder meshes of case.json with gmsh: `python mesh.py`."""

import json
import os

import gmsh

CASE_DIR = os.path.dirname(os.path.abspath(__file__))


def write(case):
    gmsh.initialize()
    gmsh.model.add(case['element'])
    length, radius = case['geometry']['length'], case['geometry']['radius']
    size = case['mesh']['size']

    disk = gmsh.model.occ.addDisk(0, 0, 0, radius, radius, zAxis=[1, 0, 0])
    gmsh.model.occ.extrude([(2, disk)], length, 0, 0)
    gmsh.model.occ.synchronize()
    gmsh.model.addPhysicalGroup(3, [tag for _, tag in gmsh.model.getEntities(3)])

    gmsh.option.setNumber('Mesh.MeshSizeMax', size)
    gmsh.model.mesh.generate(3)
    gmsh.write(os.path.join(CASE_DIR, case['mesh']['file']))
    gmsh.finalize()


if __name__ == '__main__':
    with open(os.path.join(CASE_DIR, 'case.json')) as f:
        for case in json.load(f).values():
            write(case)
