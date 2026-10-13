"""Writes the cruciform meshes of case.json with gmsh: `python mesh.py`."""

import json
import math
import os

import gmsh

CASE_DIR = os.path.dirname(os.path.abspath(__file__))


def write(case):
    gmsh.initialize()
    gmsh.model.add(case['element'])
    half_span, half_arm_width = case['geometry']['halfSpan'], case['geometry']['halfArmWidth']
    thickness, size = case['geometry']['thickness'], case['mesh']['size']

    # The cross as the 12 rectangles of a 4x4 grid without its corners. Their edges on the x and y axes put
    # mesh nodes on the lines the scenes pin.
    cuts = [-half_span, -half_arm_width, 0.0, half_arm_width, half_span]
    rectangles = [(2, gmsh.model.occ.addRectangle(cuts[i], cuts[j], -thickness / 2,
                                                  cuts[i + 1] - cuts[i], cuts[j + 1] - cuts[j]))
                  for i in range(4) for j in range(4) if i in (1, 2) or j in (1, 2)]
    surfaces, _ = gmsh.model.occ.fragment(rectangles, [])
    gmsh.model.occ.synchronize()

    if case['element'] == 'hexa':
        for _, curve in gmsh.model.getEntities(1):
            length = gmsh.model.occ.getMass(1, curve)
            gmsh.model.mesh.setTransfiniteCurve(curve, math.ceil(round(length / size, 6)) + 1)
        for _, surface in surfaces:
            gmsh.model.mesh.setTransfiniteSurface(surface)
            gmsh.model.mesh.setRecombine(2, surface)
    else:
        gmsh.option.setNumber('Mesh.MeshSizeMax', size)

    # An even number of layers puts a layer of nodes on the mid-plane z = 0
    volumes = gmsh.model.occ.extrude(surfaces, 0, 0, thickness, numElements=[case['mesh']['layers']],
                                     recombine=case['element'] == 'hexa')
    gmsh.model.occ.synchronize()
    gmsh.model.addPhysicalGroup(3, [tag for dim, tag in volumes if dim == 3])

    gmsh.model.mesh.generate(3)
    gmsh.write(os.path.join(CASE_DIR, case['mesh']['file']))
    gmsh.finalize()


if __name__ == '__main__':
    with open(os.path.join(CASE_DIR, 'case.json')) as f:
        # The laws share a mesh file per element
        cases = {case['mesh']['file']: case for case in json.load(f).values()}
    for case in cases.values():
        write(case)
