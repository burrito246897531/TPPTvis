# For generating source objects

'''
Hopefully intuitive tool for generating source object vtp files to render in TPPTvis.
Examples are given below, vtpgenerator_utils.py contains functions for:
    - Cylinder mesh
    - Rectangular prism mesh
    - Save mesh as vtp file
'''

import vtpgenerator_utils as vg

# Example 1: Ge68 line source, 2mm radius, 100mm length, centered at (0, 0, 0) along the z axis
ge68_line_source = vg.generate_cylinder_mesh(0, 0, 0, 2, 100, direction=(0, 0, 1), resolution=64)
vg.save_mesh_as_vtp(ge68_line_source, 'ge68_line_source1.vtp')

cube_source = vg.generate_rectangular_prism_mesh(0, 0, 0, 10, 10, 10)
vg.save_mesh_as_vtp(cube_source, 'cube_source.vtp')