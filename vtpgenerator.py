# For generating source objects

'''
Hopefully intuitive tool for generating source object vtp files to render in TPPTvis.
Examples are given below, vtpgenerator_utils.py contains functions for:
    - Cylinder mesh
    - Rectangular prism mesh
    - CASToR mesh
    - Save mesh as vtp file
'''

import vtpgenerator_utils as vg

# Example 1: Ge68 line source, 2mm radius, 100mm length, centered at (0, 0, 0) along the z axis
#ge68_line_source = vg.generate_cylinder_mesh(0, 0, 0, 2, 100, direction=(0, 0, 1), resolution=64)
#vg.save_mesh_as_vtp(ge68_line_source, 'ge68_line_source3.vtp')

#cube_source = vg.generate_rectangular_prism_mesh(0, 0, 0, 10, 10, 10)
#vg.save_mesh_as_vtp(cube_source, 'cube_source.vtp')

#file_name = 'Run13_ReRun11_DerenzoPhant2mm_NoColi_PMMASpacer_89.6MeV_1200MU_60cyc_30min_HWTrigOn_coinc_363.53_end_1sigma_it4'
#castor_mesh = vg.castor_mesh(file_name + '.img', file_name + '.hdr', prop = 0.001, trim = 56, color_map = 'viridis', opacity_map_func = vg.get_opacity_09cubic)
#vg.save_mesh_as_vtp(castor_mesh, 'd1v001prop09cubic.vtp')
#castor_mesh = vg.castor_mesh(file_name + '.img', file_name + '.hdr', prop = 0.001, trim = 56, color_map = 'hot_r', opacity_map_func = vg.get_opacity_09cubic)
#vg.save_mesh_as_vtp(castor_mesh, 'd1d001prop09cubic.vtp')
#castor_mesh = vg.castor_mesh(file_name + '.img', file_name + '.hdr', prop = 0.0008, trim = 56, color_map = 'viridis', opacity_map_func = vg.get_opacity_09cubic)
#vg.save_mesh_as_vtp(castor_mesh, 'd1v0008prop09cubic.vtp')
#castor_mesh = vg.castor_mesh(file_name + '.img', file_name + '.hdr', prop = 0.0008, trim = 56, color_map = 'hot_r', opacity_map_func = vg.get_opacity_09cubic)
#vg.save_mesh_as_vtp(castor_mesh, 'd1d0008prop09cubic.vtp')

derenzo_mesh = vg.generate_derenzo_mesh(0, 0, 0, 1.59 / 2, 100, angle = 270,resolution=64, color = [255, 0, 0] )
vg.save_mesh_as_vtp(derenzo_mesh, 'derenzo2mm.vtp')