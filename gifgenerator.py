# Script for generating GIFs of source_vtp objects

'''
Generate a GIF of the object rotating around the z axis, saved in gifs folder.
Need to specify:
    - The name of the vtp file, script will look in source_vtps folder for the file.
    - Frames per second
    - Duration of GIF in seconds
    - Camera angle from vertical (0 is vertical)
    - Camera distance from object
    - Focal point
    - Window (image) size
'''

vtp_names = ['source_2.vtp']
color_scheme = 'grayscale'
fps = 20
duration = 15
camera_angle = 60 # Degrees from vertical
camera_distance = 720
focal_point = [0, 0, 5]
window_size = (384, 512)




'''
Begin code, no need to touch
'''

import os
import pyvista as pv
import numpy as np
from tqdm import tqdm
import color_schemes



plotter = pv.Plotter(off_screen=True, window_size=window_size)
color_scheme = color_schemes.COLOR_SCHEMES[color_scheme]
plotter.set_background(color_scheme['background'])

current_dir = os.path.dirname(os.path.abspath(__file__))
for vtp_name in vtp_names:
    mesh_path = os.path.join(current_dir, 'source_vtps', vtp_name)
    mesh = pv.read(mesh_path)
    # Check if mesh has RGBA data (which includes opacity in alpha channel)
    if 'RGBA' in mesh.cell_data:
        # Preserve original opacities by using RGBA scalars
        plotter.add_mesh(mesh, scalars='RGBA', rgb=True, show_edges=False)
    else:
        # If no RGBA data, check for separate opacity in cell_data or field_data
        opacity = None
        if 'opacity' in mesh.cell_data:
            # Use per-cell opacity if available
            opacity = mesh.cell_data['opacity']
        elif 'opacity' in mesh.field_data:
            # Use global opacity if available
            opacity_val = mesh.field_data['opacity']
            if isinstance(opacity_val, np.ndarray):
                opacity = float(opacity_val[0])
            else:
                opacity = float(opacity_val)
            
            if opacity is not None:
                plotter.add_mesh(mesh, opacity=opacity, show_edges=False)
            else:
                # No opacity data found, use default
                plotter.add_mesh(mesh, show_edges=False)



plotter.open_gif(os.path.join(current_dir, 'gifs', vtp_name.replace('.vtp', '.gif')), fps=fps) 

# Set camera to be camera_distance away from the object (center)
plotter.camera.focal_point = focal_point

# Set camera position
camera_angle = np.deg2rad(camera_angle)
plotter.camera.position = [camera_distance * np.sin(camera_angle), 0, camera_distance * np.cos(camera_angle)]

# Rotate camera and capture frames
n_frames = fps * duration
for _ in tqdm(range(n_frames)):
    angle = 360 * _ / n_frames
    plotter.camera.azimuth = angle
    plotter.render()
    plotter.write_frame()

plotter.close()




