import os
import pyvista as pv
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

default_color = [255, 153, 51] # orange
default_opacity = 0.25  # Default opacity for meshes

def get_color(value, color_map = 'viridis', reverse = False):
    cmap = plt.get_cmap(color_map)
    if reverse:
        value = 1 - value
    rgba = cmap(value)
    rgb = [int(255 * c) for c in rgba[:3]]
    return rgb

def get_opacity_08linear(value):
    return int(round(255 * (0.8 * value + 0.2)))

def get_opacity_09linear(value):
    return int(round(255 * (0.9 * value + 0.1)))

def get_opacity_08cubic(value):
    return int(round(255 * (0.8 * value**3 + 0.2)))

def get_opacity_09cubic(value):
    return int(round(255 * (0.9 * value**3 + 0.1)))

def get_opacity_08quadratic(value):
    return int(round(255 * (0.8 * value**2 + 0.2)))

def get_opacity_09quadratic(value):
    return int(round(255 * (0.9 * value**2 + 0.1)))

def generate_cylinder_mesh(x, y, z, r, h, direction=(0, 0, 1), resolution=64, color=default_color, opacity=default_opacity):
    """
    Create a cylinder mesh centered at (x, y, z).

    Parameters
    ----------
    x, y, z : float
        Center coordinates of the cylinder.
    r : float
        Radius of the cylinder.
    h : float
        Height of the cylinder.
    direction : tuple[float, float, float], optional
        Axis direction vector for the cylinder; defaults to +Z.
    resolution : int, optional
        Number of facets around the circumference; higher = smoother.
    color : list[int, int, int], optional
        RGB color values between 0 and 255 for the mesh. Defaults to orange.
    opacity : float, optional
        Opacity value between 0 and 1 for the mesh. Defaults to 0.25.

    Returns
    -------
    pyvista.PolyData
        Cylinder mesh. If color is specified, the mesh will have cell or point data "RGB".
        Opacity is stored in field_data.
    """
    # Generate a cylinder mesh
    mesh = pv.Cylinder(center=(x, y, z), direction=direction, radius=r, height=h, resolution=resolution)
    rgba = np.array([*color, opacity])
    mesh.cell_data['RGBA'] = np.tile(rgba, (mesh.n_cells, 1))
    return mesh


def generate_rectangular_prism_mesh(x, y, z, l, w, h, color=default_color, opacity=default_opacity):
    """
    Create a rectangular prism mesh centered at (x, y, z).

    Parameters
    ----------
    x, y, z : float
        Center coordinates of the rectangular prism.
    l : float
        Length of the prism in the x direction.
    w : float
        Width of the prism in the y direction.
    h : float
        Height of the prism in the z direction.
    color : list[int, int, int], optional
        RGB color values between 0 and 255 for the mesh. Defaults to orange.
    opacity : float or array-like, optional
        Opacity value(s) between 0 and 1 for the mesh. Can be a single value for all cells,
        or an array with one value per cell. Defaults to 0.25.
    Returns
    -------
    pyvista.PolyData
        Rectangular prism mesh. Opacity is stored in cell_data.
    """
    # Calculate bounds: [x_min, x_max, y_min, y_max, z_min, z_max]
    bounds = [
        x - l / 2, x + l / 2,  # x bounds
        y - w / 2, y + w / 2,  # y bounds
        z - h / 2, z + h / 2   # z bounds
    ]
    mesh = pv.Box(bounds=bounds)
    rgba = np.array([*color, opacity])
    mesh.cell_data['RGBA'] = np.tile(rgba, (mesh.n_cells, 1))
    return mesh

def castor_mesh(img_path, hdr_path, prop=0.001, trim=3, color_map = 'viridis', opacity_map_func = get_opacity_08cubic):
    """
    Create a mesh of voxels from CASToR reconstructed image. This function is deprecated, TPPTvis now has functionality
    to generate CASToR meshes from .hdr/.img files within the app using thresholded surfaces.

    Parameters ----------
    img_path : str
        Path to the CASToR reconstructed image.
    hdr_path : str
        Path to the CASToR header file.
    prop : float, optional
        Proportion of voxels to be shown, default shows top 1% of voxels (prop = 0.001)
    trim : int, optional
        Number of voxels to trim from the edges of the image, default is 3 (trim = 3)

    Returns----------
    pyvista.PolyData
        CASToR mesh. Each voxel can have different opacity stored in cell_data.
    """
    # Read header file for dimensions and voxel sizes
    dims = []
    voxelsizes = []
    bytes_per_pixel = 0
    with open(os.path.join(os.path.dirname(__file__), 'CASToR_files', hdr_path), 'r') as f:
        for line in f:
            if '!matrix size' in line:
                dims.append(int(line[line.index(':=') + 3:-1]))
            elif 'scaling factor' in line:
                voxelsizes.append(float(line[line.index(':=') + 3:-1]))
            elif '!number of bytes per pixel' in line:
                bytes_per_pixel = int(line[line.index(':=') + 3:-1])

    # Read image file
    with open(os.path.join(os.path.dirname(__file__), 'CASToR_files', img_path), 'rb') as f:
        img_data = np.fromfile(f, dtype=np.dtype(f'uint{bytes_per_pixel * 8}')).astype(np.float32).reshape(dims, order = 'F')

    # Trim the image
    img_data = img_data[trim:-trim, trim:-trim, trim:-trim]
    dims = img_data.shape

    # Get top proportion of voxels
    sorted = np.sort(img_data.flatten())
    maxval = sorted[-1]
    threshold = sorted[int((1 - prop) * len(sorted))]
    img_data[img_data <= threshold] = 0

    # Convert to semi-normalized values for color mapping
    scalemax = sorted[int((1 - prop / 10) * len(sorted))] 
    img_data = (img_data - threshold) / (scalemax - threshold)
    img_data[img_data > 1] = 1
    
    # Geometry for meshes
    x0 = (-0.5 * (dims[0] - 1) * voxelsizes[0])
    y0 = (-0.5 * (dims[1] - 1) * voxelsizes[1])
    z0 = (-0.5 * (dims[2] - 1) * voxelsizes[2])
    mesh = None
    for x in tqdm(range(dims[0])):
        for y in range(dims[1]):
            for z in range(dims[2]):
                if img_data[x, y, z] > 0:
                    xpos = x0 + x * voxelsizes[0]
                    ypos = y0 + y * voxelsizes[1]
                    zpos = z0 + z * voxelsizes[2]
                    if mesh is None:
                        mesh = generate_rectangular_prism_mesh(xpos, ypos, zpos, 
                        voxelsizes[0], voxelsizes[1], voxelsizes[2], 
                        color=get_color(img_data[x, y, z], color_map = color_map), 
                        opacity=opacity_map_func(img_data[x, y, z]))
                    else:
                        mesh += generate_rectangular_prism_mesh(xpos, ypos, zpos, 
                        voxelsizes[0], voxelsizes[1], voxelsizes[2],
                        color=get_color(img_data[x, y, z], color_map = color_map),
                        opacity=opacity_map_func(img_data[x, y, z]))
    
    # Save metadata to use for colorbar later
    if mesh is not None:
        mesh.field_data['threshold_ratio'] = np.array([threshold / maxval])
        mesh.field_data['scalemax_ratio'] = np.array([scalemax / maxval])
    #print(mesh.cell_data['RGBA'])
    return mesh

def generate_derenzo_mesh(x, y, z, r, h, angle = 270, resolution=64, color=default_color, opacity=default_opacity):
    """
    Create a Derenzo mesh with the tip rod centered at (x, y, z).
    Angle specifies the angle that the tip rod "points", 0 angle is +x.
    By default, the tip rod points down (270 deg), remaining rods are generated behind.

    Parameters
    ----------
    x, y, z : float
        Center coordinates of the tip rod.
    r : float
        Radius of the Derenzo rods.
    h : float
        Height of the Derenzo rods.
    angle : float, optional
        Angle of the Derenzo mesh.
    resolution : int, optional
        Number of facets around the circumference; higher = smoother.
    color : list[int, int, int], optional
        RGB color values between 0 and 255 for the mesh. Defaults to orange.
    opacity : float, optional
        Opacity value between 0 and 1 for the mesh. Defaults to 0.25.
    """
    tip_rod = generate_cylinder_mesh(x, y, z, r, h, direction=(0, 0, 1), resolution=resolution, color=color, opacity=opacity)
    rel_coords = 4 * r * np.array([[-np.sqrt(3)/2, 1/2], [-np.sqrt(3)/2, -1/2], [-np.sqrt(3), 1], [-np.sqrt(3), 0], [-np.sqrt(3), -1]]) # relative coordinates of the rods
    angle = np.deg2rad(angle)
    rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    rel_coords = [rotation_matrix @ coord for coord in rel_coords]
    for i in range(5):
        tip_rod += generate_cylinder_mesh(rel_coords[i][0], rel_coords[i][1], z, r, h, direction=(0, 0, 1), resolution=resolution, color=color, opacity=opacity)
    return tip_rod

def save_mesh_as_vtp(mesh, filepath):
    """
    Save a PyVista mesh as a single VTP file, inside a 'source_vtps' folder.

    Parameters
    ----------
    mesh : pyvista.DataSet
        The mesh to save.
    filepath : str
        Destination path (basename or path); ".vtp" is appended if missing.

    Returns
    -------
    str
        The path where the mesh was saved.
    """
    import os

    if mesh is None:
        raise ValueError("mesh is None; cannot save.")

    if not filepath.lower().endswith(".vtp"):
        filepath = f"{filepath}.vtp"

    # Always save into 'source_vtps' folder (relative to current directory)
    folder = "source_vtps"
    # Extract just the base of the filepath in case user passes in directories
    filename = os.path.basename(filepath)
    out_path = os.path.join(os.path.dirname(filepath), folder, filename)

    # Ensure the 'source_vtps' folder exists
    os.makedirs(folder, exist_ok=True)

    mesh.save(out_path)
    print(f"Mesh saved to {out_path}")

