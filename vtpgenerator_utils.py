import os
import pyvista as pv
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

default_color = [255, 153, 51] # orange


def generate_cylinder_mesh(x, y, z, r, h, direction=(0, 0, 1), resolution=64, color=default_color):
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

    Returns
    -------
    pyvista.PolyData
        Cylinder mesh. If color is specified, the mesh will have cell or point data "RGB".
    """
    # Generate a cylinder mesh
    mesh = pv.Cylinder(center=(x, y, z), direction=direction, radius=r, height=h, resolution=resolution)
    mesh.cell_data['RGB'] = [color] * mesh.n_cells
    return mesh


def generate_rectangular_prism_mesh(x, y, z, l, w, h, color = default_color):
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
    Returns
    -------
    pyvista.PolyData
        Rectangular prism mesh.
    """
    # Calculate bounds: [x_min, x_max, y_min, y_max, z_min, z_max]
    bounds = [
        x - l / 2, x + l / 2,  # x bounds
        y - w / 2, y + w / 2,  # y bounds
        z - h / 2, z + h / 2   # z bounds
    ]
    mesh = pv.Box(bounds=bounds)
    mesh.cell_data['RGB'] = [color] * mesh.n_cells
    return mesh

def castor_mesh(img_path, hdr_path, prop = 0.001, trim = 3):
    """
    Create a mesh of voxels from CASToR reconstructed image.

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
        CASToR mesh.
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
                        mesh = generate_rectangular_prism_mesh(xpos, ypos, zpos, voxelsizes[0], voxelsizes[1], voxelsizes[2], color = get_color_from_value(img_data[x, y, z]))
                    else:
                        mesh += generate_rectangular_prism_mesh(xpos, ypos, zpos, voxelsizes[0], voxelsizes[1], voxelsizes[2], color = get_color_from_value(img_data[x, y, z]))
    
    # Save metadata to use for colorbar later
    mesh.field_data['threshold_ratio'] = np.array([threshold / maxval])
    mesh.field_data['scalemax_ratio'] = np.array([scalemax / maxval])
    
    return mesh

def get_color_from_value(value, color_map = 'plasma_r', reverse = False):
    cmap = plt.get_cmap(color_map)
    if reverse:
        value = 1 - value
    return cmap(value)

#def generate_derenzo_mesh(x, y, z, r, h, angle = 0, resolution=64):
    """
    Create a Derenzo mesh with the top rod centered at (x, y, z).
    The remaining rods are generated at angle specified, 0 angle corresponds to typical top down orientation.

    Parameters
    ----------
    x, y, z : float
        Center coordinates of the Derenzo mesh.
    r : float
        Radius of the Derenzo rods.
    h : float
        Height of the Derenzo mods.
    angle : float, optional
        Angle of the Derenzo mesh.
    resolution : int, optional
        Number of facets around the circumference; higher = smoother.
    """
#    pass

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
    out_path = os.path.join(folder, filename)

    # Ensure the 'source_vtps' folder exists
    os.makedirs(folder, exist_ok=True)

    mesh.save(out_path)
    return out_path

