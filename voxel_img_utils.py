import numpy as np
import os
import pyvista as pv
import matplotlib.pyplot as plt

def get_color(value, color_map = 'viridis', reverse = False):
    cmap = plt.get_cmap(color_map)
    if reverse:
        value = 1 - value
    rgba = cmap(value)
    rgb = [int(255 * c) for c in rgba[:3]]
    return rgb

def get_castor_mesh_from_img(hdr_path, img_path=None, prop=0.001, trim=0, num_bins=5, color_map='viridis'):
    """Generate a PyVista mesh from a CASToR .hdr/.img file pair.

    Parameters
    ----------
    hdr_path : str
        Full path to the CASToR header (.hdr) file.
    img_path : str, optional
        Full path to the CASToR image (.img) file.
        If None, derived from *hdr_path* by replacing the extension.
    prop : float
        Proportion parameter for percentile thresholding.
    trim : int
        Number of border voxels to trim from each axis.
    num_bins : int
        Number of intensity bins (iso-surfaces).
    color_map : str
        Matplotlib colormap name.

    Returns
    -------
    pyvista.PolyData or None
    """
    if img_path is None:
        img_path = os.path.splitext(hdr_path)[0] + '.img'

    dims = []
    voxelsizes = []
    bytes_per_pixel = 0

    # Read header file for dimensions and voxel sizes
    with open(hdr_path, 'r') as f:
        for line in f:
            if '!matrix size' in line:
                dims.append(int(line[line.index(':=') + 3:-1]))
            elif 'scaling factor' in line:
                voxelsizes.append(float(line[line.index(':=') + 3:-1]))
            elif '!number of bytes per pixel' in line:
                bytes_per_pixel = int(line[line.index(':=') + 3:-1])
    
    # Read image file
    with open(img_path, 'rb') as f:
        img_data = np.fromfile(f, dtype=np.dtype(f'uint{bytes_per_pixel * 8}')).astype(np.float32).reshape(dims, order = 'F')


    # Apply trim
    if trim > 0:
        img_data = img_data[trim:-trim, trim:-trim, trim:-trim]
    dims = img_data.shape
    

    # Calculate percentiles
    percentiles = np.logspace(np.log10(1 - prop), np.log10(1 - prop / 10), num_bins)
    #print(percentiles)
    percentiles = np.percentile(img_data.flatten(), percentiles * 100)
    #print(percentiles)

    voxel_img = pv.ImageData(dimensions=(dims[0] + 1, dims[1] + 1, dims[2] + 1))
    voxel_img.spacing = (voxelsizes[0], voxelsizes[1], voxelsizes[2])
    voxel_img.origin = (-dims[0] * voxelsizes[0] / 2, -dims[1] * voxelsizes[1] / 2, -dims[2] * voxelsizes[2] / 2)
    #voxel_img.origin = (0, 0, 0)
    voxel_img.cell_data["density"] = img_data.ravel(order="F")
    out_mesh = None
    
    for i, lo in enumerate(percentiles):
        region = voxel_img.threshold(lo, scalars="density")
        surf = region.extract_surface()
        alpha = int(255 * (0.2 + 0.7999 * (i / (len(percentiles) - 1))**2))
        surf.cell_data['RGBA'] = np.array([get_color((i+1.0) / len(percentiles), color_map=color_map) + [alpha]] * surf.n_cells, dtype=np.uint8)
        #print(surf.cell_data['RGBA'])
        if out_mesh is None:
            out_mesh = surf
        else:
            out_mesh += surf

    return out_mesh