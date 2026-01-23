import os
import pyvista as pv


def generate_cylinder_mesh(x, y, z, r, h, direction=(0, 0, 1), resolution=64):
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

    Returns
    -------
    pyvista.PolyData
        Cylinder mesh.
    """
    # True x, y, z are center minus direction vector times half the height
    x = x + direction[0] * h / 2
    y = y + direction[1] * h / 2
    z = z + direction[2] * h / 2
    # Generate a cylinder mesh
    return pv.Cylinder(center=(x, y, z), direction=direction, radius=r, height=h, resolution=resolution)


def generate_rectangular_prism_mesh(x, y, z, l, w, h):
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
    return pv.Box(bounds=bounds)

def generate_derenzo_mesh(x, y, z, r, h, angle = 0, resolution=64):
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
    pass

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

