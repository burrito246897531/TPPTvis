"""
Mesh cache utilities for generating and caching VTP files.
Handles creation of scanner mesh and grid mesh with caching.
"""
import os
import pyvista as pv
import numpy as np
import pandas as pd
from tqdm import tqdm


def build_scanner_mesh(csv_path, subsample=1):
    """
    Build the scanner crystal mesh from CSV data.
    
    Parameters
    ----------
    csv_path : str
        Path to CSV file with crystal data
    subsample : int
        Subsample rate for loading data
        
    Returns
    -------
    pyvista.PolyData
        Combined mesh with all crystals, each cell labeled with crystal_id
    """
    map_df = pd.read_csv(csv_path, header=None)[::subsample]
    
    # Precompute face centers and angles
    face_center_rows = []
    for idx, row in map_df.iterrows():
        face_center = np.asarray([row.iloc[0], row.iloc[1], row.iloc[2]])
        angle = row[3]
        face_center_rows.append((idx, face_center, angle))
    
    combined_mesh = None
    crystal = pv.Box(bounds=(0, 15, -1.5, 1.5, -1.5, 1.5))
    
    for idx, face_center, angle in tqdm(face_center_rows, desc="Building scanner mesh"):
        crystal_copy = crystal.copy()
        # Label cells with crystal index for picking/hover
        crystal_copy.cell_data['crystal_id'] = np.full(crystal_copy.n_cells, idx, dtype=int)
        # Translate crystal to face center
        crystal_copy.translate(face_center, inplace=True)
        # Rotate crystal around z axis by angle
        crystal_copy.rotate_vector(np.array([0, 0, 1]), angle, point=face_center, inplace=True)
        # Add crystal to combined mesh
        combined_mesh = crystal_copy if combined_mesh is None else combined_mesh + crystal_copy
    
    return combined_mesh


def build_grid_mesh(z_level=-55, axis_range=160, tick_spacing=20):
    """
    Build the XY coordinate plane grid mesh.
    
    Parameters
    ----------
    z_level : float
        Z coordinate for the grid plane
    axis_range : int
        Range of axes (grid extends from -axis_range to +axis_range)
    tick_spacing : int
        Spacing between grid lines
        
    Returns
    -------
    pyvista.PolyData
        Combined mesh with all grid lines
    """
    grid_lines = []
    
    # Draw grid lines every tick_spacing units
    # Vertical lines (parallel to Y axis)
    for x in range(-axis_range, axis_range + 1, tick_spacing):
        line = pv.Line(pointa=(x, -axis_range, z_level), 
                      pointb=(x, axis_range, z_level))
        grid_lines.append(line)
    
    # Horizontal lines (parallel to X axis)
    for y in range(-axis_range, axis_range + 1, tick_spacing):
        line = pv.Line(pointa=(-axis_range, y, z_level), 
                      pointb=(axis_range, y, z_level))
        grid_lines.append(line)
    
    # Draw axes (thicker lines) - X axis
    x_axis = pv.Line(pointa=(-axis_range, 0, z_level), 
                    pointb=(axis_range, 0, z_level))
    grid_lines.append(x_axis)
    
    # Y axis
    y_axis = pv.Line(pointa=(0, -axis_range, z_level), 
                    pointb=(0, axis_range, z_level))
    grid_lines.append(y_axis)
    
    # Combine all lines into a single mesh
    combined_grid = None
    for line in grid_lines:
        combined_grid = line if combined_grid is None else combined_grid + line
    
    return combined_grid


def get_scanner_mesh(csv_path, vtp_path=None, subsample=1, force_rebuild=False):
    """
    Get scanner mesh, loading from cache if available, otherwise building and caching it.
    
    Parameters
    ----------
    csv_path : str
        Path to CSV file with crystal data
    vtp_path : str, optional
        Path to cached VTP file. If None, uses 'tpptscanner.vtp' in the same directory as csv_path
    subsample : int
        Subsample rate for loading data
    force_rebuild : bool
        If True, rebuilds the mesh even if cache exists
        
    Returns
    -------
    pyvista.PolyData or None
        Scanner mesh, or None if loading/building failed
    """
    if vtp_path is None:
        vtp_path = os.path.join(os.path.dirname(csv_path), 'tpptscanner.vtp')
    
    # Try loading cached mesh
    if not force_rebuild and os.path.exists(vtp_path):
        try:
            cached_mesh = pv.read(vtp_path)
            if 'crystal_id' in cached_mesh.cell_data:
                return cached_mesh
            else:
                print(f"Cached mesh at {vtp_path} missing 'crystal_id'; rebuilding.")
        except Exception as exc:
            print(f"Failed to load cached mesh at {vtp_path}: {exc}")
    
    # Build mesh if no valid cache was found
    try:
        combined_mesh = build_scanner_mesh(csv_path, subsample=subsample)
        
        # Save built mesh for faster subsequent launches
        try:
            combined_mesh.save(vtp_path)
        except Exception as exc:
            print(f"Warning: could not save cached mesh to {vtp_path}: {exc}")
        
        return combined_mesh
    except Exception as exc:
        print(f"Failed to build scanner mesh: {exc}")
        return None


def get_grid_mesh(vtp_path=None, z_level=-55, axis_range=160, tick_spacing=20, force_rebuild=False):
    """
    Get grid mesh, loading from cache if available, otherwise building and caching it.
    
    Parameters
    ----------
    vtp_path : str, optional
        Path to cached VTP file. If None, uses 'grid.vtp' in the current directory
    z_level : float
        Z coordinate for the grid plane
    axis_range : int
        Range of axes (grid extends from -axis_range to +axis_range)
    tick_spacing : int
        Spacing between grid lines
    force_rebuild : bool
        If True, rebuilds the mesh even if cache exists
        
    Returns
    -------
    pyvista.PolyData or None
        Grid mesh, or None if loading/building failed
    """
    if vtp_path is None:
        # Use the directory of this file as default
        vtp_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'grid.vtp')
    
    # Try loading cached grid mesh
    if not force_rebuild and os.path.exists(vtp_path):
        try:
            grid_mesh = pv.read(vtp_path)
            return grid_mesh
        except Exception as exc:
            print(f"Failed to load cached grid at {vtp_path}: {exc}")
    
    # Build grid mesh if no valid cache was found
    try:
        grid_mesh = build_grid_mesh(z_level=z_level, axis_range=axis_range, tick_spacing=tick_spacing)
        
        # Save grid mesh for faster subsequent launches
        try:
            grid_mesh.save(vtp_path)
        except Exception as exc:
            print(f"Warning: could not save cached grid to {vtp_path}: {exc}")
        
        return grid_mesh
    except Exception as exc:
        print(f"Failed to build grid mesh: {exc}")
        return None

