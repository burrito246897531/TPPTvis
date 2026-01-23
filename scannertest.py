import os
import pyvista as pv
import numpy as np
import vtk
import pandas as pd
from tqdm import tqdm

current_dir = os.path.dirname(os.path.abspath(__file__))

def setup_crystal_visualization(plotter, csv_path=os.path.join(current_dir, 'TPPT_Scanner_map_vis.csv'), subsample=1, info_callback=None, selection_callback=None, delete_connection_callback=None, status_callback=None, event_counts=None, vtp_path=None, source_vtps_dir=None, draw_mode_enabled=False):
    """Setup crystal visualization on the given plotter
    
    Parameters:
    -----------
    plotter : pv.Plotter
        PyVista plotter instance
    csv_path : str
        Path to CSV file with crystal data
    subsample : int
        Subsample rate for loading data
    info_callback : callable, optional
        Callback function(crystal_id, event_count) called when hovering over a crystal.
        If None, displays info in PyVista window (standalone mode).
    event_counts : dict or array, optional
        Event counts per crystal. If dict, keys are crystal IDs. If array/list, indexed by crystal ID.
    selection_callback : callable, optional
        Callback function(connections_list) called when connections are updated.
        connections_list is a list of dicts with keys: 'id', 'crystal_ids', 'face_centers'.
    delete_connection_callback : callable, optional
        Callback function(delete_func, clear_all_func) to expose delete functions to UI.
        delete_func(connection_id) deletes a specific connection.
        clear_all_func() clears all connections.
    vtp_path : str, optional
        Path to cached VTP mesh. If present, load instead of rebuilding. If missing,
        the mesh is generated and saved to this path.
    source_vtps_dir : str, optional
        Directory containing source VTPs to render on top of the scanner. Defaults to
        a 'source_vtps' folder next to this script.
    draw_mode_enabled : bool, optional
        If True, click-to-connect mode is active. Can be toggled at runtime via set_draw_mode.
    """
    map_df = pd.read_csv(csv_path, header=None)[::subsample]
    num_crystals = len(map_df)
    
    # Resolve cache path for the pre-built mesh
    if vtp_path is None:
        vtp_path = os.path.join(os.path.dirname(__file__), 'tpptscanner.vtp')
    if source_vtps_dir is None:
        source_vtps_dir = os.path.join(os.path.dirname(__file__), 'source_vtps')
    
    combined_mesh = None
    hovered_crystal = None
    hover_scalars = None
    selected_crystals = []  # Track selected crystal IDs (temporary, for creating connections)
    face_centers_dict = {}  # Map crystal_id to face_center
    connections = []  # List of connections: [{'id': int, 'crystal_ids': [id1, id2], 'face_centers': [center1, center2], 'actor': actor}, ...]
    connection_counter = 0  # Counter for unique connection IDs
    source_mesh_actors = []  # Track loaded source meshes so we can refresh
    draw_mode = draw_mode_enabled
    
    # Precompute face centers and angles for building connections or meshes
    face_center_rows = []
    for idx, row in map_df.iterrows():
        face_center = np.asarray([row.iloc[0], row.iloc[1], row.iloc[2]])
        angle = row[3]
        face_centers_dict[idx] = face_center
        face_center_rows.append((idx, face_center, angle))
    
    # Try loading cached mesh to avoid rebuild
    if os.path.exists(vtp_path):
        try:
            cached_mesh = pv.read(vtp_path)
            if 'crystal_id' in cached_mesh.cell_data:
                combined_mesh = cached_mesh
            else:
                print(f"Cached mesh at {vtp_path} missing 'crystal_id'; rebuilding.")
        except Exception as exc:
            print(f"Failed to load cached mesh at {vtp_path}: {exc}")
    
    # Initialize event_counts as array of zeros if not provided
    if event_counts is None:
        event_counts = np.zeros(num_crystals, dtype=np.int32)
    elif isinstance(event_counts, (list, np.ndarray)):
        # Ensure it's the right size
        if len(event_counts) < num_crystals:
            # Pad with zeros if too short
            event_counts = np.pad(event_counts, (0, num_crystals - len(event_counts)), 'constant')
        elif len(event_counts) > num_crystals:
            # Truncate if too long
            event_counts = event_counts[:num_crystals]
        event_counts = np.asarray(event_counts, dtype=np.int32)
    
    # Build mesh if no valid cache was found
    if combined_mesh is None:
        crystal = pv.Box(bounds=(0, 15, -1.5, 1.5, -1.5, 1.5))
        
        for idx, face_center, angle in tqdm(face_center_rows):
            crystal_copy = crystal.copy()
            # Label cells with crystal index for picking/hover
            crystal_copy.cell_data['crystal_id'] = np.full(crystal_copy.n_cells, idx, dtype=int)
            # Translate crystal to face center
            crystal_copy.translate(face_center, inplace=True)
            # Rotate crystal around z axis by angle
            crystal_copy.rotate_vector(np.array([0, 0, 1]), angle, point=face_center, inplace=True)
            # Add crystal to combined mesh
            combined_mesh = crystal_copy if combined_mesh is None else combined_mesh + crystal_copy
        
        # Save built mesh for faster subsequent launches
        try:
            combined_mesh.save(vtp_path)
        except Exception as exc:
            print(f"Warning: could not save cached mesh to {vtp_path}: {exc}")

    def load_source_vtps(selected_paths=None):
        """
        Load and render VTP sources.

        Parameters
        ----------
        selected_paths : list[str] or None
            If provided, only these files are loaded. Otherwise, all .vtp files
            in source_vtps_dir are loaded.
        """
        nonlocal source_mesh_actors
        # Remove any previously added source actors
        for actor in source_mesh_actors:
            try:
                plotter.remove_actor(actor)
            except Exception:
                pass
        source_mesh_actors = []

        paths = []
        if selected_paths is not None:
            paths = list(selected_paths)
        else:
            if not os.path.isdir(source_vtps_dir):
                return
            for fname in sorted(os.listdir(source_vtps_dir)):
                if not fname.lower().endswith('.vtp'):
                    continue
                paths.append(os.path.join(source_vtps_dir, fname))

        for fpath in paths:
            if not fpath.lower().endswith('.vtp'):
                continue
            try:
                mesh = pv.read(fpath)
                actor = plotter.add_mesh(
                    mesh,
                    color='orange',
                    opacity=0.35,
                    show_edges=False,
                    name=f"source_{os.path.basename(fpath)}"
                )
                source_mesh_actors.append(actor)
            except Exception as exc:
                print(f"Failed to load source VTP {fpath}: {exc}")

        plotter.render()

    
    # Create event count scalars: map each cell's crystal_id to its event count
    event_count_scalars = np.zeros(combined_mesh.n_cells, dtype=float)
    for i in range(combined_mesh.n_cells):
        crystal_id = int(combined_mesh.cell_data['crystal_id'][i])
        if crystal_id < len(event_counts):
            event_count_scalars[i] = float(event_counts[crystal_id])
    
    combined_mesh.cell_data['event_count'] = event_count_scalars
    
    # Store original event count scalars for hover effect
    original_event_count_scalars = event_count_scalars.copy()
    
    # Hover scalars for darkening effect (multiplier)
    hover_scalars = np.ones(combined_mesh.n_cells, dtype=float)
    combined_mesh.cell_data['hover_scalar'] = hover_scalars
    
    # Determine color range from event counts (will be updated when counts change)
    event_count_min = 0.0
    event_count_max = float(np.max(event_counts)) if len(event_counts) > 0 else 1.0
    if event_count_max == 0:
        event_count_max = 1.0  # Avoid division by zero
    
    combined_actor = plotter.add_mesh(
        combined_mesh,
        scalars='event_count',  # Use event counts for coloring
        clim=[event_count_min, event_count_max],
        cmap='inferno',  # Heatmap colormap (can be changed to 'plasma', 'inferno', 'magma', etc.)
        opacity=0.05,
        show_edges=True,
        edge_color='black',
        line_width=1,
        pickable=True,
        show_scalar_bar=False,  # Hide color bar
        name='combined_prism'
    )
    
    # Hover handler to show crystal id and darken on hover
    def on_mouse_move(obj, event):
        nonlocal hovered_crystal, event_count_scalars, original_event_count_scalars
        if combined_mesh is None:
            return
        interactor = obj
        mouse_pos = interactor.GetEventPosition()
        picker = vtk.vtkCellPicker()
        picker.SetTolerance(0.0005)
        picker.Pick(mouse_pos[0], mouse_pos[1], 0, plotter.renderer)
        picked_actor = picker.GetActor()
        
        if picked_actor is not None and picked_actor == combined_actor:
            cell_id = picker.GetCellId()
            if 0 <= cell_id < combined_mesh.n_cells:
                crystal_id = int(combined_mesh.cell_data['crystal_id'][cell_id])
                if crystal_id != hovered_crystal:
                    # Darken hovered crystal cells by reducing their event count scalar (for visual effect)
                    # Restore all to original first
                    event_count_scalars[:] = original_event_count_scalars
                    mask = combined_mesh.cell_data['crystal_id'] == crystal_id
                    # Darken by reducing the scalar value (multiply by 0.5 for darkening effect)
                    event_count_scalars[mask] = original_event_count_scalars[mask] * 0.5
                    combined_mesh.cell_data['event_count'] = event_count_scalars
                    
                    mapper = combined_actor.GetMapper()
                    mapper.SetInputData(combined_mesh)
                    mapper.SetScalarModeToUseCellData()
                    mapper.SelectColorArray('event_count')
                    mapper.Update()
                    
                    # Get event count if available
                    event_count = None
                    if crystal_id < len(event_counts):
                        event_count = int(event_counts[crystal_id])
                    
                    # Call callback if provided, otherwise show in PyVista window (standalone mode)
                    if info_callback is not None:
                        info_callback(crystal_id, event_count)
                    else:
                        pos = picker.GetPickPosition()
                        text = f"ID: {crystal_id}"
                        if event_count is not None:
                            text += f"\nEvents: {event_count}"
                        text_actor = plotter.add_text(
                            text,
                            position='upper_right',
                            font_size=11,
                            name='hover_info'
                        )
                        # Use a supported VTK font family (Arial/Courier/Times)
                        text_actor.GetTextProperty().SetFontFamilyToArial()
                    
                    hovered_crystal = crystal_id
                    plotter.render()
                return
        
        # Clear hover info when not over a crystal
        if hovered_crystal is not None:
            # Restore original event count scalars
            event_count_scalars[:] = original_event_count_scalars
            combined_mesh.cell_data['event_count'] = event_count_scalars
            mapper = combined_actor.GetMapper()
            mapper.SetInputData(combined_mesh)
            mapper.SetScalarModeToUseCellData()
            mapper.SelectColorArray('event_count')
            mapper.Update()
            
            # Clear info display
            if info_callback is not None:
                info_callback(None, None)  # Signal to clear
            else:
                plotter.remove_actor('hover_info')
            
            hovered_crystal = None
            plotter.render()
    
    # Function to add a new connection
    def add_connection(crystal_id1, crystal_id2, count=None):
        nonlocal connection_counter
        if crystal_id1 == crystal_id2:
            return  # Can't connect a crystal to itself
        
        # Check if connection already exists
        for conn in connections:
            if (conn['crystal_ids'][0] == crystal_id1 and conn['crystal_ids'][1] == crystal_id2) or \
               (conn['crystal_ids'][0] == crystal_id2 and conn['crystal_ids'][1] == crystal_id1):
                return  # Connection already exists
        
        face_center1 = face_centers_dict[crystal_id1]
        face_center2 = face_centers_dict[crystal_id2]
        
        # Create line between the two face centers
        line = pv.Line(face_center1, face_center2)
        line_actor = plotter.add_mesh(
            line,
            color='red',
            line_width=3,
            name=f'connection_line_{connection_counter}'
        )
        
        # Store connection
        connection_id = connection_counter
        connection_counter += 1
        connection = {
            'id': connection_id,
            'crystal_ids': [crystal_id1, crystal_id2],
            'face_centers': [face_center1, face_center2],
            'actor': line_actor,
            'count': count
        }
        connections.append(connection)
        
        # Update UI
        if selection_callback is not None:
            update_connections_list()
        
        return connection_id
    
    # Function to delete a connection by ID
    def delete_connection(connection_id):
        for i, conn in enumerate(connections):
            if conn['id'] == connection_id:
                plotter.remove_actor(conn['actor'])
                connections.pop(i)
                if selection_callback is not None:
                    update_connections_list()
                plotter.render()
                return True
        return False
    
    # Function to clear all connections
    def clear_all_connections():
        for conn in connections:
            plotter.remove_actor(conn['actor'])
        connections.clear()
        if selection_callback is not None:
            update_connections_list()
        plotter.render()
    
    # Function to update connections list in UI
    def update_connections_list():
        if selection_callback is not None:
            conn_list = []
            for conn in connections:
                conn_list.append({
                    'id': conn['id'],
                    'crystal_ids': conn['crystal_ids'],
                    'face_centers': conn['face_centers'],
                    'count': conn.get('count')
                })
            selection_callback(conn_list)
    
    # Expose delete function to external callbacks
    if delete_connection_callback is not None:
        delete_connection_callback(delete_connection, clear_all_connections)
    
    # Click handler to select crystals and create connections
    def on_mouse_click(obj, event):
        nonlocal selected_crystals
        if combined_mesh is None:
            return
        if not draw_mode:
            return  # Draw mode off; ignore clicks for pairing
        interactor = obj
        click_pos = interactor.GetEventPosition()
        picker = vtk.vtkCellPicker()
        picker.SetTolerance(0.0005)
        picker.Pick(click_pos[0], click_pos[1], 0, plotter.renderer)
        picked_actor = picker.GetActor()
        
        if picked_actor is not None and picked_actor == combined_actor:
            cell_id = picker.GetCellId()
            if 0 <= cell_id < combined_mesh.n_cells:
                crystal_id = int(combined_mesh.cell_data['crystal_id'][cell_id])
                
                # If no selection, start with this crystal
                if len(selected_crystals) == 0:
                    selected_crystals = [crystal_id]
                    if status_callback is not None:
                        status_callback(f"Selected Crystal {crystal_id}. Click another crystal to connect.")
                # If one selection, create connection and clear selection
                elif len(selected_crystals) == 1:
                    if crystal_id != selected_crystals[0]:
                        add_connection(selected_crystals[0], crystal_id)
                        if status_callback is not None:
                            status_callback("Click two crystals to create a connection")
                    else:
                        if status_callback is not None:
                            status_callback("Cannot connect a crystal to itself. Click another crystal.")
                    selected_crystals = []
                # If somehow we have 2, replace with new selection
                else:
                    selected_crystals = [crystal_id]
                    if status_callback is not None:
                        status_callback(f"Selected Crystal {crystal_id}. Click another crystal to connect.")
                
                plotter.render()
    
    # Function to update event counts (can be called externally)
    def update_event_counts(new_counts):
        nonlocal event_counts, event_count_scalars, original_event_count_scalars, event_count_max
        if isinstance(new_counts, (list, np.ndarray)):
            new_counts = np.asarray(new_counts, dtype=np.int32)
            if len(new_counts) <= num_crystals:
                # Update the event counts array
                event_counts[:len(new_counts)] = new_counts
                if len(new_counts) < num_crystals:
                    # Zero out remaining if new array is shorter
                    event_counts[len(new_counts):] = 0
            else:
                # Truncate if too long
                event_counts[:] = new_counts[:num_crystals]
            
            # Update event count scalars for each cell
            for i in range(combined_mesh.n_cells):
                crystal_id = int(combined_mesh.cell_data['crystal_id'][i])
                if crystal_id < len(event_counts):
                    event_count_scalars[i] = float(event_counts[crystal_id])
            
            # Update original scalars (for hover restoration)
            original_event_count_scalars[:] = event_count_scalars
            
            # Update color range
            event_count_max = float(np.max(event_counts)) if len(event_counts) > 0 else 1.0
            if event_count_max == 0:
                event_count_max = 1.0
            
            # Update mesh data and mapper
            combined_mesh.cell_data['event_count'] = event_count_scalars
            mapper = combined_actor.GetMapper()
            mapper.SetInputData(combined_mesh)
            mapper.SetScalarModeToUseCellData()
            mapper.SelectColorArray('event_count')
            mapper.SetScalarRange(0.0, event_count_max)
            mapper.Update()
            
            # Render to show updated colors
            plotter.render()
            
        return event_counts

    def set_draw_mode(enabled: bool):
        nonlocal draw_mode
        draw_mode = bool(enabled)
        if status_callback is not None:
            status_callback("Draw mode ON" if draw_mode else "Draw mode OFF")

    def render_top_lors(pairs, top_n=100, clear_existing=True, right_offset=True):
        """
        Render top-N LOR pairs as connections.

        Parameters
        ----------
        pairs : iterable of (left_id, right_id, count)
        top_n : int
            Number of pairs to draw.
        clear_existing : bool
            If True, clears current connections before drawing.
        right_offset : bool
            If True, adds 3072 to right_id to map to global crystal IDs.
        """
        try:
            if pairs is None:
                return
            # Normalize to list of tuples and sort by count descending
            norm_pairs = []
            for item in pairs:
                if len(item) < 3:
                    continue
                l_id, r_id, c = int(item[0]), int(item[1]), int(item[2])
                norm_pairs.append((l_id, r_id, c))
            if not norm_pairs:
                return
            norm_pairs.sort(key=lambda x: x[2], reverse=True)
            sel_pairs = norm_pairs[: max(1, int(top_n))]

            if clear_existing:
                clear_all_connections()

            for l_id, r_id, _ in sel_pairs:
                target_r = r_id + 3072 if right_offset else r_id
                # Guard against out-of-range ids
                if l_id < 0 or target_r < 0 or l_id >= len(event_counts) or target_r >= len(event_counts):
                    continue
                add_connection(l_id, target_r, count=_)

            plotter.render()
            if status_callback is not None:
                status_callback(f"Drew top {len(sel_pairs)} channel pairs.")
        except Exception as exc:
            print(f"Failed to render LOR pairs: {exc}")
    
    # Register hover and click observers
    plotter.iren.add_observer("MouseMoveEvent", on_mouse_move)
    plotter.iren.add_observer("LeftButtonPressEvent", on_mouse_click)
    
    # Setup plotter appearance
    plotter.add_axes()
    plotter.set_background('white')
    plotter.camera_position = 'iso'

    # Return helper functions for external use
    return update_event_counts, load_source_vtps, set_draw_mode, render_top_lors


# Standalone execution
if __name__ == "__main__":
    plotter = pv.Plotter()
    setup_crystal_visualization(plotter)
    plotter.show()