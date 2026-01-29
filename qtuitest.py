"""
Qt UI example with PyVista integration
Includes: Menu bar, Sidebar, and PyVista plotter window
"""
import os
import sys
import shutil
import pyvista as pv
import numpy as np
import scannertest as st
import vtk

try:
    from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                                    QHBoxLayout, QMenuBar, QMenu, QDockWidget, 
                                    QListWidget, QPushButton, QLabel, QTextEdit, 
                                    QListWidgetItem, QScrollArea, QFileDialog, QMessageBox,
                                    QSpinBox)
    from PySide6.QtCore import Qt, QTimer
    from pyvistaqt import QtInteractor
    QT_AVAILABLE = True
except ImportError as e:
    print(f"Error: Qt dependencies not installed. Please run: pip install PySide6 pyvistaqt")
    print(f"Import error: {e}")
    QT_AVAILABLE = False

print('Loading...')

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("TPPT Visualization")
        self.setGeometry(100, 100, 1400, 900)
        
        # Create central widget with PyVista plotter
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Create PyVista Qt interactor
        self.plotter = QtInteractor(central_widget)
        layout.addWidget(self.plotter)
        
        # Rotation lock state
        self.z_axis_lock = False
        
        # Event counts storage
        self.event_counts = None
        
        # Track clear button connection
        self._clear_button_connected = False
        
        # Store loaded LOR data for re-rendering when top N changes
        self.loaded_lor_data = None

        # Debounce timer for top-N LOR rendering
        self.top_n_timer = QTimer(self)
        self.top_n_timer.setSingleShot(True)
        self.top_n_timer.timeout.connect(self.update_top_lors_display_now)
        
        # Create menu bar
        self.create_menu_bar()
        
        # Create sidebar (must be before visualization to set up info label)
        self.create_sidebar()
        
        # Setup PyVista visualization (after sidebar so callback can reference sidebar widgets)
        self.setup_visualization()
        
    def setup_visualization(self):
        """Setup initial PyVista visualization"""
        # Define callback to update sidebar with crystal info
        def update_crystal_info(crystal_id, event_count):
            if crystal_id is not None:
                info_text = f"Crystal ID: {crystal_id}\n"
                if event_count is not None:
                    info_text += f"Event Count: {event_count:,}"
                self.crystal_info_label.setText(info_text)
            else:
                self.crystal_info_label.setText("")
        
        # Store delete functions
        self.delete_connection_func = None
        self.clear_all_connections_func = None
        
        # Define callback to update sidebar with connections list
        def update_connections_list(connections_list):
            self.connections_list.clear()
            
            for conn in connections_list:
                crystal_id1, crystal_id2 = conn['crystal_ids']
                count = conn.get('count', 0)
                if count is None:
                    count = 0
                
                # Create item text
                item_text = f"{crystal_id1}, {crystal_id2} | Count: {count}"
                
                # Create list item WITHOUT text (we'll use custom widget instead)
                item = QListWidgetItem()
                item.setData(Qt.UserRole, conn['id'])  # Store connection ID
                
                # Create widget with delete button
                item_widget = QWidget()
                item_layout = QHBoxLayout(item_widget)
                item_layout.setContentsMargins(5, 2, 5, 2)
                
                text_label = QLabel(item_text)
                text_label.setWordWrap(True)
                item_layout.addWidget(text_label)
                
                delete_btn = QPushButton("×")
                delete_btn.setFixedSize(16, 16)
                delete_btn.setStyleSheet("""
                    QPushButton {
                        background-color: #f0f0f0;
                        color: white;
                        font-weight: bold;
                        border-radius: 8px;
                        border: 1px solid #cccccc;
                    }
                    QPushButton::hover {
                        background-color: #e57373;
                    }
                """)
                # Use a closure to capture the connection ID
                def make_delete_handler(cid):
                    return lambda: self.delete_connection(cid)
                delete_btn.clicked.connect(make_delete_handler(conn['id']))
                item_layout.addWidget(delete_btn)
                
                self.connections_list.addItem(item)
                self.connections_list.setItemWidget(item, item_widget)
        
        # Define callback to receive delete functions
        def set_delete_functions(delete_func, clear_all_func):
            self.delete_connection_func = delete_func
            self.clear_all_connections_func = clear_all_func
            # Disconnect existing connection if it exists
            if self._clear_button_connected:
                try:
                    self.clear_all_button.clicked.disconnect(self.clear_all_connections)
                except:
                    pass
            # Connect the clear function
            self.clear_all_button.clicked.connect(self.clear_all_connections)
            self._clear_button_connected = True
        
        # Define status callback to update instructions (no-op since instructions label was removed)
        def update_status(message):
            pass  # Instructions label was removed, so this is a no-op
        
        # Load crystal visualization from scannertest with callbacks
        # This returns (update_event_counts_func, reload_sources_func, set_draw_mode_func, render_top_lors_func)
        self.update_event_counts_func, self.reload_sources_func, self.set_draw_mode_func, self.render_top_lors_func = st.setup_crystal_visualization(
            self.plotter, 
            info_callback=update_crystal_info,
            selection_callback=update_connections_list,
            delete_connection_callback=set_delete_functions,
            status_callback=update_status,
            event_counts=None  # Will be initialized as zeros
        )
        
        # Initialize event_counts as None (will be set when importing)
        self.event_counts = None
        
    def create_menu_bar(self):
        """Create menu bar with menus"""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu("File")
        #file_menu.addAction("New", self.file_new)
        #file_menu.addAction("Open", self.file_open)
        file_menu.addSeparator()
        file_menu.addAction("Import Source VTPs...", self.import_source_vtps)
        file_menu.addAction("Import Event Counts...", self.import_event_counts)
        file_menu.addAction("Import Top LORs...", self.import_top_lors)
        file_menu.addSeparator()
        #file_menu.addAction("Exit", self.close)
        
        # View menu
        view_menu = menubar.addMenu("View")
        view_menu.addAction("Reset View", self.view_reset)
        view_menu.addAction("Zoom In", self.view_zoom_in)
        view_menu.addAction("Zoom Out", self.view_zoom_out)
        view_menu.addSeparator()
        #self.z_axis_lock_action = view_menu.addAction("Lock Rotation to Z-Axis")
        #self.z_axis_lock_action.setCheckable(True)
        #self.z_axis_lock_action.triggered.connect(self.toggle_z_axis_lock)
        
        # Tools menu
        #tools_menu = menubar.addMenu("Tools")
        #tools_menu.addAction("Tool 1", self.tool_1)
        #tools_menu.addAction("Tool 2", self.tool_2)
        
        # Help menu
        #help_menu = menubar.addMenu("Help")
        #help_menu.addAction("About", self.help_about)
        
    def create_sidebar(self):
        """Create sidebar dock widget"""
        # Create dock widget for sidebar
        sidebar_dock = QDockWidget(self)
        sidebar_dock.setAllowedAreas(Qt.LeftDockWidgetArea)
        # Lock sidebar in place - disable moving, floating, and closing
        sidebar_dock.setFeatures(QDockWidget.DockWidgetFeature.NoDockWidgetFeatures)
        
        # Create sidebar content
        sidebar_widget = QWidget()
        sidebar_layout = QVBoxLayout(sidebar_widget)
    
        self.crystal_info_label = QLabel()
        self.crystal_info_label.setStyleSheet("background-color: #f0f0f0; padding: 5px; border: 1px solid #ccc; border-radius: 4px;")
        self.crystal_info_label.setWordWrap(True)
        self.crystal_info_label.setMinimumHeight(40)
        sidebar_layout.addWidget(self.crystal_info_label)
        
        # Connections section
        connections_label = QLabel("Channel Pairs")
        connections_label.setStyleSheet("font-weight: bold; font-size: 12px; padding: 5px;")
        sidebar_layout.addWidget(connections_label)
        
        # Connections list widget
        self.connections_list = QListWidget()
        self.connections_list.setMaximumHeight(200)
        sidebar_layout.addWidget(self.connections_list)
        
        # Buttons and controls for connections
        connections_button_layout = QHBoxLayout()
        self.clear_all_button = QPushButton("Clear")
        self.clear_all_button.setStyleSheet("background-color: #f0f0f0; color: black; padding: 5px;")
        connections_button_layout.addWidget(self.clear_all_button)

        # Draw mode toggle button
        self.draw_mode_enabled = False
        self.draw_mode_button = QPushButton("Draw Mode: Off")
        self.draw_mode_button.setCheckable(True)
        self.draw_mode_button.setStyleSheet("background-color: #f0f0f0; color: black; padding: 5px;")
        self.draw_mode_button.clicked.connect(self.toggle_draw_mode)
        connections_button_layout.addWidget(self.draw_mode_button)

        sidebar_layout.addLayout(connections_button_layout)

        # Top LORs controls
        lor_controls_layout = QHBoxLayout()
        self.import_lors_button = QPushButton("Import Top LORs")
        self.import_lors_button.setStyleSheet("background-color: #f0f0f0; color: black; padding: 5px;")
        self.import_lors_button.clicked.connect(self.import_top_lors)
        lor_controls_layout.addWidget(self.import_lors_button)

        top_label = QLabel("Top")
        lor_controls_layout.addWidget(top_label)

        self.top_n_spin = QSpinBox()
        self.top_n_spin.setRange(1, 1000)
        self.top_n_spin.setValue(100)
        self.top_n_spin.valueChanged.connect(lambda _: self.schedule_top_lors_render())
        lor_controls_layout.addWidget(self.top_n_spin)

        sidebar_layout.addLayout(lor_controls_layout)

        # Add stretch to push widgets to top
        sidebar_layout.addStretch()
        
        sidebar_dock.setWidget(sidebar_widget)
        self.addDockWidget(Qt.LeftDockWidgetArea, sidebar_dock)
        
    # Menu action handlers (empty for now)
    def file_new(self):
        print("File -> New")
        
    def file_open(self):
        print("File -> Open")
    
    def import_event_counts(self):
        """Import event counts from binary file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Import Event Counts",
            "",
            "Binary Files (*.bin *.dat);;All Files (*.*)"
        )
        
        if file_path:
            try:
                # Read binary file: 6144 int32 values
                with open(file_path, 'rb') as f:
                    # Read 6144 int32 values (4 bytes each)
                    data = f.read(6144 * 4)
                    if len(data) < 6144 * 4:
                        QMessageBox.warning(
                            self,
                            "Import Error",
                            f"File is too short. Expected {6144 * 4} bytes, got {len(data)} bytes."
                        )
                        return
                    
                    # Convert to numpy array of int32
                    event_counts = np.frombuffer(data, dtype=np.int32)
                    
                    if len(event_counts) != 6144:
                        QMessageBox.warning(
                            self,
                            "Import Error",
                            f"Unexpected number of values. Expected 6144, got {len(event_counts)}."
                        )
                        return
                    
                    # Update event counts without re-rendering
                    if self.update_event_counts_func:
                        self.update_event_counts_func(event_counts)
                        self.event_counts = event_counts
                        QMessageBox.information(
                            self,
                            "Import Successful",
                            f"Successfully imported event counts for {len(event_counts)} crystals."
                        )
                    else:
                        QMessageBox.warning(
                            self,
                            "Import Error",
                            "Visualization not initialized. Please restart the application."
                        )
                    
            except Exception as e:
                QMessageBox.critical(
                    self,
                    "Import Error",
                    f"Failed to import event counts:\n{str(e)}"
                )

    def import_source_vtps(self):
        """Import one or more source VTP files and reload the scene."""
        file_paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Import Source VTPs",
            "",
            "VTP Files (*.vtp);;All Files (*.*)"
        )

        if not file_paths:
            return

        dest_dir = os.path.join(os.path.dirname(__file__), "source_vtps")
        os.makedirs(dest_dir, exist_ok=True)

        copied = 0
        errors = []
        copied_paths = []
        for fpath in file_paths:
            try:
                fname = os.path.basename(fpath)
                dest = os.path.join(dest_dir, fname)
                # If destination already exists and is the same file, count it as imported and keep using it
                if os.path.exists(dest):
                    try:
                        if os.path.samefile(fpath, dest):
                            copied += 1
                            copied_paths.append(dest)
                            continue
                    except OSError:
                        # If samefile check fails (e.g., across filesystems), proceed with copy/overwrite
                        pass
                # Copy or overwrite the file
                shutil.copyfile(fpath, dest)
                copied += 1
                copied_paths.append(dest)
            except Exception as exc:
                errors.append(f"{fpath}: {exc}")

        # Reload sources on success
        if copied > 0 and getattr(self, "reload_sources_func", None):
            try:
                self.reload_sources_func(selected_paths=copied_paths)
            except Exception as exc:
                errors.append(f"Reload failed: {exc}")

        if errors:
            QMessageBox.warning(
                self,
                "Import Sources",
                f"Imported {copied} file(s) with some errors:\n" + "\n".join(errors)
            )
        else:
            QMessageBox.information(
                self,
                "Import Sources",
                f"Successfully imported {copied} source VTP file(s)."
            )

    def import_top_lors(self):
        """Import a Top LORs int16 binary file and render top-N pairs (expects 1000x3)."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Import Top LORs",
            "",
            "Binary Files (*.bin *.dat);;All Files (*.*)"
        )

        if not file_path:
            return

        try:
            with open(file_path, "rb") as f:
                raw = f.read()

            arr = np.frombuffer(raw, dtype=np.int16)
            if arr.size < 3000:
                raise ValueError(f"File too small for 1000x3 int16 entries (got {arr.size} values).")
            # Trim to first 1000 rows (each row 3 values) and reshape
            arr = arr[:3000].reshape(1000, 3).astype(np.int32)

            # Store the loaded data for re-rendering when top N changes
            self.loaded_lor_data = arr.tolist()
            # Render with current top N value (debounced render)
            self.schedule_top_lors_render(delay_ms=0)
            QMessageBox.information(
                self,
                "Import Top LORs",
                f"Successfully imported Top LORs data."
            )
        except Exception as exc:
            QMessageBox.critical(
                self,
                "Import Top LORs Error",
                f"Failed to import Top LORs:\n{exc}"
            )
    
    def schedule_top_lors_render(self, delay_ms=500):
        """Debounce rendering of top LORs."""
        if self.loaded_lor_data is None:
            return
        self.top_n_timer.start(max(0, int(delay_ms)))

    def update_top_lors_display_now(self):
        """Update the displayed LOR pairs based on current top N value."""
        if self.loaded_lor_data is None:
            return  # No data loaded yet
        
        if not hasattr(self, "render_top_lors_func") or not self.render_top_lors_func:
            return  # Render function not available
        
        try:
            top_n = int(self.top_n_spin.value())
            self.render_top_lors_func(self.loaded_lor_data, top_n=top_n, clear_existing=True, right_offset=True)
        except Exception as e:
            print(f"Error updating Top LORs display: {e}")

        
    def view_reset(self):
        if self.plotter:
            self.plotter.camera_position = 'iso'
            self.plotter.render()
        
    def view_zoom_in(self):
        if self.plotter:
            self.plotter.camera.zoom(1.2)
            self.plotter.render()
            
    def view_zoom_out(self):
        if self.plotter:
            self.plotter.camera.zoom(0.8)
            self.plotter.render()
    
    def toggle_z_axis_lock(self, checked):
        """Toggle Z-axis rotation lock"""
        self.z_axis_lock = checked
        self.setup_z_axis_lock()
    
    def setup_z_axis_lock(self):
        """Setup or remove Z-axis rotation lock"""
        if not hasattr(self, 'plotter') or self.plotter is None:
            return
        
        # Remove existing custom style if any
        if hasattr(self, '_original_style') and self._original_style is not None:
            self.plotter.iren.SetInteractorStyle(self._original_style)
            self._original_style = None
        
        if self.z_axis_lock:
            # Store original style
            self._original_style = self.plotter.iren.get_interactor_style()
            
            # Create custom style that only allows Z-axis rotation
            class ZAxisLockStyle(vtk.vtkInteractorStyleTrackballCamera):
                def __init__(self, parent_window):
                    vtk.vtkInteractorStyleTrackballCamera.__init__(self)
                    self.parent = parent_window
                
                def Rotate(self):
                    """Override Rotate to only allow azimuth (Z-axis) rotation"""
                    if self.parent.z_axis_lock:
                        # Get current camera
                        camera = self.parent.plotter.camera
                        focal_point = np.array(camera.focal_point)
                        position = np.array(camera.position)
                        
                        # Calculate current view vector
                        view_vector = position - focal_point
                        distance = np.linalg.norm(view_vector)
                        
                        if distance < 0.01:
                            return
                        
                        # Get mouse movement
                        rwi = self.plotter.iren.get_interactor()
                        dx = rwi.GetEventPosition()[0] - rwi.GetLastEventPosition()[0]
                        dy = rwi.GetEventPosition()[1] - rwi.GetLastEventPosition()[1]
                        
                        # Only use horizontal movement (azimuth)
                        # Convert to rotation angle (azimuth only)
                        size = rwi.GetRenderWindow().GetSize()
                        if size[0] > 0:
                            rotation_factor = 10.0 / size[0]
                            angle = dx * rotation_factor
                            
                            # Rotate around Z-axis only
                            # Project view vector to XY plane
                            view_xy = np.array([view_vector[0], view_vector[1], 0])
                            if np.linalg.norm(view_xy) < 0.01:
                                view_xy = np.array([1, 0, 0])
                            else:
                                view_xy = view_xy / np.linalg.norm(view_xy)
                            
                            # Calculate current azimuth angle
                            current_azimuth = np.arctan2(view_xy[1], view_xy[0])
                            new_azimuth = current_azimuth + np.radians(angle)
                            
                            # Calculate new XY direction
                            new_view_xy = np.array([np.cos(new_azimuth), np.sin(new_azimuth), 0])
                            
                            # Maintain Z distance
                            z_distance = view_vector[2]
                            xy_distance = np.linalg.norm(view_vector[:2])
                            
                            # Reconstruct position
                            new_position = focal_point + new_view_xy * xy_distance + np.array([0, 0, z_distance])
                            camera.position = tuple(new_position)
                            
                            # Force view up to be Z-axis
                            camera.up = (0, 0, 1)
                            
                            # Update render
                            rwi.Render()
                    else:
                        # Normal rotation if lock is off
                        vtk.vtkInteractorStyleTrackballCamera.Rotate(self)
            
            # Create and set custom style
            custom_style = ZAxisLockStyle(self)
            self.plotter.iren.SetInteractorStyle(custom_style)
            
            # Apply constraint immediately
            camera = self.plotter.camera
            camera.up = (0, 0, 1)
            self.plotter.render()
        else:
            # Restore original style
            if hasattr(self, '_original_style') and self._original_style is not None:
                self.plotter.iren.SetInteractorStyle(self._original_style)
                self._original_style = None
            
    def tool_1(self):
        print("Tool 1")
        
    def tool_2(self):
        print("Tool 2")
        
    def help_about(self):
        print("About dialog")
        
    # Sidebar action handlers (empty for now)
    def sidebar_button1(self):
        print("Sidebar Button 1 clicked")
        
    def sidebar_button2(self):
        print("Sidebar Button 2 clicked")
    
    def delete_connection(self, connection_id):
        """Delete a specific connection"""
        if self.delete_connection_func:
            self.delete_connection_func(connection_id)
    
    def clear_all_connections(self):
        """Clear all connections"""
        if self.clear_all_connections_func:
            self.clear_all_connections_func()

    def toggle_draw_mode(self):
        """Toggle draw mode for creating channel pairs."""
        self.draw_mode_enabled = not self.draw_mode_enabled
        if hasattr(self, "set_draw_mode_func") and self.set_draw_mode_func:
            try:
                self.set_draw_mode_func(self.draw_mode_enabled)
            except Exception:
                pass
        # Update button text
        self.draw_mode_button.setText("Draw Mode: On" if self.draw_mode_enabled else "Draw Mode: Off")


def main():
    if not QT_AVAILABLE:
        print("Qt dependencies are not available. Please install PySide6 and pyvistaqt.")
        return
        
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

