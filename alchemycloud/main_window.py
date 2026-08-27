"""PyQt5 main window for the LiDAR PCD editor + occupancy-grid exporter.

Single-window layout (Paint/Photoshop-style):
- Edit toolbar      — filters + Crop toggle + crop-mode combo + undo/reset + color axis
- Occupancy toolbar — map generation
- Central widget    — EmbeddedViewer (pyqtgraph GLViewWidget)
- Info bar (status) — file path + cloud stats as a permanent status-bar widget

All 3D rendering happens inside this window; no pop-out viewer.
"""

from __future__ import annotations

import copy
import os

import numpy as np
import open3d as o3d
from PyQt5 import QtCore, QtWidgets

from . import pcd_ops
from .embedded_viewer import EmbeddedViewer, mask_points_in_brush, mask_points_in_rect
from .loading_overlay import LoadingOverlay
from .occupancy_preview import OccupancyPreviewDialog
from .widgets import ToggleSwitch
from .worker import run_async

# ---------------------------------------------------------------------------
# Parameter dialog
# ---------------------------------------------------------------------------


class _ParamDialog(QtWidgets.QDialog):
    def __init__(self, title: str, fields: list[tuple], parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self._widgets = {}
        form = QtWidgets.QFormLayout(self)
        for label, default, kind in fields:
            if kind == "float":
                w = QtWidgets.QDoubleSpinBox()
                w.setDecimals(4)
                w.setRange(-1e9, 1e9)
                w.setSingleStep(0.01)
                w.setValue(float(default))
            elif kind == "int":
                w = QtWidgets.QSpinBox()
                w.setRange(-10_000_000, 10_000_000)
                w.setValue(int(default))
            elif kind.startswith("choice:"):
                w = QtWidgets.QComboBox()
                w.addItems(kind.split(":", 1)[1].split("|"))
                idx = w.findText(str(default))
                if idx >= 0:
                    w.setCurrentIndex(idx)
            elif kind == "bool":
                w = QtWidgets.QCheckBox()
                w.setChecked(bool(default))
            else:
                raise ValueError(f"Unknown field kind: {kind}")
            form.addRow(label, w)
            self._widgets[label] = (w, kind)
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)

    def values(self) -> dict:
        out = {}
        for label, (w, kind) in self._widgets.items():
            if kind == "float":
                out[label] = w.value()
            elif kind == "int":
                out[label] = int(w.value())
            elif kind.startswith("choice:"):
                out[label] = w.currentText()
            elif kind == "bool":
                out[label] = w.isChecked()
        return out


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("LiDAR PCD Editor & Occupancy Grid Exporter")
        self.resize(1200, 800)

        self.original_pcd: o3d.geometry.PointCloud | None = None
        self.current_pcd: o3d.geometry.PointCloud | None = None
        self.history: list[o3d.geometry.PointCloud] = []
        self.redo_stack: list[o3d.geometry.PointCloud] = []
        self.current_path: str | None = None

        self.viewer = EmbeddedViewer(self)
        self.setCentralWidget(self.viewer)
        self.viewer.cropRequested.connect(self._on_crop_requested)
        self.viewer.markerMoved.connect(self._on_marker_moved)
        self.viewer.eraserStrokeFinished.connect(self._on_eraser_stroke)

        self.loading = LoadingOverlay(self.viewer)
        self._busy = False
        self._busy_worker_ref = None  # keep worker alive while running
        # Guards the markerMoved → spin-box → set_marker feedback loop.
        self._syncing_marker_spins = False

        # Debounce timer for Z-Slab spin drags. Without this every spin tick
        # at OS auto-repeat rate triggers a full slab-mask + GL upload.
        self._slab_debounce = QtCore.QTimer(self)
        self._slab_debounce.setSingleShot(True)
        self._slab_debounce.setInterval(80)
        self._slab_debounce.timeout.connect(self._apply_zslab_bounds)

        self._build_menus()
        self._build_toolbars()
        self._build_statusbar()
        self._refresh_state()

    # -- UI scaffolding ----------------------------------------------------

    def _build_menus(self):
        mbar = self.menuBar()
        m_file = mbar.addMenu("&File")
        m_file.addAction("&Open PCD…", self.action_open, "Ctrl+O")
        m_file.addAction("&Save PCD…", self.action_save, "Ctrl+S")
        m_file.addAction("Save PCD As (ASCII)…", lambda: self.action_save(ascii_format=True))
        m_file.addAction("Save as &PLY…", self.action_save_as_ply, "Ctrl+Shift+S")
        m_file.addSeparator()
        m_file.addAction("Export &Map (PGM + YAML)…", self.action_export_map, "Ctrl+M")
        m_file.addSeparator()
        m_file.addAction("E&xit", self.close, "Ctrl+Q")

        m_edit = mbar.addMenu("&Edit")
        m_edit.addAction("Undo", self.action_undo, "Ctrl+Z")
        m_edit.addAction("Redo", self.action_redo, "Ctrl+Y")
        m_edit.addAction("Reset to original", self.action_reset)

        m_view = mbar.addMenu("&View")
        m_view.addAction("Reset camera", self.action_reset_camera, "Ctrl+R")

    def _build_toolbars(self):
        edit_tb = self.addToolBar("Edit")
        edit_tb.setObjectName("EditToolbar")
        edit_tb.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)

        self._a_voxel = edit_tb.addAction("Voxel…", self.action_voxel)
        self._a_sor = edit_tb.addAction("Outlier (SOR)…", self.action_sor)
        self._a_sor.setToolTip(
            "Statistical Outlier Removal — matches PCL `pcl_outlier_removal "
            "-method statistical`. Defaults: 30 neighbours, std-ratio 1.5."
        )
        self._a_radius = edit_tb.addAction("Outlier (Radius)…", self.action_radius_outlier)
        self._a_radius.setToolTip(
            "Radius Outlier Removal — matches PCL `pcl_outlier_removal "
            "-method radius`. Defaults: 30 points within 0.10 m."
        )
        self._a_zclip = edit_tb.addAction("Z-Clip…", self.action_zclip)
        self._a_ground = edit_tb.addAction("Ground (RANSAC)…", self.action_ground)
        self._a_level = edit_tb.addAction("Level Ground", self.action_level_ground)
        self._a_level.setToolTip(
            "Auto-level the cloud so the dominant plane (usually the floor) "
            "becomes horizontal and lands at z = 0. Uses RANSAC. Undoable."
        )

        edit_tb.addSeparator()

        # Crop is now a toggle with a Keep/Remove mode combo.
        self._a_crop = edit_tb.addAction("Crop (drag a box)", self._on_crop_toggled)
        self._a_crop.setCheckable(True)
        self.crop_mode_combo = QtWidgets.QComboBox()
        self.crop_mode_combo.addItems(["Keep inside", "Remove inside"])
        edit_tb.addWidget(self.crop_mode_combo)

        edit_tb.addSeparator()
        self._a_undo = edit_tb.addAction("Undo", self.action_undo)
        self._a_redo = edit_tb.addAction("Redo", self.action_redo)
        self._a_reset = edit_tb.addAction("Reset", self.action_reset)
        edit_tb.addSeparator()

        edit_tb.addWidget(QtWidgets.QLabel(" Color: "))
        self.color_combo = QtWidgets.QComboBox()
        self.color_combo.addItems(["Z", "X", "Y"])
        self.color_combo.setCurrentText("Z")
        self.color_combo.currentTextChanged.connect(self.action_set_color_axis)
        edit_tb.addWidget(self.color_combo)

        edit_tb.addWidget(QtWidgets.QLabel(" Point: "))
        self.point_size_spin = QtWidgets.QDoubleSpinBox()
        self.point_size_spin.setDecimals(1)
        self.point_size_spin.setRange(0.5, 15.0)
        self.point_size_spin.setSingleStep(0.5)
        self.point_size_spin.setValue(1.5)
        self.point_size_spin.setToolTip("Rendered point size in pixels — increase to see sparse clouds more clearly.")
        self.point_size_spin.valueChanged.connect(self.viewer.set_point_size)
        edit_tb.addWidget(self.point_size_spin)

        edit_tb.addSeparator()
        display_wrap = QtWidgets.QWidget()
        dl = QtWidgets.QHBoxLayout(display_wrap)
        dl.setContentsMargins(8, 0, 4, 0)
        dl.setSpacing(6)
        dl.addWidget(QtWidgets.QLabel("Display:"))
        self.lod_label_left = QtWidgets.QLabel("LOD")
        self.lod_label_right = QtWidgets.QLabel("Full")
        dl.addWidget(self.lod_label_left)
        self.display_switch = ToggleSwitch()
        self.display_switch.setChecked(False)  # default = LOD on
        self.display_switch.setToolTip(
            "Toggle full-resolution display.  Off (default) voxel-downsamples "
            "clouds over 500 000 points for fast rendering.  On renders every "
            "point — slower on huge clouds."
        )
        self.display_switch.toggled.connect(self._on_display_lod_toggled)
        dl.addWidget(self.display_switch)
        dl.addWidget(self.lod_label_right)
        edit_tb.addWidget(display_wrap)
        self._update_lod_label_styles(False)

        self.addToolBarBreak()
        marker_tb = self.addToolBar("Marker")
        marker_tb.setObjectName("MarkerToolbar")
        marker_tb.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
        self._a_marker = marker_tb.addAction("Move Marker (LMB → XY, RMB → yaw)", self._on_marker_toggled)
        self._a_marker.setCheckable(True)
        self._a_reset_marker = marker_tb.addAction("Reset Marker", self.action_reset_marker)
        self._a_apply_marker = marker_tb.addAction("Apply Marker as Origin", self.action_apply_marker)
        self._a_apply_marker.setToolTip(
            "Re-frame the cloud so the current marker pose becomes (0, 0, 0) "
            "with yaw 0. Undoable. Save then writes the new frame."
        )
        marker_tb.addSeparator()
        # Typed coordinate entry — X / Y clamped to the cloud's XY scope
        # (set in _refresh_state), yaw in [-180, 180]. Applies live.
        marker_tb.addWidget(QtWidgets.QLabel(" X:"))
        self.marker_x_spin = QtWidgets.QDoubleSpinBox()
        self.marker_x_spin.setDecimals(3)
        self.marker_x_spin.setSingleStep(0.05)
        self.marker_x_spin.setRange(-1e6, 1e6)  # widened to the scope in _refresh_state
        self.marker_x_spin.setKeyboardTracking(False)  # commit on Enter/focus-out, not per digit
        self.marker_x_spin.setToolTip("Marker X (m). Clamped to the cloud's X bounds.")
        self.marker_x_spin.valueChanged.connect(self._on_marker_coord_entered)
        marker_tb.addWidget(self.marker_x_spin)

        marker_tb.addWidget(QtWidgets.QLabel(" Y:"))
        self.marker_y_spin = QtWidgets.QDoubleSpinBox()
        self.marker_y_spin.setDecimals(3)
        self.marker_y_spin.setSingleStep(0.05)
        self.marker_y_spin.setRange(-1e6, 1e6)
        self.marker_y_spin.setKeyboardTracking(False)
        self.marker_y_spin.setToolTip("Marker Y (m). Clamped to the cloud's Y bounds.")
        self.marker_y_spin.valueChanged.connect(self._on_marker_coord_entered)
        marker_tb.addWidget(self.marker_y_spin)

        marker_tb.addWidget(QtWidgets.QLabel(" θ°:"))
        self.marker_yaw_spin = QtWidgets.QDoubleSpinBox()
        self.marker_yaw_spin.setDecimals(1)
        self.marker_yaw_spin.setSingleStep(1.0)
        self.marker_yaw_spin.setRange(-180.0, 180.0)
        self.marker_yaw_spin.setWrapping(True)
        self.marker_yaw_spin.setKeyboardTracking(False)
        self.marker_yaw_spin.setToolTip("Marker yaw (degrees), wraps at ±180.")
        self.marker_yaw_spin.valueChanged.connect(self._on_marker_coord_entered)
        marker_tb.addWidget(self.marker_yaw_spin)

        marker_tb.addSeparator()
        self.marker_label = QtWidgets.QLabel("Marker: (0.000, 0.000, 0.000)  θ=0.0°")
        self.marker_label.setStyleSheet("font-family: monospace; padding: 0 6px;")
        marker_tb.addWidget(self.marker_label)

        self.addToolBarBreak()
        tools_tb = self.addToolBar("Tools")
        tools_tb.setObjectName("ToolsToolbar")
        tools_tb.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)

        # Z-Slab is a *scope*: it limits which points editing tools (Crop,
        # Eraser) operate on. It doesn't remove anything by itself.
        self._a_zslab = tools_tb.addAction("Z-Slab (scope)", self._on_zslab_toggled)
        self._a_zslab.setCheckable(True)
        self._a_zslab.setToolTip(
            "Show two horizontal planes at z_low / z_high; points between are "
            "tinted red. While on, Crop and Eraser only affect in-slab points."
        )
        tools_tb.addWidget(QtWidgets.QLabel(" low:"))
        self.slab_low_spin = QtWidgets.QDoubleSpinBox()
        self.slab_low_spin.setDecimals(3)
        self.slab_low_spin.setRange(-1e6, 1e6)
        self.slab_low_spin.setSingleStep(0.05)
        self.slab_low_spin.valueChanged.connect(self._on_zslab_bounds_changed)
        tools_tb.addWidget(self.slab_low_spin)
        tools_tb.addWidget(QtWidgets.QLabel(" high:"))
        self.slab_high_spin = QtWidgets.QDoubleSpinBox()
        self.slab_high_spin.setDecimals(3)
        self.slab_high_spin.setRange(-1e6, 1e6)
        self.slab_high_spin.setSingleStep(0.05)
        self.slab_high_spin.valueChanged.connect(self._on_zslab_bounds_changed)
        tools_tb.addWidget(self.slab_high_spin)

        tools_tb.addSeparator()

        # Eraser
        self._a_eraser = tools_tb.addAction("Eraser", self._on_eraser_toggled)
        self._a_eraser.setCheckable(True)
        self._a_eraser.setToolTip(
            "Drag with the left mouse button to paint over points and erase "
            "them. Brush size adjustable. Honours Z-Slab if active."
        )
        tools_tb.addWidget(QtWidgets.QLabel(" brush px:"))
        self.eraser_size_spin = QtWidgets.QSpinBox()
        self.eraser_size_spin.setRange(4, 200)
        self.eraser_size_spin.setSingleStep(2)
        self.eraser_size_spin.setValue(30)
        self.eraser_size_spin.valueChanged.connect(self._on_eraser_size_changed)
        tools_tb.addWidget(self.eraser_size_spin)

        tools_tb.addWidget(QtWidgets.QLabel(" shape:"))
        self.eraser_shape_combo = QtWidgets.QComboBox()
        self.eraser_shape_combo.addItems(["Circle", "Square", "Rectangle"])
        self.eraser_shape_combo.setToolTip(
            "Brush shape. Circle/Square take a single half-extent (Brush px). "
            "Rectangle uses Brush px for height and (Brush px × aspect) for width."
        )
        self.eraser_shape_combo.currentTextChanged.connect(self._on_eraser_shape_changed)
        tools_tb.addWidget(self.eraser_shape_combo)

        tools_tb.addWidget(QtWidgets.QLabel(" aspect:"))
        self.eraser_aspect_spin = QtWidgets.QDoubleSpinBox()
        self.eraser_aspect_spin.setDecimals(1)
        self.eraser_aspect_spin.setRange(0.2, 10.0)
        self.eraser_aspect_spin.setSingleStep(0.1)
        self.eraser_aspect_spin.setValue(2.0)
        self.eraser_aspect_spin.setToolTip("Rectangle width / height ratio. Only used when shape = Rectangle.")
        self.eraser_aspect_spin.setEnabled(False)  # only relevant for Rectangle
        self.eraser_aspect_spin.valueChanged.connect(self._on_eraser_aspect_changed)
        tools_tb.addWidget(self.eraser_aspect_spin)

        self.addToolBarBreak()
        occ_tb = self.addToolBar("Occupancy")
        occ_tb.setObjectName("OccupancyToolbar")
        occ_tb.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
        self._a_export = occ_tb.addAction("Generate Map (PGM + YAML)…", self.action_export_map)

    def _build_statusbar(self):
        sb = self.statusBar()
        self.path_label = QtWidgets.QLabel("No PCD loaded")
        self.path_label.setStyleSheet("color: #888;")
        self.stats_label = QtWidgets.QLabel("—")
        self.stats_label.setStyleSheet("font-family: monospace;")
        sb.addPermanentWidget(self.path_label, 1)
        sb.addPermanentWidget(self.stats_label, 2)

    # -- state helpers -----------------------------------------------------

    def _refresh_state(self):
        has = self.current_pcd is not None
        enable_action = has and not self._busy
        for a in (
            self._a_voxel,
            self._a_sor,
            self._a_radius,
            self._a_zclip,
            self._a_ground,
            self._a_level,
            self._a_crop,
            self._a_export,
            self._a_marker,
            self._a_reset_marker,
            self._a_apply_marker,
            self._a_zslab,
            self._a_eraser,
        ):
            a.setEnabled(enable_action)
        self._a_undo.setEnabled(bool(self.history) and not self._busy)
        self._a_redo.setEnabled(bool(self.redo_stack) and not self._busy)
        self._a_reset.setEnabled(has and self.original_pcd is not None and not self._busy)
        self.color_combo.setEnabled(enable_action)
        self.point_size_spin.setEnabled(enable_action)
        self.crop_mode_combo.setEnabled(enable_action)
        self.display_switch.setEnabled(enable_action)
        self.slab_low_spin.setEnabled(enable_action)
        self.slab_high_spin.setEnabled(enable_action)
        self.eraser_size_spin.setEnabled(enable_action)
        self.eraser_shape_combo.setEnabled(enable_action)
        # The aspect spin is only meaningful for Rectangle; keep it gated on
        # both the global enable AND the current shape selection.
        self.eraser_aspect_spin.setEnabled(enable_action and self.eraser_shape_combo.currentText() == "Rectangle")
        for spin in (self.marker_x_spin, self.marker_y_spin, self.marker_yaw_spin):
            spin.setEnabled(enable_action)

        if has:
            s = pcd_ops.stats(self.current_pcd)
            self.stats_label.setText(s.describe())
            # Clamp the X/Y coordinate boxes to the cloud scope, expanded to
            # always include 0.0 so the default origin marker is representable.
            for spin, lo, hi in (
                (self.marker_x_spin, min(float(s.min_bound[0]), 0.0), max(float(s.max_bound[0]), 0.0)),
                (self.marker_y_spin, min(float(s.min_bound[1]), 0.0), max(float(s.max_bound[1]), 0.0)),
            ):
                spin.blockSignals(True)
                spin.setRange(lo, hi)
                spin.blockSignals(False)
        else:
            self.stats_label.setText("—")
            self._a_crop.setChecked(False)
            self.viewer.set_crop_mode(False)
            self._a_marker.setChecked(False)
            self.viewer.set_marker_mode(False)
            self._a_zslab.setChecked(False)
            self.viewer.set_zslab_visible(False)
            self._a_eraser.setChecked(False)
            self.viewer.set_eraser_mode(False)
        self.path_label.setText(self.current_path or "No PCD loaded")

    def _push_history(self, clear_redo: bool = True):
        if self.current_pcd is not None:
            self.history.append(copy.deepcopy(self.current_pcd))
            if len(self.history) > 10:
                self.history = self.history[-10:]
        if clear_redo:
            self.redo_stack.clear()

    def _push_redo(self):
        if self.current_pcd is not None:
            self.redo_stack.append(copy.deepcopy(self.current_pcd))
            if len(self.redo_stack) > 10:
                self.redo_stack = self.redo_stack[-10:]

    def _warn(self, title: str, msg: str):
        QtWidgets.QMessageBox.warning(self, title, msg)

    # -- async plumbing ----------------------------------------------------

    def _begin_busy(self, message: str):
        self._busy = True
        self.loading.start(message)
        self._refresh_state()

    def _end_busy(self):
        self._busy = False
        self.loading.stop()
        self._busy_worker_ref = None
        self._refresh_state()

    def _run_async(self, work_fn, on_success, message: str, error_title: str = "Operation failed"):
        """Schedule `work_fn` on the global thread pool with a loading overlay.
        `on_success(result)` runs on the main thread."""
        if self._busy:
            return  # ignore re-entrancy — UI is already disabled, but be safe
        self._begin_busy(message)
        self._busy_worker_ref = run_async(
            work_fn,
            on_success=on_success,
            on_error=lambda msg: self._warn(error_title, msg),
            on_finished=self._end_busy,
        )

    def _apply_async(
        self,
        label: str,
        work_fn,
        message: str | None = None,
        reset_view: bool = False,
        on_extra=None,
    ):
        """Run `work_fn` -> new_pcd on a worker, pre-compute display arrays,
        then route through the standard apply/history pipeline on the main
        thread.

        `work_fn()` must return either:
          * a single ``PointCloud``  (no extras), or
          * a tuple ``(new_pcd, extra1, extra2, ...)`` — each ``extraN`` is
            passed as a positional argument to ``on_extra``. So if you want
            ``on_extra(plane_before, tilt_deg)``, return
            ``(pcd, plane_before, tilt_deg)`` (FLAT) — *not*
            ``(pcd, (plane_before, tilt_deg))`` (NESTED), which would invoke
            ``on_extra((plane_before, tilt_deg))`` instead.
        """

        def wrapped():
            res = work_fn()
            if isinstance(res, tuple):
                new_pcd, extra = res[0], res[1:]
            else:
                new_pcd, extra = res, None
            if new_pcd is None or len(new_pcd.points) == 0:
                return new_pcd, None, extra
            prepared = self.viewer.prepare_cloud(new_pcd)
            return new_pcd, prepared, extra

        def on_success(result):
            new_pcd, prepared, extra = result
            if new_pcd is None or len(new_pcd.points) == 0:
                self._warn(label, "Operation produced an empty cloud — ignored.")
                return
            self._push_history()
            self.current_pcd = new_pcd
            self.statusBar().showMessage(f"{label}: now {len(new_pcd.points):,} points", 5000)
            self._refresh_state()
            display, pts, rgba = prepared
            self.viewer.apply_prepared_cloud(display, pts, rgba, reset_view=reset_view)
            if on_extra is not None and extra is not None:
                on_extra(*extra)

        self._run_async(wrapped, on_success, message or f"{label}…", error_title=f"{label} failed")

    def closeEvent(self, event):
        # If a worker is still running, disconnect its signals so its queued
        # success / error / finished deliveries don't fire into a destroyed
        # window. The worker thread runs to completion in the background; we
        # just stop listening. `Worker.setAutoDelete(False)` keeps the
        # WorkerSignals object alive until Python's GC collects it normally.
        w = self._busy_worker_ref
        if w is not None:
            for sig in (w.signals.success, w.signals.error, w.signals.finished):
                try:
                    sig.disconnect()
                except (TypeError, RuntimeError):
                    pass  # already disconnected or wrapper gone
            self._busy_worker_ref = None
        super().closeEvent(event)

    # -- file actions ------------------------------------------------------

    def action_open(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open PCD", "", "Point clouds (*.pcd *.ply *.xyz);;All files (*)"
        )
        if not path:
            return
        self.load_path(path)

    def load_path(self, path: str):
        if self._busy:
            return

        def work():
            pcd = pcd_ops.load_pcd(path)
            prepared = self.viewer.prepare_cloud(pcd)
            return pcd, prepared

        def on_success(result):
            pcd, (display, pts, rgba) = result
            self.original_pcd = copy.deepcopy(pcd)
            self.current_pcd = pcd
            self.current_path = path
            self.history.clear()
            self._refresh_state()
            self.viewer.apply_prepared_cloud(display, pts, rgba, reset_view=True)

        self._run_async(
            work,
            on_success,
            message=f"Loading {os.path.basename(path)}…",
            error_title="Open failed",
        )

    def action_save(self, ascii_format: bool = False):
        if self.current_pcd is None or self._busy:
            return
        default = os.path.splitext(self.current_path or "cloud.pcd")[0] + "_edited.pcd"
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save PCD", default, "PCD files (*.pcd);;PLY files (*.ply);;All files (*)"
        )
        if not path:
            return
        pcd = self.current_pcd
        self._run_async(
            lambda: (pcd_ops.save_pcd(pcd, path, ascii_format=ascii_format), path)[1],
            on_success=lambda saved: self.statusBar().showMessage(f"Saved to {saved}", 5000),
            message=f"Saving {os.path.basename(path)}…",
            error_title="Save failed",
        )

    def action_save_as_ply(self):
        """Save the current cloud as a .ply file (PCD → PLY conversion).

        Open3D infers the file format from the extension, so this is just
        ``action_save`` with the dialog defaulted to `.ply`. Useful when
        downstream tools (FAST-LIO localisation, MeshLab, CloudCompare)
        expect PLY rather than PCD."""
        if self.current_pcd is None or self._busy:
            return
        default = os.path.splitext(self.current_path or "cloud.ply")[0] + ".ply"
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save as PLY", default, "PLY files (*.ply);;All files (*)"
        )
        if not path:
            return
        if not path.lower().endswith(".ply"):
            path += ".ply"
        pcd = self.current_pcd
        self._run_async(
            lambda: (pcd_ops.save_pcd(pcd, path, ascii_format=False), path)[1],
            on_success=lambda saved: self.statusBar().showMessage(f"Saved PLY → {saved}", 5000),
            message=f"Saving {os.path.basename(path)}…",
            error_title="Save as PLY failed",
        )

    # -- view / color actions ---------------------------------------------

    def action_reset_camera(self):
        if self.current_pcd is None or self._busy:
            return
        cur = self.current_pcd

        def work():
            return self.viewer.prepare_cloud(cur)

        def on_success(prepared):
            display, pts, rgba = prepared
            self.viewer.apply_prepared_cloud(display, pts, rgba, reset_view=True)

        self._run_async(work, on_success, message="Resetting camera…")

    def action_set_color_axis(self, axis_label: str):
        axis = axis_label.lower()
        self.viewer.set_color_axis(axis)
        self.statusBar().showMessage(f"Color axis → {axis.upper()}", 3000)

    def _update_lod_label_styles(self, full_on: bool) -> None:
        active = "color: #e8e8e8; font-weight: bold;"
        muted = "color: #888;"
        self.lod_label_left.setStyleSheet(muted if full_on else active)
        self.lod_label_right.setStyleSheet(active if full_on else muted)

    def _on_display_lod_toggled(self, checked: bool):
        # Switch ON  → render full cloud (LOD disabled)
        # Switch OFF → render with display LOD (default)
        self.viewer.set_display_lod_enabled(not checked)
        self._update_lod_label_styles(checked)
        if self.current_pcd is None:
            return
        cur = self.current_pcd

        def work():
            return self.viewer.prepare_cloud(cur)

        def on_success(prepared):
            display, pts, rgba = prepared
            self.viewer.apply_prepared_cloud(display, pts, rgba)
            n = len(self.current_pcd.points)
            shown = pts.shape[0]
            msg = (
                f"Display: Full — rendering {shown:,} of {n:,} points"
                if checked
                else f"Display: LOD — rendering {shown:,} of {n:,} points"
            )
            self.statusBar().showMessage(msg, 6000)

        self._run_async(
            work,
            on_success,
            message="Rendering full cloud…" if checked else "Downsampling for display…",
            error_title="Display update failed",
        )

    # -- crop -------------------------------------------------------------

    def _on_crop_toggled(self):
        on = self._a_crop.isChecked()
        self.viewer.set_crop_mode(on)
        if on:
            self._a_marker.setChecked(False)
            self.viewer.set_marker_mode(False)
            self.statusBar().showMessage(
                "Crop mode: drag a rectangle on the canvas. Escape via the toolbar toggle.",
                0,
            )
        else:
            self.statusBar().clearMessage()

    # -- marker -----------------------------------------------------------

    def _on_marker_toggled(self):
        on = self._a_marker.isChecked()
        self.viewer.set_marker_mode(on)
        if on:
            self._a_crop.setChecked(False)
            self.viewer.set_crop_mode(False)
            self.statusBar().showMessage(
                "Marker mode: LMB-drag → move on XY plane, RMB-drag horizontally → yaw angle.",
                0,
            )
        else:
            self.statusBar().clearMessage()

    def action_reset_marker(self):
        self.viewer.reset_marker()

    def action_apply_marker(self):
        if self.current_pcd is None or self._busy:
            return
        pos = np.array(self.viewer.marker_position(), dtype=np.float64)
        yaw = float(np.radians(self.viewer.marker_yaw_degrees()))
        if np.allclose(pos, 0.0) and abs(yaw) < 1e-9:
            self.statusBar().showMessage("Marker is already at the origin — nothing to apply.", 4000)
            return
        cur = self.current_pcd
        self._apply_async(
            "Apply marker as origin",
            lambda: pcd_ops.transform_to_marker_frame(cur, pos, yaw),
            message="Re-framing cloud to marker…",
        )

    def _on_marker_coord_entered(self, _value=None):
        """A marker X / Y / yaw spin box was edited — push it to the viewer.
        Skips while we're syncing the spins *from* a markerMoved event, which
        would otherwise recurse."""
        if self.current_pcd is None or self._busy or self._syncing_marker_spins:
            return
        # Capture all three BEFORE any setter: set_marker_position emits
        # markerMoved, which syncs the yaw spin back to the marker's current
        # yaw — reading the yaw spin afterwards would see the clobbered value.
        x = self.marker_x_spin.value()
        y = self.marker_y_spin.value()
        yaw = np.radians(self.marker_yaw_spin.value())
        self.viewer.set_marker_position(x, y)
        self.viewer.set_marker_yaw(yaw)

    def _on_marker_moved(self, x: float, y: float, z: float, yaw_deg: float):
        self.marker_label.setText(f"Marker: ({x:.3f}, {y:.3f}, {z:.3f})  θ={yaw_deg:+6.1f}°")
        # Mirror mouse drags / Reset / load-time reset into the spin boxes.
        # Guarded both by a flag and blockSignals so neither path can loop.
        if not hasattr(self, "marker_x_spin"):
            return  # signal wired before toolbars are built
        self._syncing_marker_spins = True
        for spin, val in (
            (self.marker_x_spin, x),
            (self.marker_y_spin, y),
            (self.marker_yaw_spin, yaw_deg),
        ):
            spin.blockSignals(True)
            spin.setValue(val)
            spin.blockSignals(False)
        self._syncing_marker_spins = False

    # -- Z-Slab editor -----------------------------------------------------

    def _on_zslab_toggled(self):
        on = self._a_zslab.isChecked()
        if on:
            # Mutually exclusive with other modes (eraser stays free since it's
            # explicit). Initialise the bounds from the current cloud's Z extent
            # so the planes appear at a sensible default.
            if self.current_pcd is not None:
                s = pcd_ops.stats(self.current_pcd)
                z_min, z_max = float(s.min_bound[2]), float(s.max_bound[2])
                # Place planes at the middle 30% of Z by default — the user
                # can widen as needed via the spin boxes.
                lo = z_min + 0.35 * (z_max - z_min)
                hi = z_min + 0.65 * (z_max - z_min)
                self.slab_low_spin.blockSignals(True)
                self.slab_high_spin.blockSignals(True)
                self.slab_low_spin.setValue(lo)
                self.slab_high_spin.setValue(hi)
                self.slab_low_spin.blockSignals(False)
                self.slab_high_spin.blockSignals(False)
                self.viewer.set_zslab_bounds(lo, hi)
            self.viewer.set_zslab_visible(True)
            self.statusBar().showMessage(
                "Z-Slab scope on: only the red (in-slab) points can be edited by Crop / Eraser.",
                0,
            )
        else:
            self.viewer.set_zslab_visible(False)
            self.statusBar().clearMessage()

    def _on_zslab_bounds_changed(self, _value=None):
        if not self._a_zslab.isChecked():
            return
        # Debounce: collapse a burst of spin-arrow auto-repeats into one
        # GL upload. _apply_zslab_bounds fires when the user pauses.
        self._slab_debounce.start()

    def _apply_zslab_bounds(self):
        if not self._a_zslab.isChecked():
            return
        self.viewer.set_zslab_bounds(self.slab_low_spin.value(), self.slab_high_spin.value())

    # -- Eraser ------------------------------------------------------------

    def _on_eraser_toggled(self):
        on = self._a_eraser.isChecked()
        self.viewer.set_eraser_mode(on)
        if on:
            # Mutex with crop / marker (Z-Slab stays — eraser respects the slab).
            self._a_crop.setChecked(False)
            self._a_marker.setChecked(False)
            self.statusBar().showMessage(
                "Eraser: drag with LMB to paint over points and remove them. "
                "If Z-Slab is on, only in-slab points are erased.",
                0,
            )
        else:
            self.statusBar().clearMessage()

    def _on_eraser_size_changed(self, value: int):
        self.viewer.set_eraser_brush(value)

    def _on_eraser_shape_changed(self, label: str):
        shape = label.lower()  # "Circle" → "circle"
        self.viewer.set_eraser_shape(shape)
        # Aspect spin only applies to Rectangle — disable otherwise.
        self.eraser_aspect_spin.setEnabled(shape == "rectangle")

    def _on_eraser_aspect_changed(self, value: float):
        self.viewer.set_eraser_aspect(value)

    def _on_eraser_stroke(self, path: list, brush_radius: int):
        if self.current_pcd is None or self._busy or not path:
            return
        from PIL import Image, ImageDraw

        w_px, h_px = self.viewer.width(), self.viewer.height()
        mvp = self.viewer.mvp_matrix()

        # Rasterise the brush stroke into a 2D boolean mask of the canvas
        # using the current brush shape.
        img = Image.new("L", (w_px, h_px), 0)
        draw = ImageDraw.Draw(img)
        r = int(brush_radius)
        shape = self.viewer.brush_shape()
        aspect = self.viewer.brush_aspect()
        if shape == "rectangle":
            w_half = max(int(round(r * aspect)), 1)
        for pt in path:
            x, y = int(pt.x()), int(pt.y())
            if shape == "circle":
                draw.ellipse([x - r, y - r, x + r, y + r], fill=255)
            elif shape == "square":
                draw.rectangle([x - r, y - r, x + r, y + r], fill=255)
            else:  # rectangle
                draw.rectangle([x - w_half, y - r, x + w_half, y + r], fill=255)
        canvas_mask = np.array(img, dtype=np.uint8) > 0  # shape (H, W)

        cur = self.current_pcd
        slab_on = self._a_zslab.isChecked()
        z_low = float(self.slab_low_spin.value())
        z_high = float(self.slab_high_spin.value())

        def work():
            pts_w = np.asarray(cur.points)
            in_brush, in_front = mask_points_in_brush(pts_w, mvp, w_px, h_px, canvas_mask)
            hit = in_brush & in_front
            if slab_on:
                z_in_slab = (pts_w[:, 2] >= z_low) & (pts_w[:, 2] <= z_high)
                hit = hit & z_in_slab
            keep = np.where(~hit)[0]
            return cur.select_by_index(keep.tolist())

        self._apply_async("Eraser", work, message="Erasing…")

    def _on_crop_requested(self, rect: QtCore.QRect):
        if self.current_pcd is None or self._busy:
            return
        mvp = self.viewer.mvp_matrix()
        w, h = self.viewer.width(), self.viewer.height()
        mode = self.crop_mode_combo.currentText()
        slab_on = self._a_zslab.isChecked()
        z_low = float(self.slab_low_spin.value())
        z_high = float(self.slab_high_spin.value())
        label = "Crop (keep)" if mode == "Keep inside" else "Crop (remove)"
        if slab_on:
            label += " · slab"
        cur = self.current_pcd

        def work():
            pts = np.asarray(cur.points)
            in_rect = mask_points_in_rect(pts, mvp, w, h, rect)
            if slab_on:
                # Z-Slab as a scope: only in-slab points may be affected.
                # Out-of-slab points are always preserved regardless of mode.
                in_slab = (pts[:, 2] >= z_low) & (pts[:, 2] <= z_high)
                if mode == "Keep inside":
                    # Drop only in-slab points outside the rectangle.
                    remove_mask = in_slab & ~in_rect
                else:  # Remove inside
                    remove_mask = in_slab & in_rect
                keep_idx = np.where(~remove_mask)[0]
            else:
                keep_idx = np.where(in_rect if mode == "Keep inside" else ~in_rect)[0]
            return cur.select_by_index(keep_idx.tolist())

        # Auto-exit crop mode whether the crop succeeds or fails — feels
        # natural for the user since the rectangle was just released.
        self._a_crop.setChecked(False)
        self.viewer.set_crop_mode(False)

        self._apply_async(label, work, message=f"{label}…")

    # -- edit actions ------------------------------------------------------

    def action_voxel(self):
        if self.current_pcd is None or self._busy:
            return
        dlg = _ParamDialog("Voxel Downsample", [("Voxel size (m)", 0.05, "float")], self)
        if dlg.exec_() != QtWidgets.QDialog.Accepted:
            return
        v = dlg.values()
        cur = self.current_pcd
        self._apply_async(
            "Voxel downsample",
            lambda: pcd_ops.voxel_downsample(cur, v["Voxel size (m)"]),
            message="Voxel downsampling…",
        )

    def action_sor(self):
        if self.current_pcd is None or self._busy:
            return
        dlg = _ParamDialog(
            "Statistical Outlier Removal",
            [("Neighbors", 30, "int"), ("Std ratio", 1.5, "float")],
            self,
        )
        if dlg.exec_() != QtWidgets.QDialog.Accepted:
            return
        v = dlg.values()
        cur = self.current_pcd
        self._apply_async(
            "Outlier (SOR)",
            lambda: pcd_ops.statistical_outlier_removal(cur, nb_neighbors=v["Neighbors"], std_ratio=v["Std ratio"]),
            message="Removing statistical outliers…",
        )

    def action_radius_outlier(self):
        if self.current_pcd is None or self._busy:
            return
        dlg = _ParamDialog(
            "Radius Outlier Removal",
            [("Min points in radius", 30, "int"), ("Radius (m)", 0.10, "float")],
            self,
        )
        if dlg.exec_() != QtWidgets.QDialog.Accepted:
            return
        v = dlg.values()
        cur = self.current_pcd
        self._apply_async(
            "Outlier (Radius)",
            lambda: pcd_ops.radius_outlier_removal(cur, nb_points=v["Min points in radius"], radius=v["Radius (m)"]),
            message="Removing radius outliers…",
        )

    def action_zclip(self):
        if self.current_pcd is None or self._busy:
            return
        s = pcd_ops.stats(self.current_pcd)
        dlg = _ParamDialog(
            "Z-Clip",
            [("Z min (m)", float(s.min_bound[2]), "float"), ("Z max (m)", float(s.max_bound[2]), "float")],
            self,
        )
        if dlg.exec_() != QtWidgets.QDialog.Accepted:
            return
        v = dlg.values()
        cur = self.current_pcd
        self._apply_async(
            "Z-clip",
            lambda: pcd_ops.z_clip(cur, v["Z min (m)"], v["Z max (m)"]),
            message="Clipping by Z…",
        )

    def action_ground(self):
        if self.current_pcd is None or self._busy:
            return
        dlg = _ParamDialog(
            "Remove Ground (RANSAC)",
            [("Distance threshold (m)", 0.05, "float"), ("Iterations", 1000, "int")],
            self,
        )
        if dlg.exec_() != QtWidgets.QDialog.Accepted:
            return
        v = dlg.values()
        cur = self.current_pcd

        def work():
            non_ground, _, plane = pcd_ops.remove_ground_ransac(
                cur,
                distance_threshold=v["Distance threshold (m)"],
                num_iterations=v["Iterations"],
            )
            return non_ground, tuple(plane)

        def on_extra(plane):
            a, b, c, d = plane
            self.statusBar().showMessage(f"Plane: {a:.3f}x + {b:.3f}y + {c:.3f}z + {d:.3f} = 0", 8000)

        self._apply_async("Ground removal", work, message="Fitting ground plane…", on_extra=on_extra)

    def action_level_ground(self):
        if self.current_pcd is None or self._busy:
            return
        cur = self.current_pcd

        def work():
            new_pcd, plane_before, tilt_deg = pcd_ops.level_ground(cur)
            return new_pcd, plane_before, tilt_deg

        def on_extra(plane_before, tilt_deg):
            a, b, c, d = plane_before
            if tilt_deg < 0.05:
                msg = f"Already level (tilt = {tilt_deg:.3f}°) — no change."
            else:
                msg = f"Levelled {tilt_deg:.2f}°  (plane: {a:.3f}x + {b:.3f}y + {c:.3f}z + {d:.3f} = 0)"
            self.statusBar().showMessage(msg, 10000)

        self._apply_async("Level ground", work, message="Detecting ground plane…", on_extra=on_extra)

    def action_undo(self):
        if not self.history or self._busy:
            return
        prev = self.history.pop()
        self._push_redo()  # save current state for Redo

        def work():
            return self.viewer.prepare_cloud(prev)

        def on_success(prepared):
            display, pts, rgba = prepared
            self.current_pcd = prev
            self.statusBar().showMessage("Undo", 3000)
            self._refresh_state()
            self.viewer.apply_prepared_cloud(display, pts, rgba)

        self._run_async(work, on_success, message="Undo…")

    def action_redo(self):
        if not self.redo_stack or self._busy:
            return
        nxt = self.redo_stack.pop()
        # Save the current state back onto the history stack, but DON'T clear
        # the rest of redo_stack — re-doing should not invalidate further
        # forward steps.
        self._push_history(clear_redo=False)

        def work():
            return self.viewer.prepare_cloud(nxt)

        def on_success(prepared):
            display, pts, rgba = prepared
            self.current_pcd = nxt
            self.statusBar().showMessage("Redo", 3000)
            self._refresh_state()
            self.viewer.apply_prepared_cloud(display, pts, rgba)

        self._run_async(work, on_success, message="Redo…")

    def action_reset(self):
        if self.original_pcd is None or self._busy:
            return
        self._push_history()
        clone = copy.deepcopy(self.original_pcd)

        def work():
            return self.viewer.prepare_cloud(clone)

        def on_success(prepared):
            display, pts, rgba = prepared
            self.current_pcd = clone
            self.statusBar().showMessage("Reset to original", 3000)
            self._refresh_state()
            self.viewer.apply_prepared_cloud(display, pts, rgba, reset_view=True)

        self._run_async(work, on_success, message="Resetting…")

    # -- occupancy grid ----------------------------------------------------

    def action_export_map(self):
        if self.current_pcd is None or self._busy:
            return
        default_path = os.path.splitext(self.current_path or "map.pgm")[0] + "_map.pgm"
        dlg = OccupancyPreviewDialog(self.current_pcd, default_save_path=default_path, parent=self)
        dlg.setModal(False)
        dlg.show()
        # Keep a reference so the dialog isn't garbage-collected.
        self._occupancy_dialog = dlg
