"""PyQt5 main window for the LiDAR PCD editor + occupancy-grid exporter.

Single-window layout (Paint/Photoshop-style):
- Edit toolbar      — filters + Crop toggle + crop-mode combo + undo/reset + color axis
- Occupancy toolbar — map generation
- Central widget    — EmbeddedViewer (pyqtgraph GLViewWidget)
- Info bar (status) — file path + cloud stats as a permanent status-bar widget

All 3D rendering happens inside this window; no pop-out viewer.
"""

from __future__ import annotations

import os

import numpy as np
import open3d as o3d
from PyQt5 import QtCore, QtWidgets

from . import __version__, pcd_ops
from .embedded_viewer import EmbeddedViewer, mask_points_in_brush, mask_points_in_rect
from .loading_overlay import LoadingOverlay
from .occupancy_preview import OccupancyPreviewDialog
from .widgets import ToggleSwitch
from .worker import run_async, split_error

# ---------------------------------------------------------------------------
# Parameter dialog
# ---------------------------------------------------------------------------


class _ParamDialog(QtWidgets.QDialog):
    def __init__(self, title: str, fields: list[tuple], parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self._widgets = {}
        form = QtWidgets.QFormLayout(self)
        for spec in fields:
            label, default, kind = spec[0], spec[1], spec[2]
            # Optional 4th/5th tuple entries give an explicit (min, max). Most
            # of these parameters are physically positive; the old blanket
            # range of +/-1e9 let a user enter voxel_size=0 or radius=-1 and
            # meet a raw Python traceback in a warning box.
            lo, hi = (spec[3], spec[4]) if len(spec) >= 5 else (None, None)
            if kind == "float":
                w = QtWidgets.QDoubleSpinBox()
                w.setDecimals(4)
                w.setRange(-1e9 if lo is None else float(lo), 1e9 if hi is None else float(hi))
                w.setSingleStep(0.01)
                w.setValue(float(default))
            elif kind == "int":
                w = QtWidgets.QSpinBox()
                w.setRange(
                    -10_000_000 if lo is None else int(lo),
                    10_000_000 if hi is None else int(hi),
                )
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

# Undo/redo entries are references to earlier clouds (see _push_history), but
# each is still a whole cloud in memory (~100 MB at 2.2 M points with colours
# and normals), so this cap is a memory bound, not a UX preference.
HISTORY_DEPTH = 10


class MainWindow(QtWidgets.QMainWindow):
    # Emitted whenever `current_pcd` is replaced, so non-modal children (the
    # occupancy preview) can follow the edit instead of holding a stale
    # snapshot they were constructed with.
    cloudChanged = QtCore.pyqtSignal(object)

    def __init__(self):
        super().__init__()
        # Version in the title so a screenshot in a bug report identifies the
        # build. Previously nothing in the UI named the product or its version.
        self.setWindowTitle(f"AlchemyCloud {__version__} — LiDAR PCD Editor & Occupancy Grid Exporter")
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
        # Set on every mutating operation, cleared on load and on a successful
        # save. closeEvent prompts on it, so an hour of edits can no longer be
        # discarded by closing the window.
        self._dirty = False
        # Set False by headless drivers and tests: closeEvent's unsaved-changes
        # prompt is modal, so with nobody to answer it the close would block
        # forever.
        self.prompt_unsaved_on_close = True
        self._busy = False
        self._busy_worker_ref = None  # keep worker alive while running
        # Point-removal edits (eraser strokes, crop) run through their own
        # queue so the canvas never locks: the points vanish from the display
        # at once, the full-cloud edit runs on a worker, and further strokes
        # queue up behind it. `_busy` stays True while the queue is active so
        # every other action's existing `if self._busy` guard still holds.
        self._mask_queue: list[tuple[str, object]] = []
        self._mask_active = False
        self._mask_worker_ref = None
        self._mask_result = None  # ("ok", new_pcd, n_removed) | ("error", msg)
        self._mask_base = None  # cloud the running job started from
        self._mask_removed_total = 0
        self._deferred_action = None  # Undo/Redo pressed mid-queue, run on drain
        # Workers whose signals we disconnected at close: held so the C++
        # QRunnable is not freed while the pool thread may still touch it.
        self._closing_worker_ref = None
        self._closing_mask_worker_ref = None
        # Every open preview dialog, so a second "Generate Map" click cannot
        # orphan the first (which stayed on screen with a live Save button).
        self._occupancy_dialogs: list[QtWidgets.QDialog] = []
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
        """Build the menu bar.

        Menu actions are kept in `self._menu_actions` because the loading
        overlay only eats mouse events on the canvas: menus and their keyboard
        shortcuts stayed live during a long operation, so Ctrl+O mid-RANSAC
        could start a second job. `_refresh_state` disables them while busy.
        """
        mbar = self.menuBar()
        m_file = mbar.addMenu("&File")
        a_open = m_file.addAction("&Open PCD…", self.action_open, "Ctrl+O")
        a_save = m_file.addAction("&Save PCD…", self.action_save, "Ctrl+S")
        a_save_ascii = m_file.addAction("Save PCD As (ASCII)…", lambda: self.action_save(ascii_format=True))
        a_ply = m_file.addAction("Save as &PLY…", self.action_save_as_ply, "Ctrl+Shift+S")
        m_file.addSeparator()
        a_map = m_file.addAction("Export &Map (PGM + YAML)…", self.action_export_map, "Ctrl+M")
        m_file.addSeparator()
        m_file.addAction("E&xit", self.close, "Ctrl+Q")

        m_edit = mbar.addMenu("&Edit")
        a_undo = m_edit.addAction("Undo", self.action_undo, "Ctrl+Z")
        a_redo = m_edit.addAction("Redo", self.action_redo, "Ctrl+Y")
        a_reset = m_edit.addAction("Reset to original", self.action_reset)

        m_view = mbar.addMenu("&View")
        a_cam = m_view.addAction("Reset camera", self.action_reset_camera, "Ctrl+R")

        # Disabled whenever a worker is running.
        self._menu_actions = [a_open, a_save, a_save_ascii, a_ply, a_map, a_undo, a_redo, a_reset, a_cam]
        # Undo/Redo stay live while eraser strokes are queued (they defer).
        self._menu_undo, self._menu_redo = a_undo, a_redo
        # Disabled additionally when no cloud is loaded.
        self._menu_needs_cloud = [a_save, a_save_ascii, a_ply, a_map, a_cam]

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
        self.color_combo.addItems(["Z", "X", "Y", "RGB"])
        self.color_combo.setCurrentText("Z")
        self.color_combo.setToolTip(
            "Z / X / Y colour the cloud by position along that axis.\nRGB shows the file's own colours, if it has any."
        )
        self.color_combo.currentTextChanged.connect(self.action_set_color_axis)
        edit_tb.addWidget(self.color_combo)

        # RGB tone controls — only meaningful while Color is set to RGB.
        edit_tb.addWidget(QtWidgets.QLabel(" Bright: "))
        self.rgb_gamma_spin = QtWidgets.QDoubleSpinBox()
        self.rgb_gamma_spin.setDecimals(2)
        self.rgb_gamma_spin.setRange(0.20, 3.00)
        self.rgb_gamma_spin.setSingleStep(0.05)
        self.rgb_gamma_spin.setValue(self.viewer.rgb_gamma())
        self.rgb_gamma_spin.setToolTip(
            "RGB brightness (gamma). Below 1.00 brightens, above 1.00 darkens.\n"
            "Set 1.00 with Auto-level off to see the file's colours unmodified."
        )
        self.rgb_gamma_spin.valueChanged.connect(self.viewer.set_rgb_gamma)
        edit_tb.addWidget(self.rgb_gamma_spin)

        self.rgb_auto_check = QtWidgets.QCheckBox("Auto-level")
        self.rgb_auto_check.setChecked(self.viewer.rgb_autolevel())
        self.rgb_auto_check.setToolTip(
            "Stretch each RGB channel to its 1–99 % range before applying gamma.\n"
            "Indoor colour scans are usually under-exposed; switch off for raw values."
        )
        self.rgb_auto_check.toggled.connect(self.viewer.set_rgb_autolevel)
        edit_tb.addWidget(self.rgb_auto_check)

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
            "Toggle full-resolution display.  Off (default) shows at most 500 000 "
            "points, a random subset of larger clouds, for fast rendering.  On renders every "
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

        marker_tb.addWidget(QtWidgets.QLabel(" Z:"))
        self.marker_z_spin = QtWidgets.QDoubleSpinBox()
        self.marker_z_spin.setDecimals(3)
        self.marker_z_spin.setSingleStep(0.05)
        self.marker_z_spin.setRange(-1e6, 1e6)
        self.marker_z_spin.setKeyboardTracking(False)
        self.marker_z_spin.setToolTip(
            "Marker Z (m). Applying the marker subtracts this, so the chosen height "
            "becomes z = 0 — use it to put the origin at the robot's base height "
            "rather than at the floor."
        )
        self.marker_z_spin.valueChanged.connect(self._on_marker_coord_entered)
        marker_tb.addWidget(self.marker_z_spin)

        self._a_snap_z = marker_tb.addAction("Snap Z to floor", self.action_snap_marker_z)
        self._a_snap_z.setToolTip(
            "Set marker Z to the detected floor height under the marker, so applying "
            "the marker lands the floor on z = 0."
        )

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

        tools_tb.addSeparator()

        # Manual tilt trim. Level Ground aligns the dominant plane and stops;
        # when a floor and a ceiling disagree there is no single rotation that
        # levels both, and the remaining call belongs to the operator.
        self._a_measure_tilt = tools_tb.addAction("Measure tilt", self.action_measure_tilt)
        self._a_measure_tilt.setToolTip(
            "Report floor and ceiling tilt separately, and whether one rotation can "
            "level both. If they disagree, auto-levelling only moves the problem."
        )
        tools_tb.addWidget(QtWidgets.QLabel(" roll°:"))
        self.tilt_roll_spin = QtWidgets.QDoubleSpinBox()
        self.tilt_roll_spin.setDecimals(3)
        self.tilt_roll_spin.setSingleStep(0.05)
        self.tilt_roll_spin.setRange(-45.0, 45.0)
        self.tilt_roll_spin.setKeyboardTracking(False)
        self.tilt_roll_spin.setToolTip("Rotation about +X, in degrees. Positive lifts +Y.")
        tools_tb.addWidget(self.tilt_roll_spin)

        tools_tb.addWidget(QtWidgets.QLabel(" pitch°:"))
        self.tilt_pitch_spin = QtWidgets.QDoubleSpinBox()
        self.tilt_pitch_spin.setDecimals(3)
        self.tilt_pitch_spin.setSingleStep(0.05)
        self.tilt_pitch_spin.setRange(-45.0, 45.0)
        self.tilt_pitch_spin.setKeyboardTracking(False)
        self.tilt_pitch_spin.setToolTip("Rotation about +Y, in degrees. Positive lifts -X.")
        tools_tb.addWidget(self.tilt_pitch_spin)

        self._a_tilt_trim = tools_tb.addAction("Apply trim", self.action_tilt_trim)
        self._a_tilt_trim.setToolTip(
            "Rotate the cloud by the roll/pitch above, about its own centroid so it "
            "stays where the camera is looking. Undoable; the boxes reset to zero "
            "afterwards so repeated clicks do not compound."
        )

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
            self._a_snap_z,
            self._a_measure_tilt,
            self._a_tilt_trim,
        ):
            a.setEnabled(enable_action)
        for a in getattr(self, "_menu_actions", []):
            a.setEnabled(not self._busy)
        for a in getattr(self, "_menu_needs_cloud", []):
            a.setEnabled(enable_action)
        # While eraser/crop edits are queued, Undo/Redo stay pressable and run
        # once the queue drains — Ctrl+Z straight after a stroke used to be the
        # natural move, and a greyed-out button would silently swallow it.
        can_undo = self._mask_active or (bool(self.history) and not self._busy)
        can_redo = bool(self.redo_stack) and (self._mask_active or not self._busy)
        self._a_undo.setEnabled(can_undo)
        self._a_redo.setEnabled(can_redo)
        if hasattr(self, "_menu_undo"):
            # Menu items keep their old rule (live whenever idle) plus stay
            # live while edits are queued, so Ctrl+Z / Ctrl+Y can defer.
            self._menu_undo.setEnabled(not self._busy or self._mask_active)
            self._menu_redo.setEnabled(not self._busy or self._mask_active)
        self._a_reset.setEnabled(has and self.original_pcd is not None and not self._busy)
        self.color_combo.setEnabled(enable_action)
        # The RGB tone controls do nothing outside RGB mode — gate them on it
        # so the toolbar shows which knobs are actually live.
        rgb_mode = self.color_combo.currentText() == "RGB"
        self.rgb_gamma_spin.setEnabled(enable_action and rgb_mode)
        self.rgb_auto_check.setEnabled(enable_action and rgb_mode)
        self.point_size_spin.setEnabled(enable_action)
        self.crop_mode_combo.setEnabled(enable_action)
        self.display_switch.setEnabled(enable_action)
        self.slab_low_spin.setEnabled(enable_action)
        self.slab_high_spin.setEnabled(enable_action)
        self.eraser_size_spin.setEnabled(enable_action)
        self.eraser_shape_combo.setEnabled(enable_action)
        self.tilt_roll_spin.setEnabled(enable_action)
        self.tilt_pitch_spin.setEnabled(enable_action)
        # The aspect spin is only meaningful for Rectangle; keep it gated on
        # both the global enable AND the current shape selection.
        self.eraser_aspect_spin.setEnabled(enable_action and self.eraser_shape_combo.currentText() == "Rectangle")
        if has and self._mask_active:
            # The eraser keeps working while earlier strokes are committed in
            # the background — that is the whole point of the queue.
            self._a_eraser.setEnabled(True)
            self.eraser_size_spin.setEnabled(True)
            self.eraser_shape_combo.setEnabled(True)
            self.eraser_aspect_spin.setEnabled(self.eraser_shape_combo.currentText() == "Rectangle")
        for spin in (
            self.marker_x_spin,
            self.marker_y_spin,
            self.marker_z_spin,
            self.marker_yaw_spin,
        ):
            spin.setEnabled(enable_action)

        if has:
            # Cached per cloud object: _refresh_state runs on every busy flip,
            # and stats() is O(N) on the GUI thread.
            cache = getattr(self, "_stats_cache", None)
            if cache is not None and cache[0] is self.current_pcd:
                s = cache[1]
            else:
                s = pcd_ops.stats(self.current_pcd)
                self._stats_cache = (self.current_pcd, s)
            self.stats_label.setText(s.describe())
            # Clamp the X/Y coordinate boxes to the cloud scope, expanded to
            # always include 0.0 (so the default origin marker is
            # representable) and the marker's *current* position. Without the
            # latter, a marker placed outside the cloud bbox — legitimate, e.g.
            # a robot start pose just outside the scanned area — was pulled
            # inside the moment setRange ran, so editing yaw alone teleported it.
            mx, my, mz = self.viewer.marker_position()
            for spin, lo, hi in (
                (
                    self.marker_x_spin,
                    min(float(s.min_bound[0]), 0.0, mx),
                    max(float(s.max_bound[0]), 0.0, mx),
                ),
                (
                    self.marker_y_spin,
                    min(float(s.min_bound[1]), 0.0, my),
                    max(float(s.max_bound[1]), 0.0, my),
                ),
                (
                    self.marker_z_spin,
                    min(float(s.min_bound[2]), 0.0, mz),
                    max(float(s.max_bound[2]), 0.0, mz),
                ),
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

    def _reset_undo_stacks(self) -> None:
        """Clear both stacks. Loading a file used to clear only `history`, so
        Redo stayed enabled and popped a cloud belonging to the *previous* file
        straight into `current_pcd` — while `current_path` still named the new
        one, so Ctrl+S then wrote the old cloud under the new file's name."""
        self.history.clear()
        self.redo_stack.clear()

    # History, redo and `original_pcd` hold *references*, not copies: no
    # operation modifies a cloud in place (every pcd_ops transform returns a
    # new cloud, and the viewer recolours only its own display copy), so a
    # replaced cloud is already an immutable snapshot. Copying it here cost
    # ~160 ms of frozen GUI per operation on a 1.8M-point cloud.
    # tests/test_gui_thread_perf.py guards that invariant.
    def _push_history(self, clear_redo: bool = True):
        if self.current_pcd is not None:
            self.history.append(self.current_pcd)
            if len(self.history) > HISTORY_DEPTH:
                self.history = self.history[-HISTORY_DEPTH:]
        if clear_redo:
            self.redo_stack.clear()

    def _push_redo(self):
        if self.current_pcd is not None:
            self.redo_stack.append(self.current_pcd)
            if len(self.redo_stack) > HISTORY_DEPTH:
                self.redo_stack = self.redo_stack[-HISTORY_DEPTH:]

    def _warn(self, title: str, msg: str):
        """Warning dialog. A worker failure arrives with its traceback
        appended; show only the reason and keep the traceback under "Show
        Details…" (it is in the log file too, see worker.py)."""
        summary, details = split_error(msg)
        if details is None:
            QtWidgets.QMessageBox.warning(self, title, msg)
            return
        box = QtWidgets.QMessageBox(QtWidgets.QMessageBox.Warning, title, summary, QtWidgets.QMessageBox.Ok, self)
        box.setDetailedText(details)
        box.exec_()

    # -- async plumbing ----------------------------------------------------

    # Most operations now finish in well under half a second; showing the
    # spinner only after this long keeps them from flashing it. The tools are
    # locked (`_busy`) from the start either way.
    OVERLAY_DELAY_MS = 300

    def _begin_busy(self, message: str):
        self._busy = True
        # The overlay that swallows canvas clicks is not up yet for the first
        # OVERLAY_DELAY_MS, so lock the viewer's tools (marker drag, crop,
        # eraser) directly. The camera stays movable.
        self.viewer.set_tools_locked(True)
        self.loading.start(message, delay_ms=self.OVERLAY_DELAY_MS)
        self._refresh_state()

    def _end_busy(self):
        self._busy = False
        self.viewer.set_tools_locked(False)
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
        reset_marker: bool = False,
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
            self._dirty = True
            self.cloudChanged.emit(new_pcd)
            self.statusBar().showMessage(f"{label}: now {len(new_pcd.points):,} points", 5000)
            self._refresh_state()
            display, pts, rgba = prepared
            self.viewer.apply_prepared_cloud(display, pts, rgba, reset_view=reset_view, reset_marker=reset_marker)
            if on_extra is not None and extra is not None:
                on_extra(*extra)

        self._run_async(wrapped, on_success, message or f"{label}…", error_title=f"{label} failed")

    # -- point-removal queue (eraser strokes, crop) ---------------------------
    #
    # An eraser stroke on a 1.8 M-point cloud used to lock the whole window for
    # ~2 s: project every point, build a Python index list, deep-copy the cloud
    # for undo, re-sample the display. Now:
    #   1. GUI thread, instantly: hide the hit points from the existing display
    #      subset (`viewer.remove_display_points`) — no re-sample.
    #   2. Worker, one job at a time, in stroke order: apply the same
    #      `remove_fn` to the full cloud and build the new cloud from numpy.
    #      Each job starts from the cloud the previous job produced.
    #   3. GUI thread, per job: commit. The replaced cloud object goes onto the
    #      undo stack as-is (no deep copy) — nothing mutates a cloud in place,
    #      and it is no longer `current_pcd`.
    # The canvas stays live throughout, and strokes made meanwhile queue up.

    def _submit_mask_edit(self, label: str, remove_fn) -> None:
        if self.current_pcd is None:
            return
        self.viewer.remove_display_points(remove_fn)
        self._mask_queue.append((label, remove_fn))
        self._dirty = True
        if not self._mask_active:
            self._mask_active = True
            self._busy = True  # blocks every other action via its existing guard
            self._mask_removed_total = 0
            self._refresh_state()
        if self._mask_worker_ref is None:
            self._start_next_mask_job()
        else:
            self._show_mask_progress()

    def _show_mask_progress(self) -> None:
        n = len(self._mask_queue) + (1 if self._mask_worker_ref is not None else 0)
        self.statusBar().showMessage(f"Applying edits in the background… ({n} pending)", 0)

    def _start_next_mask_job(self) -> None:
        label, remove_fn = self._mask_queue.pop(0)
        base = self.current_pcd
        self._mask_base = base
        self._mask_label = label
        self._mask_result = None

        def work():
            hit = np.asarray(remove_fn(np.asarray(base.points)), dtype=bool)
            n = int(hit.sum())
            if n == 0:
                return base, 0
            return pcd_ops.select_by_mask(base, ~hit), n

        def on_success(res):
            self._mask_result = ("ok", res[0], res[1])

        def on_error(msg):
            self._mask_result = ("error", msg)

        self._mask_worker_ref = run_async(
            work, on_success=on_success, on_error=on_error, on_finished=self._on_mask_job_finished
        )
        self._show_mask_progress()

    def _on_mask_job_finished(self) -> None:
        self._mask_worker_ref = None
        res, label, base = self._mask_result, self._mask_label, self._mask_base
        self._mask_result = None
        self._mask_base = None

        problem = None  # (title, message) when this job cannot be committed
        if res is None or self.current_pcd is not base:
            problem = (f"{label} failed", "The edit could not be applied.")
        elif res[0] == "error":
            problem = (f"{label} failed", res[1])
        elif len(res[1].points) == 0:
            problem = (label, "Operation produced an empty cloud — ignored.")
        if problem is not None:
            # The display already hides this job's points (and any queued
            # ones). Drop the queue and rebuild the display from the cloud we
            # actually have, so the screen never disagrees with current_pcd.
            self._mask_queue.clear()
            # An Undo pressed mid-queue was aimed at a history this failure
            # just changed (later strokes were dropped), so running it now
            # could undo a stroke the user meant to keep. Cancel it.
            self._deferred_action = None
            if self.current_pcd is not None:
                self.viewer.set_cloud(self.current_pcd)
            self._end_mask_queue()
            self._warn(*problem)
            return

        _, new_pcd, n_removed = res
        if n_removed:
            self.history.append(base)  # replaced object, not a copy — see above
            if len(self.history) > HISTORY_DEPTH:
                self.history = self.history[-HISTORY_DEPTH:]
            self.redo_stack.clear()
            self.current_pcd = new_pcd
            self._dirty = True
            self._mask_removed_total += n_removed
            self.cloudChanged.emit(new_pcd)

        if self._mask_queue:
            self._start_next_mask_job()
        else:
            self._maybe_refill_display()

    def _maybe_refill_display(self) -> None:
        """After heavy erasing the display subset can thin out (it only ever
        loses points). If it fell below half its budget, re-sample it from the
        current cloud on a worker; otherwise finish straight away."""
        cur = self.current_pcd
        n = 0 if cur is None else len(cur.points)
        target = min(self.viewer.DISPLAY_MAX, n) if self.viewer.is_display_lod_enabled() else n
        if n == 0 or self.viewer.display_point_count() >= 0.5 * target:
            self._end_mask_queue()
            return

        def work():
            return self.viewer.prepare_cloud(cur)

        def on_success(prepared):
            self._mask_result = ("refill", cur, prepared)

        def on_error(msg):
            self._mask_result = ("error", msg)

        self._mask_result = None
        self._mask_worker_ref = run_async(
            work, on_success=on_success, on_error=on_error, on_finished=self._on_refill_finished
        )

    def _on_refill_finished(self) -> None:
        self._mask_worker_ref = None
        res, self._mask_result = self._mask_result, None
        if res is not None and res[0] == "refill" and res[1] is self.current_pcd:
            display, pts, rgba = res[2]
            self.viewer.apply_prepared_cloud(display, pts, rgba)
            # Strokes made while re-sampling were hidden on the old display.
            for _, fn in self._mask_queue:
                self.viewer.remove_display_points(fn)
        if self._mask_queue:
            self._start_next_mask_job()
        else:
            self._end_mask_queue()

    def _end_mask_queue(self) -> None:
        self._mask_active = False
        self._busy = False
        self._refresh_state()
        if self._mask_removed_total and self.current_pcd is not None:
            self.statusBar().showMessage(
                f"Removed {self._mask_removed_total:,} points — now {len(self.current_pcd.points):,} points",
                5000,
            )
        else:
            self.statusBar().clearMessage()
        self._mask_removed_total = 0
        deferred, self._deferred_action = self._deferred_action, None
        if deferred is not None:
            deferred()

    def _forget_occupancy_dialog(self, dlg) -> None:
        if dlg in self._occupancy_dialogs:
            self._occupancy_dialogs.remove(dlg)

    def _on_saved(self, saved_path: str, prefix: str) -> None:
        self._dirty = False
        self.statusBar().showMessage(f"{prefix} {saved_path}", 5000)

    def closeEvent(self, event):
        # Eraser/crop edits still committing: refuse this close (they finish in
        # well under a second) rather than prompt about, or save, a cloud that
        # is about to change. Headless callers (no prompt) fall through and
        # the queue is abandoned below.
        if self._mask_active and self.prompt_unsaved_on_close:
            self.statusBar().showMessage("Finishing pending edits — close again in a moment.", 5000)
            event.ignore()
            return

        # Unsaved edits: ask before throwing them away. There was previously no
        # dirty tracking at all, so an hour of filtering and cropping vanished
        # on close with no prompt.
        # A Save choice runs on a worker, so the close is refused and retried
        # once it lands rather than racing teardown against the file write.
        if not self._ok_to_discard_edits("close again"):
            event.ignore()
            return

        # Close any child preview dialogs first so their timers and workers stop
        # before this window is torn down.
        for dlg in list(self._occupancy_dialogs):
            dlg.close()
        self._occupancy_dialogs.clear()

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
            # Keep the reference alive until the pool thread is done with the
            # C++ QRunnable: dropping the last Python ref while the task is
            # still queued frees it under the pool and segfaults.
            self._closing_worker_ref = w
            self._busy_worker_ref = None
        # Same for a running eraser/crop job; queued ones are simply dropped.
        self._mask_queue.clear()
        mw = self._mask_worker_ref
        if mw is not None:
            for sig in (mw.signals.success, mw.signals.error, mw.signals.finished):
                try:
                    sig.disconnect()
                except (TypeError, RuntimeError):
                    pass
            self._closing_mask_worker_ref = mw
            self._mask_worker_ref = None
        super().closeEvent(event)

    # -- file actions ------------------------------------------------------

    def _ok_to_discard_edits(self, retry_hint: str) -> bool:
        """Ask before unsaved edits are thrown away. True means go ahead.

        Save starts an async write, so the caller cannot proceed in the same
        call; the user is told to retry (`retry_hint`) once it lands.
        """
        if not (self._dirty and self.prompt_unsaved_on_close and self.current_pcd is not None):
            return True
        choice = QtWidgets.QMessageBox.warning(
            self,
            "Unsaved changes",
            "This cloud has unsaved edits. Save them first?",
            QtWidgets.QMessageBox.Save | QtWidgets.QMessageBox.Discard | QtWidgets.QMessageBox.Cancel,
            QtWidgets.QMessageBox.Save,
        )
        if choice == QtWidgets.QMessageBox.Cancel:
            return False
        if choice == QtWidgets.QMessageBox.Save:
            self.action_save()
            if self._busy:  # the write is still in flight
                self.statusBar().showMessage(f"Saving… {retry_hint} once the save completes.", 6000)
                return False
            return not self._dirty  # dialog cancelled or save failed
        return True

    def action_open(self):
        if self._busy:
            return
        # Opening a file replaces the cloud and clears undo, so it loses
        # unsaved edits exactly like closing the window does.
        if not self._ok_to_discard_edits("open the file again"):
            return
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
            # The dropped count comes from the load itself: re-reading the file
            # to count it froze the GUI for seconds after every load.
            pcd, dropped = pcd_ops.load_pcd_counted(path)
            prepared = self.viewer.prepare_cloud(pcd)
            return pcd, dropped, prepared

        def on_success(result):
            pcd, dropped, (display, pts, rgba) = result
            self.original_pcd = pcd  # shared, never modified (see _push_history)
            self.current_pcd = pcd
            self.current_path = path
            self.cloudChanged.emit(pcd)
            self._reset_undo_stacks()
            self._dirty = False
            self._refresh_state()
            # A new file is a new frame of reference, so the marker resets here
            # (and only here, plus after Apply Marker).
            self.viewer.apply_prepared_cloud(display, pts, rgba, reset_view=True, reset_marker=True)
            if dropped:
                self.statusBar().showMessage(
                    f"Loaded {len(pcd.points):,} points — dropped {dropped:,} non-finite (NaN/inf) point(s).",
                    8000,
                )

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
            on_success=lambda saved: self._on_saved(saved, "Saved to"),
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
            on_success=lambda saved: self._on_saved(saved, "Saved PLY →"),
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
        self._refresh_state()
        if axis != "rgb":
            self.statusBar().showMessage(f"Color axis → {axis.upper()}", 3000)
        elif self.viewer.has_source_colors():
            self.statusBar().showMessage(
                "Colour → RGB (the file's own colours). Use Bright / Auto-level to expose them.",
                5000,
            )
        else:
            self.statusBar().showMessage(
                "Colour → RGB, but this cloud has no colour — showing flat grey. "
                "Colour needs an 'rgb' field in the .pcd / .ply.",
                6000,
            )

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
            # set_crop_mode already clears eraser state inside the viewer, but
            # the button kept its checked look — so the toolbar showed two
            # active tools and Eraser needed two clicks to come back.
            self._a_marker.setChecked(False)
            self.viewer.set_marker_mode(False)
            self._a_eraser.setChecked(False)
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
            self._a_eraser.setChecked(False)
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
            reset_marker=True,  # the marker *became* the origin
        )

    def action_snap_marker_z(self):
        """Set marker Z to the floor height, so Apply lands the floor on z = 0.

        Uses the same constrained RANSAC as Level Ground, so it inherits the
        near-horizontal guard and will refuse rather than snap to a wall.
        """
        if self.current_pcd is None or self._busy:
            return
        cur = self.current_pcd

        # The RANSAC fit is the slow part, so run it off the main thread.
        def fit():
            n, d = pcd_ops.fit_ground_plane(cur)
            mx, my, _mz = self.viewer.marker_position()
            # Height of the plane directly under the marker: n·p + d = 0.
            z = -(d + n[0] * mx + n[1] * my) / n[2]
            return float(z), float(np.degrees(np.arccos(float(np.clip(n[2], -1.0, 1.0)))))

        def done(result):
            z, tilt = result
            mx, my, _ = self.viewer.marker_position()
            self.viewer.set_marker_position(mx, my, z)
            self.statusBar().showMessage(
                f"Marker Z snapped to the floor at z = {z:.3f} m "
                f"(plane tilt {tilt:.2f}°). Apply Marker as Origin to land it on 0.",
                8000,
            )

        self._run_async(fit, done, message="Fitting the floor…", error_title="Snap Z failed")

    def action_tilt_trim(self):
        """Apply the typed roll/pitch trim to the cloud."""
        if self.current_pcd is None or self._busy:
            return
        roll = float(self.tilt_roll_spin.value())
        pitch = float(self.tilt_pitch_spin.value())
        if abs(roll) < 1e-9 and abs(pitch) < 1e-9:
            self.statusBar().showMessage("Roll and pitch are both zero — nothing to apply.", 4000)
            return
        cur = self.current_pcd
        self._apply_async(
            "Tilt trim",
            lambda: pcd_ops.apply_tilt(cur, roll, pitch),
            message=f"Tilting roll {roll:+.2f}° pitch {pitch:+.2f}°…",
        )
        # A trim is a nudge, not a setting; zero the boxes so repeated clicks
        # do not silently compound the same rotation.
        for spin in (self.tilt_roll_spin, self.tilt_pitch_spin):
            spin.blockSignals(True)
            spin.setValue(0.0)
            spin.blockSignals(False)

    def action_measure_tilt(self):
        """Report floor vs ceiling tilt, and whether one rotation can fix both."""
        if self.current_pcd is None or self._busy:
            return
        cur = self.current_pcd

        def work():
            nf, tf, cf = pcd_ops.surface_tilt(cur, percentile=2.0)
            nc, tc, cc = pcd_ops.surface_tilt(cur, percentile=98.0)
            between = float(np.degrees(np.arccos(float(np.clip(abs(nf @ nc), -1.0, 1.0)))))
            return tf, cf, tc, cc, between

        def done(result):
            tf, cf, tc, cc, between = result
            if between < 2.0:
                verdict = "consistent — Level Ground can fix both."
            else:
                verdict = (
                    f"inconsistent by {between:.2f}° — no single rotation levels both. "
                    "Level the surface you care about, then trim by eye."
                )
            self._warn(
                "Tilt measurement",
                f"Floor  ({cf:,} pts):  {tf:.2f}° from horizontal\n"
                f"Ceiling ({cc:,} pts):  {tc:.2f}° from horizontal\n\n"
                f"They disagree by {between:.2f}°.\n\n{verdict}",
            )

        self._run_async(work, done, message="Measuring tilt…", error_title="Measure failed")

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
        z = self.marker_z_spin.value()
        yaw = np.radians(self.marker_yaw_spin.value())
        self.viewer.set_marker_position(x, y, z)
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
            (self.marker_z_spin, z),
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
        # Strokes may queue behind earlier ones; any *other* busy job blocks.
        if self.current_pcd is None or not path or (self._busy and not self._mask_active):
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

        slab_on = self._a_zslab.isChecked()
        # Ask the viewer, not the spin boxes: it is the component that
        # normalises an inverted low/high pair, and the scope filter must match
        # the slab the user can actually see.
        z_low, z_high = self.viewer.slab_bounds()

        # Everything the stroke depends on (camera, canvas size, brush mask,
        # slab) is captured now, so a queued stroke still erases what the user
        # painted even if the camera moves before it runs.
        def remove_fn(pts_w: np.ndarray) -> np.ndarray:
            in_brush, in_front = mask_points_in_brush(pts_w, mvp, w_px, h_px, canvas_mask)
            hit = in_brush & in_front
            if slab_on:
                hit &= (pts_w[:, 2] >= z_low) & (pts_w[:, 2] <= z_high)
            return hit

        self._submit_mask_edit("Eraser", remove_fn)

    def _on_crop_requested(self, rect: QtCore.QRect):
        if self.current_pcd is None or self._busy:
            return
        mvp = self.viewer.mvp_matrix()
        w, h = self.viewer.width(), self.viewer.height()
        mode = self.crop_mode_combo.currentText()
        slab_on = self._a_zslab.isChecked()
        z_low, z_high = self.viewer.slab_bounds()
        label = "Crop (keep)" if mode == "Keep inside" else "Crop (remove)"
        if slab_on:
            label += " · slab"

        def remove_fn(pts: np.ndarray) -> np.ndarray:
            in_rect = mask_points_in_rect(pts, mvp, w, h, rect)
            if slab_on:
                # Z-Slab as a scope: only in-slab points may be affected.
                # Out-of-slab points are always preserved regardless of mode.
                in_slab = (pts[:, 2] >= z_low) & (pts[:, 2] <= z_high)
                if mode == "Keep inside":
                    # Drop only in-slab points outside the rectangle.
                    return in_slab & ~in_rect
                return in_slab & in_rect  # Remove inside
            return ~in_rect if mode == "Keep inside" else in_rect

        # Auto-exit crop mode whether the crop succeeds or fails — feels
        # natural for the user since the rectangle was just released.
        self._a_crop.setChecked(False)
        self.viewer.set_crop_mode(False)

        self._submit_mask_edit(label, remove_fn)

    # -- edit actions ------------------------------------------------------

    def action_voxel(self):
        if self.current_pcd is None or self._busy:
            return
        dlg = _ParamDialog(
            "Voxel Downsample",
            [("Voxel size (m)", 0.05, "float", 0.0001, 100.0)],
            self,
        )
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
            [("Neighbors", 30, "int", 1, 10_000), ("Std ratio", 1.5, "float", 0.0001, 100.0)],
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
            [("Min points in radius", 30, "int", 1, 10_000), ("Radius (m)", 0.10, "float", 0.0001, 100.0)],
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
            [("Distance threshold (m)", 0.05, "float", 0.0001, 10.0), ("Iterations", 1000, "int", 10, 100_000)],
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
        if self._mask_active:
            self._deferred_action = self.action_undo
            self.statusBar().showMessage("Undo will run as soon as the pending edits finish…", 0)
            return
        if not self.history or self._busy:
            return
        # Stacks change only once the worker succeeds, so a failed prepare
        # cannot lose or duplicate an undo step. `_busy` keeps them untouched
        # meanwhile.
        prev = self.history[-1]

        def work():
            return self.viewer.prepare_cloud(prev)

        def on_success(prepared):
            display, pts, rgba = prepared
            self.history.pop()
            self._push_redo()  # save current state for Redo
            self.current_pcd = prev
            self._dirty = True
            self.cloudChanged.emit(prev)
            self.statusBar().showMessage("Undo", 3000)
            self._refresh_state()
            self.viewer.apply_prepared_cloud(display, pts, rgba)

        self._run_async(work, on_success, message="Undo…")

    def action_redo(self):
        if self._mask_active:
            self._deferred_action = self.action_redo
            self.statusBar().showMessage("Redo will run as soon as the pending edits finish…", 0)
            return
        if not self.redo_stack or self._busy:
            return
        nxt = self.redo_stack[-1]  # popped on success, as in action_undo

        def work():
            return self.viewer.prepare_cloud(nxt)

        def on_success(prepared):
            display, pts, rgba = prepared
            self.redo_stack.pop()
            # Save the current state back onto the history stack, but DON'T
            # clear the rest of redo_stack — re-doing should not invalidate
            # further forward steps.
            self._push_history(clear_redo=False)
            self.current_pcd = nxt
            self._dirty = True
            self.cloudChanged.emit(nxt)
            self.statusBar().showMessage("Redo", 3000)
            self._refresh_state()
            self.viewer.apply_prepared_cloud(display, pts, rgba)

        self._run_async(work, on_success, message="Redo…")

    def action_reset(self):
        if self.original_pcd is None or self._busy:
            return
        clone = self.original_pcd  # shared, never modified (see _push_history)

        def work():
            return self.viewer.prepare_cloud(clone)

        def on_success(prepared):
            display, pts, rgba = prepared
            self._push_history()  # only once it worked, as in action_undo
            self.current_pcd = clone
            self._dirty = True
            self.cloudChanged.emit(clone)
            self.statusBar().showMessage("Reset to original", 3000)
            self._refresh_state()
            self.viewer.apply_prepared_cloud(display, pts, rgba, reset_view=True)

        self._run_async(work, on_success, message="Resetting…")

    # -- occupancy grid ----------------------------------------------------

    def action_export_map(self):
        if self.current_pcd is None or self._busy:
            return
        default_path = os.path.splitext(self.current_path or "map.pgm")[0] + "_map.pgm"
        # Close any previous preview first. Reassigning a single attribute
        # left the old dialog visible with a live Save button, exporting a cloud
        # that no longer matched the window.
        for old in list(self._occupancy_dialogs):
            old.close()
        self._occupancy_dialogs.clear()

        dlg = OccupancyPreviewDialog(self.current_pcd, default_save_path=default_path, parent=self)
        dlg.setModal(False)
        # The dialog is non-modal, so the cloud can change under it. Feed it the
        # new cloud instead of letting it export the snapshot it was born with.
        self.cloudChanged.connect(dlg.set_cloud)
        dlg.destroyed.connect(lambda *_, d=dlg: self._forget_occupancy_dialog(d))
        dlg.show()
        self._occupancy_dialogs.append(dlg)
