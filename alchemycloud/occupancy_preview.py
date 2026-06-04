"""Non-modal occupancy-grid preview dialog.

Mirrors the workflow of ``temp file/occupancy_grid_tuner.py`` (sliders +
live PGM preview), built on PyQt5 and the existing ``occupancy.build`` /
``occupancy.save`` pipeline. Heavy builds run on a `QThreadPool` worker so
the UI stays responsive; rapid slider drags are debounced so we don't
queue an avalanche of rebuilds.

The dialog operates on whatever ``pcd`` you hand it on construction —
the caller (`MainWindow.action_export_map`) passes in `self.current_pcd`.
"""
from __future__ import annotations

import os

import numpy as np
import open3d as o3d
from PyQt5 import QtCore, QtGui, QtWidgets

from . import occupancy
from . import pcd_ops
from .worker import run_async


class OccupancyPreviewDialog(QtWidgets.QDialog):
    DEBOUNCE_MS = 120

    def __init__(
        self,
        pcd: o3d.geometry.PointCloud,
        default_save_path: str = "",
        parent: QtWidgets.QWidget | None = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Occupancy Grid Preview")
        self.setWindowFlags(self.windowFlags() | QtCore.Qt.WindowMinMaxButtonsHint)
        self.resize(900, 560)

        self._pcd = pcd
        self._default_save_path = default_save_path
        self._busy = False
        self._dirty = False  # request landed while we were busy
        self._last_grid: occupancy.OccupancyGrid | None = None
        self._last_params: occupancy.GridParams | None = None
        self._worker_ref = None

        s = pcd_ops.stats(pcd)
        z_min = float(s.min_bound[2])
        z_max = float(s.max_bound[2])
        z_span = max(z_max - z_min, 1e-3)
        # Default to the full Z range so the first preview always has points.
        # The user can trim with the spin boxes. Threshold defaults to 30%
        # up the Z range — the friend's heuristic for "robot body height".
        self._z_low_default = z_min
        self._z_high_default = z_max
        self._threshold_default = z_min + 0.30 * z_span

        # Debounce timer must exist before _build_ui() — connecting widgets
        # below will start emitting `_schedule` callbacks immediately.
        self._debounce = QtCore.QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(self.DEBOUNCE_MS)
        self._debounce.timeout.connect(self._render_preview)

        self._build_ui()

        # Kick off the initial preview.
        QtCore.QTimer.singleShot(0, self._render_preview)

    # -- UI ----------------------------------------------------------------

    def _build_ui(self):
        outer = QtWidgets.QHBoxLayout(self)

        # ---- Controls panel
        ctrl = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout(ctrl)
        form.setLabelAlignment(QtCore.Qt.AlignRight)
        ctrl.setMinimumWidth(260)
        ctrl.setMaximumWidth(320)

        self.method_combo = QtWidgets.QComboBox()
        self.method_combo.addItems(["projection", "voxel3d"])
        self.method_combo.currentTextChanged.connect(self._schedule)
        form.addRow("Method", self.method_combo)

        self.resolution_spin = QtWidgets.QDoubleSpinBox()
        self.resolution_spin.setDecimals(3)
        self.resolution_spin.setRange(0.005, 5.0)
        self.resolution_spin.setSingleStep(0.01)
        self.resolution_spin.setValue(0.05)
        self.resolution_spin.valueChanged.connect(self._schedule)
        form.addRow("Resolution (m/cell)", self.resolution_spin)

        self.zmin_spin = QtWidgets.QDoubleSpinBox()
        self.zmin_spin.setDecimals(3)
        self.zmin_spin.setRange(-1e6, 1e6)
        self.zmin_spin.setSingleStep(0.05)
        self.zmin_spin.setValue(self._z_low_default)
        self.zmin_spin.valueChanged.connect(self._schedule)
        form.addRow("Z min (m)", self.zmin_spin)

        self.zmax_spin = QtWidgets.QDoubleSpinBox()
        self.zmax_spin.setDecimals(3)
        self.zmax_spin.setRange(-1e6, 1e6)
        self.zmax_spin.setSingleStep(0.05)
        self.zmax_spin.setValue(self._z_high_default)
        self.zmax_spin.valueChanged.connect(self._schedule)
        form.addRow("Z max (m)", self.zmax_spin)

        self.tristate_check = QtWidgets.QCheckBox("Tristate (use threshold)")
        self.tristate_check.setChecked(True)
        self.tristate_check.toggled.connect(self._on_tristate_toggled)
        form.addRow("Mode", self.tristate_check)

        self.threshold_spin = QtWidgets.QDoubleSpinBox()
        self.threshold_spin.setDecimals(3)
        self.threshold_spin.setRange(-1e6, 1e6)
        self.threshold_spin.setSingleStep(0.05)
        self.threshold_spin.setValue(self._threshold_default)
        self.threshold_spin.valueChanged.connect(self._schedule)
        form.addRow("Height threshold (m)", self.threshold_spin)

        self.min_pts_spin = QtWidgets.QSpinBox()
        self.min_pts_spin.setRange(1, 10_000)
        self.min_pts_spin.setValue(1)
        self.min_pts_spin.valueChanged.connect(self._schedule)
        form.addRow("Min points / cell", self.min_pts_spin)

        self.padding_spin = QtWidgets.QSpinBox()
        self.padding_spin.setRange(0, 1000)
        self.padding_spin.setValue(5)
        self.padding_spin.valueChanged.connect(self._schedule)
        form.addRow("Padding (cells)", self.padding_spin)

        self.unknown_check = QtWidgets.QCheckBox("Empty cells = UNKNOWN (binary mode only)")
        self.unknown_check.setChecked(False)
        self.unknown_check.toggled.connect(self._schedule)
        form.addRow("", self.unknown_check)

        btn_row = QtWidgets.QHBoxLayout()
        self.save_btn = QtWidgets.QPushButton("Save…")
        self.save_btn.setShortcut(QtGui.QKeySequence.Save)
        self.save_btn.clicked.connect(self._on_save)
        self.close_btn = QtWidgets.QPushButton("Close")
        self.close_btn.clicked.connect(self.accept)
        btn_row.addWidget(self.save_btn)
        btn_row.addWidget(self.close_btn)
        form.addRow(btn_row)

        self.status_label = QtWidgets.QLabel("Building preview…")
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet("color: #888;")
        form.addRow(self.status_label)

        # ---- Preview panel
        preview_box = QtWidgets.QGroupBox("Preview (PGM)")
        pv = QtWidgets.QVBoxLayout(preview_box)
        self.preview_label = QtWidgets.QLabel("Building…")
        self.preview_label.setAlignment(QtCore.Qt.AlignCenter)
        self.preview_label.setMinimumSize(480, 360)
        self.preview_label.setStyleSheet(
            "background: #15171c; color: #888; border: 1px solid #2a2d34;"
        )
        pv.addWidget(self.preview_label, 1)
        legend = QtWidgets.QLabel(
            "<span style='color:#000;background:#fff;padding:0 4px;'>occ</span>  "
            "<span style='color:#000;background:#dddddd;padding:0 4px;'>free</span>  "
            "<span style='color:#fff;background:#444;padding:0 4px;'>unknown</span>"
        )
        legend.setAlignment(QtCore.Qt.AlignCenter)
        pv.addWidget(legend)

        outer.addWidget(ctrl, 0)
        outer.addWidget(preview_box, 1)

        self._on_tristate_toggled(self.tristate_check.isChecked())

    # -- state helpers -----------------------------------------------------

    def _on_tristate_toggled(self, on: bool):
        self.threshold_spin.setEnabled(on)
        self.unknown_check.setEnabled(not on)
        self._schedule()

    def _schedule(self, *_):
        # Debounce: collapse a burst of slider drags into one render.
        self._debounce.start()

    def _current_params(self) -> occupancy.GridParams:
        z_lo = float(self.zmin_spin.value())
        z_hi = float(self.zmax_spin.value())
        if z_hi <= z_lo:
            z_hi = z_lo + 1e-3
        return occupancy.GridParams(
            resolution=float(self.resolution_spin.value()),
            z_min=z_lo,
            z_max=z_hi,
            method=self.method_combo.currentText(),
            min_points_per_cell=int(self.min_pts_spin.value()),
            unknown_outside_footprint=self.unknown_check.isChecked()
            and not self.tristate_check.isChecked(),
            padding_cells=int(self.padding_spin.value()),
            height_threshold=(
                float(self.threshold_spin.value())
                if self.tristate_check.isChecked()
                else None
            ),
        )

    # -- preview rebuild ---------------------------------------------------

    def _render_preview(self):
        if self._busy:
            self._dirty = True
            return
        params = self._current_params()
        pcd = self._pcd
        self._busy = True
        self.status_label.setText("Building preview…")

        def work():
            grid = occupancy.build(pcd, params)
            return grid, params

        def on_success(result):
            grid, used = result
            self._last_grid = grid
            self._last_params = used
            self._update_preview_pixmap(grid)
            self._update_status(grid)

        def on_error(msg: str):
            self.status_label.setText(f"<span style='color:#d66;'>Build failed: {msg.splitlines()[0]}</span>")
            self.preview_label.setText("error")

        def on_finished():
            self._busy = False
            self._worker_ref = None
            if self._dirty:
                self._dirty = False
                # Re-render with the latest parameters.
                QtCore.QTimer.singleShot(0, self._render_preview)

        self._worker_ref = run_async(work, on_success=on_success, on_error=on_error, on_finished=on_finished)

    def _update_preview_pixmap(self, grid: occupancy.OccupancyGrid):
        img_u8 = np.ascontiguousarray(grid.image, dtype=np.uint8)
        h, w = img_u8.shape
        qimg = QtGui.QImage(img_u8.data, w, h, w, QtGui.QImage.Format_Grayscale8).copy()
        pix = QtGui.QPixmap.fromImage(qimg)
        lbl = self.preview_label
        scaled = pix.scaled(
            lbl.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.FastTransformation
        )
        lbl.setPixmap(scaled)

    def _update_status(self, grid: occupancy.OccupancyGrid):
        img = grid.image
        n = img.size
        occ = float((img == occupancy.OCC).mean() * 100)
        free = float((img == occupancy.FREE).mean() * 100)
        unk = float((img == occupancy.UNKNOWN).mean() * 100)
        self.status_label.setText(
            f"{grid.width}×{grid.height} cells @ {grid.resolution:.3f} m\n"
            f"occupied {occ:.2f}% · free {free:.2f}% · unknown {unk:.2f}%"
        )

    # -- save --------------------------------------------------------------

    def _on_save(self):
        """Save click. If a parameter change is still pending (debounce
        timer running, or worker mid-build), flush it through first so the
        saved grid corresponds to the *current* spin values — not whatever
        the previous render produced."""
        if self._debounce.isActive() or self._busy:
            self._debounce.stop()
            self._render_preview()
            self.status_label.setText("Building latest preview before saving…")
            QtCore.QTimer.singleShot(50, self._on_save_when_ready)
            return
        self._do_save()

    def _on_save_when_ready(self):
        if self._debounce.isActive() or self._busy:
            QtCore.QTimer.singleShot(50, self._on_save_when_ready)
            return
        self._do_save()

    def _do_save(self):
        if self._last_grid is None or self._last_params is None:
            self.status_label.setText("Nothing to save yet — preview still building.")
            return
        default = self._default_save_path or "map.pgm"
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save map (PGM)", default, "PGM image (*.pgm);;All files (*)"
        )
        if not path:
            return
        pgm_path, yaml_path = occupancy.save(self._last_grid, path, self._last_params)
        self.status_label.setText(
            f"Saved {os.path.basename(str(pgm_path))} + {os.path.basename(str(yaml_path))}"
        )

    # -- resize: rescale the preview pixmap --------------------------------

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        if self._last_grid is not None:
            self._update_preview_pixmap(self._last_grid)
