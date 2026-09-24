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

from . import occupancy, pcd_ops
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
        self._last_error: str | None = None
        self._save_pending = False
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

        # Two separate controls, because the two policies mean different things
        # by "empty". In tristate it is *never observed* (observed floor stays
        # FREE), so grey is the sensible default. In binary every non-occupied
        # cell is empty, so grey there paints the navigable interior too and the
        # map ends up with zero FREE cells — which is why this one defaults to
        # Free and matches `GridParams.unknown_outside_footprint=False`.
        self.tristate_bg_combo = QtWidgets.QComboBox()
        self.tristate_bg_combo.addItems(["Unknown (grey)", "Free (white)"])
        self.tristate_bg_combo.setCurrentIndex(0)
        self.tristate_bg_combo.setToolTip(
            "Colour for cells the scan never observed. Only the tristate policy "
            "can tell observed-and-empty from never-observed."
        )
        self.tristate_bg_combo.currentIndexChanged.connect(self._schedule)
        form.addRow("Unobserved cells (tristate)", self.tristate_bg_combo)

        self.binary_bg_combo = QtWidgets.QComboBox()
        self.binary_bg_combo.addItems(["Free (white)", "Unknown (grey)"])
        self.binary_bg_combo.setCurrentIndex(0)
        self.binary_bg_combo.setToolTip(
            "Colour for every cell that is not an obstacle. Binary mode has no "
            "observed/unobserved distinction, so 'Unknown' greys out the free "
            "space too and the exported map contains no FREE cells at all."
        )
        self.binary_bg_combo.currentIndexChanged.connect(self._schedule)
        form.addRow("Non-obstacle cells (binary)", self.binary_bg_combo)

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
        self.preview_label.setStyleSheet("background: #15171c; color: #888; border: 1px solid #2a2d34;")
        pv.addWidget(self.preview_label, 1)
        # Swatches must match the rendered PGM: occupied is 0 (black), free is
        # 254 (white), unknown is 205 (light grey). They were previously the
        # other way round, so the key contradicted the image beside it.
        legend = QtWidgets.QLabel(
            "<span style='color:#fff;background:#000;padding:0 4px;'>occ&nbsp;0</span>  "
            "<span style='color:#000;background:#fefefe;padding:0 4px;'>free&nbsp;254</span>  "
            "<span style='color:#000;background:#cdcdcd;padding:0 4px;'>unknown&nbsp;205</span>"
        )
        legend.setAlignment(QtCore.Qt.AlignCenter)
        pv.addWidget(legend)

        outer.addWidget(ctrl, 0)
        outer.addWidget(preview_box, 1)

        self._on_tristate_toggled(self.tristate_check.isChecked())

    # -- state helpers -----------------------------------------------------

    def set_cloud(self, pcd) -> None:
        """Point the dialog at a new cloud and re-render.

        The dialog is non-modal, so the main window stays editable while it is
        open. It used to hold the cloud it was constructed with forever, so
        applying a Z-clip and then hitting Save exported the pre-clip map.
        MainWindow connects its `cloudChanged` signal here.
        """
        if pcd is None or len(pcd.points) == 0:
            return
        self._pcd = pcd
        self._last_grid = None
        self._last_params = None
        self._last_error = None
        self.status_label.setText("Cloud changed — rebuilding preview…")
        self._schedule()

    def closeEvent(self, event):
        """Stop the debounce timer and drop any pending save before teardown."""
        self._debounce.stop()
        self._save_pending = False
        self._dirty = False
        super().closeEvent(event)

    def _on_tristate_toggled(self, on: bool):
        self.threshold_spin.setEnabled(on)
        self.tristate_bg_combo.setEnabled(on)
        self.binary_bg_combo.setEnabled(not on)
        self._schedule()

    def _schedule(self, *_):
        # Debounce: collapse a burst of slider drags into one render.
        self._debounce.start()

    def _current_params(self) -> occupancy.GridParams:
        z_lo = float(self.zmin_spin.value())
        z_hi = float(self.zmax_spin.value())
        if z_hi <= z_lo:
            z_hi = z_lo + 1e-3
        tristate = self.tristate_check.isChecked()
        return occupancy.GridParams(
            resolution=float(self.resolution_spin.value()),
            z_min=z_lo,
            z_max=z_hi,
            method=self.method_combo.currentText(),
            min_points_per_cell=int(self.min_pts_spin.value()),
            # index 1 == "Unknown (grey)" for the binary combo
            unknown_outside_footprint=self.binary_bg_combo.currentIndex() == 1,
            # index 0 == "Unknown (grey)" for the tristate combo
            tristate_empty_unknown=self.tristate_bg_combo.currentIndex() == 0,
            padding_cells=int(self.padding_spin.value()),
            height_threshold=float(self.threshold_spin.value()) if tristate else None,
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
            self._last_error = None
            self._update_preview_pixmap(grid)
            self._update_status(grid)

        def on_error(msg: str):
            # Keep the failure on record. `_do_save` refuses to write while this
            # is set: previously a failed rebuild left `_last_grid` holding the
            # *previous* successful render, and the pending-save path then wrote
            # that stale grid and reported "Saved".
            self._last_error = msg
            self.status_label.setText(f"<span style='color:#d66;'>Build failed: {msg.splitlines()[0]}</span>")
            self.preview_label.setText("error")

        def on_finished():
            self._busy = False
            self._worker_ref = None
            if self._dirty:
                self._dirty = False
                # Re-render with the latest parameters.
                QtCore.QTimer.singleShot(0, self._render_preview)
            elif self._save_pending:
                self._save_pending = False
                if self._last_error is None:
                    self._do_save()
                else:
                    self.status_label.setText(
                        "<span style='color:#d66;'>Not saved — the current parameters "
                        "do not produce a grid. Fix them and try again.</span>"
                    )

        self._worker_ref = run_async(work, on_success=on_success, on_error=on_error, on_finished=on_finished)

    @staticmethod
    def home_pixel(grid: occupancy.OccupancyGrid) -> tuple[float, float]:
        """Image (col, row) of world (0, 0) — the robot's home pose.

        Not the same thing as the YAML ``origin``, which is the world position
        of the image's lower-left *corner*. Home is where "Apply Marker as
        Origin" puts the robot, and on a map whose SLAM origin sits outside the
        surveyed floor it can land in unknown space — worth seeing before export.
        """
        ox, oy, _ = grid.origin
        res = grid.resolution
        col = (0.0 - ox) / res - 0.5
        row = (grid.height - 1) - ((0.0 - oy) / res - 0.5)
        return col, row

    def _update_preview_pixmap(self, grid: occupancy.OccupancyGrid):
        img_u8 = np.ascontiguousarray(grid.image, dtype=np.uint8)
        h, w = img_u8.shape
        qimg = QtGui.QImage(img_u8.data, w, h, w, QtGui.QImage.Format_Grayscale8).copy()
        pix = QtGui.QPixmap.fromImage(qimg)
        lbl = self.preview_label
        scaled = pix.scaled(lbl.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.FastTransformation)

        # Home marker is drawn AFTER scaling, at a fixed pixel size. Drawing it
        # on the full-resolution map first would shrink it with everything else
        # — on a 2275 x 3566 grid scaled into a 480 px label it would vanish.
        col, row = self.home_pixel(grid)
        if 0 <= col < w and 0 <= row < h and scaled.width() > 0:
            sx = col * scaled.width() / w
            sy = row * scaled.height() / h
            painter = QtGui.QPainter(scaled)
            painter.setRenderHint(QtGui.QPainter.Antialiasing, True)
            arm = 9.0
            # Dark halo first so the cross survives on white (free) areas, then
            # cyan on top — the map is strictly 0/205/254 grey, so a saturated
            # hue cannot be mistaken for map content.
            for colour, width in ((QtGui.QColor(0, 0, 0, 200), 4.0), (QtGui.QColor(0, 229, 255), 2.0)):
                pen = QtGui.QPen(colour, width)
                pen.setCapStyle(QtCore.Qt.RoundCap)
                painter.setPen(pen)
                painter.drawLine(QtCore.QPointF(sx - arm, sy), QtCore.QPointF(sx + arm, sy))
                painter.drawLine(QtCore.QPointF(sx, sy - arm), QtCore.QPointF(sx, sy + arm))
                painter.drawEllipse(QtCore.QPointF(sx, sy), arm * 0.45, arm * 0.45)
            painter.end()

        lbl.setPixmap(scaled)

    def _update_status(self, grid: occupancy.OccupancyGrid):
        img = grid.image
        occ = float((img == occupancy.OCC).mean() * 100)
        free = float((img == occupancy.FREE).mean() * 100)
        unk = float((img == occupancy.UNKNOWN).mean() * 100)
        col, row = self.home_pixel(grid)
        if 0 <= col < grid.width and 0 <= row < grid.height:
            value = int(img[int(round(row)), int(round(col))])
            where = {occupancy.OCC: "OCCUPIED", occupancy.FREE: "free", occupancy.UNKNOWN: "UNKNOWN"}.get(
                value, str(value)
            )
            note = f"home (0,0) at px {int(col)},{int(row)} — on {where}"
            # A home in unknown or occupied space is a real problem: the robot
            # would localise into grey, or inside an obstacle.
            colour = "#888" if value == occupancy.FREE else "#d66"
        else:
            note = "home (0,0) is OUTSIDE the map — apply the marker first"
            colour = "#d66"

        # <br>, not \n: the span puts QLabel into rich-text mode, where a
        # literal newline is collapsed like any other whitespace.
        self.status_label.setText(
            f"{grid.width}×{grid.height} cells @ {grid.resolution:.3f} m<br>"
            f"occupied {occ:.2f}% · free {free:.2f}% · unknown {unk:.2f}%<br>"
            f"<span style='color:{colour};'>{note}</span>"
        )

    # -- save --------------------------------------------------------------

    def _on_save(self):
        """Save click. If a parameter change is still pending (debounce timer
        running, or a worker mid-build), flush it first so the saved grid
        matches the *current* spin values.

        Implemented as a flag consumed by the render's `on_finished` rather than
        a self-rescheduling timer. The old 50 ms polling chain kept firing after
        the dialog was closed — eventually popping a file dialog and writing a
        file from a dead window — and had no way to tell a successful flush from
        a failed one.
        """
        if self._debounce.isActive() or self._busy:
            self._save_pending = True
            self._debounce.stop()
            self.status_label.setText("Building latest preview before saving…")
            if not self._busy:
                self._render_preview()
            return
        self._do_save()

    def _do_save(self):
        if self._last_grid is None or self._last_params is None:
            self.status_label.setText("Nothing to save yet — preview still building.")
            return
        if self._last_error is not None:
            self.status_label.setText(
                "<span style='color:#d66;'>Not saved — the last build failed, so the "
                "only grid available is out of date.</span>"
            )
            return
        default = self._default_save_path or "map.pgm"
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save map (PGM)", default, "PGM image (*.pgm);;All files (*)"
        )
        if not path:
            return
        try:
            pgm_path, yaml_path = occupancy.save(self._last_grid, path, self._last_params)
        except Exception as exc:
            # This runs on the main thread, so Worker's error plumbing does not
            # apply: an unwritable path used to raise straight out of the slot,
            # print a traceback nobody sees, and leave the old stats on screen.
            QtWidgets.QMessageBox.warning(self, "Export failed", f"{type(exc).__name__}: {exc}")
            self.status_label.setText(f"<span style='color:#d66;'>Export failed: {exc}</span>")
            return
        self.status_label.setText(f"Saved {os.path.basename(str(pgm_path))} + {os.path.basename(str(yaml_path))}")

    # -- resize: rescale the preview pixmap --------------------------------

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        if self._last_grid is not None:
            self._update_preview_pixmap(self._last_grid)
