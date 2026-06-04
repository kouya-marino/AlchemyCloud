"""GUI smoke tests under offscreen Qt.

Validates that MainWindow + dialogs construct and the async pipeline
fires `_busy → not _busy` end-to-end. Doesn't try to render anything —
the offscreen platform has a stub GL context that's good enough for
the geometry-only paths we care about.
"""
from __future__ import annotations

import os
import tempfile

import copy
import numpy as np
import open3d as o3d
import pytest
from PyQt5 import QtCore, QtWidgets


# Helper: build a MainWindow with a planted cloud already attached.
def _seed_window(qapp, pcd):
    from alchemycloud.main_window import MainWindow
    w = MainWindow()
    w.show()
    qapp.processEvents()
    w.original_pcd = copy.deepcopy(pcd)
    w.current_pcd = pcd
    w.current_path = "<synth>"
    w._refresh_state()
    disp, pts, rgba = w.viewer.prepare_cloud(w.current_pcd)
    w.viewer.apply_prepared_cloud(disp, pts, rgba, reset_view=True)
    qapp.processEvents()
    return w


# ---------------------------------------------------------------------------

def test_main_window_constructs(qapp):
    from alchemycloud.main_window import MainWindow
    w = MainWindow()
    w.show()
    qapp.processEvents()
    toolbar_names = sorted(tb.objectName() for tb in w.findChildren(QtWidgets.QToolBar))
    assert toolbar_names == [
        "EditToolbar", "MarkerToolbar", "OccupancyToolbar", "ToolsToolbar",
    ]
    # Color combo defaults to Z, display LOD switch is off (= LOD on).
    assert w.color_combo.currentText() == "Z"
    assert not w.display_switch.isChecked()
    w.close()


def test_load_path_populates_state(qapp, busy_wait, cube_cloud, tmp_path):
    # Write the synthetic cloud out so load_path has a real file.
    path = str(tmp_path / "cube.pcd")
    o3d.io.write_point_cloud(path, cube_cloud)
    from alchemycloud.main_window import MainWindow
    w = MainWindow()
    w.show()
    qapp.processEvents()
    w.load_path(path)
    assert busy_wait(lambda: not w._busy and w.current_pcd is not None,
                     timeout_s=5.0)
    assert len(w.current_pcd.points) == len(cube_cloud.points)
    assert w.current_path == path
    w.close()


def test_apply_filter_updates_history(qapp, busy_wait, cube_cloud):
    w = _seed_window(qapp, cube_cloud)
    assert len(w.history) == 0
    # Skip the parameter dialog by calling the worker directly via _apply_async.
    from alchemycloud import pcd_ops
    cur = w.current_pcd
    w._apply_async("Voxel test", lambda: pcd_ops.voxel_downsample(cur, 0.2))
    assert busy_wait(lambda: not w._busy, timeout_s=5.0)
    assert len(w.history) == 1
    # Cloud should now have fewer points.
    assert len(w.current_pcd.points) < len(cube_cloud.points)
    w.close()


def test_undo_redo_round_trip(qapp, busy_wait, cube_cloud):
    w = _seed_window(qapp, cube_cloud)
    n_before = len(w.current_pcd.points)
    from alchemycloud import pcd_ops
    cur = w.current_pcd
    w._apply_async("Voxel test", lambda: pcd_ops.voxel_downsample(cur, 0.2))
    assert busy_wait(lambda: not w._busy, timeout_s=5.0)
    n_after = len(w.current_pcd.points)
    assert n_after < n_before

    w.action_undo()
    assert busy_wait(lambda: not w._busy, timeout_s=5.0)
    assert len(w.current_pcd.points) == n_before
    assert len(w.redo_stack) == 1

    w.action_redo()
    assert busy_wait(lambda: not w._busy, timeout_s=5.0)
    assert len(w.current_pcd.points) == n_after
    w.close()


def test_occupancy_preview_renders_initial_grid(qapp, busy_wait, floor_with_chair):
    from alchemycloud.main_window import MainWindow
    from alchemycloud.occupancy_preview import OccupancyPreviewDialog
    w = MainWindow()
    w.show()
    w.original_pcd = copy.deepcopy(floor_with_chair)
    w.current_pcd = floor_with_chair
    w.current_path = "<synth>"
    w._refresh_state()
    qapp.processEvents()
    dlg = OccupancyPreviewDialog(floor_with_chair, default_save_path="/tmp/x.pgm",
                                 parent=w)
    dlg.setModal(False)
    dlg.show()
    assert busy_wait(lambda: dlg._last_grid is not None, timeout_s=5.0)
    # Default mode is tristate.
    assert dlg._last_params.height_threshold is not None
    assert dlg._last_grid.image.size > 0
    dlg.close()
    w.close()


def test_marker_coord_entry_moves_marker(qapp, cube_cloud):
    w = _seed_window(qapp, cube_cloud)
    # Type coordinates → marker jumps live.
    w.marker_x_spin.setValue(0.5)
    w.marker_y_spin.setValue(-0.25)
    w.marker_yaw_spin.setValue(45.0)
    qapp.processEvents()
    px, py, _ = w.viewer.marker_position()
    assert abs(px - 0.5) < 1e-6
    assert abs(py - (-0.25)) < 1e-6
    assert abs(w.viewer.marker_yaw_degrees() - 45.0) < 1e-6
    w.close()


def test_marker_drag_syncs_back_to_spins(qapp, cube_cloud):
    w = _seed_window(qapp, cube_cloud)
    # Moving the marker via the viewer API should mirror into the spins
    # (this is the path mouse drags / Reset Marker take).
    w.viewer.set_marker_position(0.3, 0.2)
    w.viewer.set_marker_yaw(np.radians(90.0))
    qapp.processEvents()
    assert abs(w.marker_x_spin.value() - 0.3) < 1e-6
    assert abs(w.marker_y_spin.value() - 0.2) < 1e-6
    assert abs(w.marker_yaw_spin.value() - 90.0) < 1e-6
    # Reset marker → spins return to 0.
    w.action_reset_marker()
    qapp.processEvents()
    assert abs(w.marker_x_spin.value()) < 1e-6
    assert abs(w.marker_y_spin.value()) < 1e-6
    assert abs(w.marker_yaw_spin.value()) < 1e-6
    w.close()


def test_marker_spin_ranges_clamp_to_cloud_scope(qapp, cube_cloud):
    # cube_cloud spans roughly [-1, 1]^3; ranges should track those bounds
    # (expanded to include 0.0). Allow a little float slack.
    w = _seed_window(qapp, cube_cloud)
    assert w.marker_x_spin.maximum() <= 1.01
    assert w.marker_x_spin.minimum() >= -1.01
    assert w.marker_y_spin.maximum() <= 1.01
    assert w.marker_y_spin.minimum() >= -1.01
    # Yaw is a fixed ±180 range.
    assert w.marker_yaw_spin.minimum() == -180.0
    assert w.marker_yaw_spin.maximum() == 180.0
    w.close()


def test_point_size_spin_updates_viewer(qapp, cube_cloud):
    from alchemycloud import pcd_ops
    w = _seed_window(qapp, cube_cloud)
    assert w.viewer.point_size() == 1.5  # default
    w.point_size_spin.setValue(6.0)
    qapp.processEvents()
    assert w.viewer.point_size() == 6.0
    assert w.viewer._scatter.size == 6.0
    # Size must persist across a re-prepare (filter / reload).
    cur = w.current_pcd
    w._apply_async("Voxel test", lambda: pcd_ops.voxel_downsample(cur, 0.2))
    # drain
    import time
    t0 = time.monotonic()
    while w._busy and time.monotonic() - t0 < 5.0:
        qapp.processEvents(); time.sleep(0.02)
    assert w.viewer.point_size() == 6.0
    assert w.viewer._scatter.size == 6.0
    w.close()


def test_brush_shape_combo_mirrors_viewer(qapp, cube_cloud):
    w = _seed_window(qapp, cube_cloud)
    assert w.viewer.brush_shape() == "circle"
    w.eraser_shape_combo.setCurrentText("Rectangle")
    qapp.processEvents()
    assert w.viewer.brush_shape() == "rectangle"
    assert w.eraser_aspect_spin.isEnabled()
    w.eraser_shape_combo.setCurrentText("Square")
    qapp.processEvents()
    assert w.viewer.brush_shape() == "square"
    assert not w.eraser_aspect_spin.isEnabled()
    w.close()
