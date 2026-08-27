"""Regression tests for the audit fixes that live in the Qt layer.

Each test names the defect it pins. Grouped by the file the fix lives in.
"""

from __future__ import annotations

import numpy as np
import open3d as o3d
import OpenGL.GL as GL
import pytest
from PyQt5 import QtCore, QtGui, QtWidgets

from alchemycloud import pcd_ops
from alchemycloud.embedded_viewer import EmbeddedViewer
from alchemycloud.main_window import HISTORY_DEPTH, MainWindow


def _cloud(n=800, seed=0, offset=0.0):
    rng = np.random.default_rng(seed)
    return o3d.geometry.PointCloud(o3d.utility.Vector3dVector(rng.uniform(0, 2, (n, 3)) + offset))


def _window(qapp, cloud=None):
    w = MainWindow()
    w.prompt_unsaved_on_close = False
    if cloud is not None:
        w.current_pcd = cloud
        w.original_pcd = cloud
        w.viewer.set_cloud(cloud)
        w._refresh_state()
    return w


# ---------------------------------------------------------------------------
# embedded_viewer: GL state
# ---------------------------------------------------------------------------


def test_point_cloud_is_depth_tested_and_not_additively_blended(qapp):
    """pyqtgraph defaults GLScatterPlotItem to 'additive': depth test off and
    glBlendFunc(SRC_ALPHA, ONE), so overlapping points summed to white and
    nothing occluded anything."""
    v = EmbeddedViewer()
    opts = v._scatter._GLGraphicsItem__glOpts
    assert opts.get(GL.GL_DEPTH_TEST) is True
    assert opts.get(GL.GL_BLEND) is False
    assert "glBlendFunc" not in opts


# ---------------------------------------------------------------------------
# embedded_viewer: marker survives ordinary edits
# ---------------------------------------------------------------------------


def test_ordinary_edit_does_not_reset_the_marker(qapp):
    v = EmbeddedViewer()
    v.set_cloud(_cloud())
    v.set_marker_position(1.25, -0.5)
    v.set_marker_yaw(np.radians(37.0))
    before = (v.marker_position(), v.marker_yaw_degrees())

    v.set_cloud(_cloud(seed=1))  # e.g. the result of a voxel downsample
    assert v.marker_position() == pytest.approx(before[0])
    assert v.marker_yaw_degrees() == pytest.approx(before[1])


def test_reset_marker_flag_still_zeroes_it(qapp):
    v = EmbeddedViewer()
    v.set_cloud(_cloud())
    v.set_marker_position(1.25, -0.5)
    v.set_cloud(_cloud(seed=2), reset_marker=True)
    assert v.marker_position()[:2] == pytest.approx((0.0, 0.0))


# ---------------------------------------------------------------------------
# embedded_viewer: unprojection and eraser input
# ---------------------------------------------------------------------------


def test_screen_to_world_rejects_a_ray_that_never_reaches_the_plane(qapp):
    """t < 0 means the plane is behind the camera; returning that intersection
    put the marker at a mirrored point behind the viewer."""
    v = EmbeddedViewer()
    v.resize(400, 300)
    v.setCameraPosition(distance=10.0, elevation=20.0, azimuth=0.0)
    qapp.processEvents()
    # Far above the horizon: the ray goes up, away from z=0.
    assert v.screen_to_world_xy(QtCore.QPoint(200, 1), 0.0) is None
    # Through the middle of the scene it must still hit.
    assert v.screen_to_world_xy(QtCore.QPoint(200, 150), 0.0) is not None


def test_eraser_mode_leaves_other_mouse_buttons_to_the_camera(qapp):
    """Every button used to be swallowed, so the view could not be orbited
    without toggling the eraser off."""
    v = EmbeddedViewer()
    v.resize(400, 300)
    v.set_eraser_mode(True)
    right = QtGuiMouseEvent(v, QtCore.Qt.RightButton)
    v.mousePressEvent(right)
    # A right press must not start a stroke.
    assert v._eraser_path is None
    left = QtGuiMouseEvent(v, QtCore.Qt.LeftButton)
    v.mousePressEvent(left)
    assert v._eraser_path is not None


def QtGuiMouseEvent(widget, button):
    return QtGui.QMouseEvent(
        QtCore.QEvent.MouseButtonPress,
        QtCore.QPointF(10.0, 10.0),
        button,
        button,
        QtCore.Qt.NoModifier,
    )


def test_eraser_stroke_interpolates_across_a_fast_drag(qapp):
    """Qt delivers one move event per repaint, so a fast drag arrived as widely
    spaced samples and erased a dotted line."""
    v = EmbeddedViewer()
    v.resize(400, 300)
    v.set_eraser_mode(True)
    v.set_eraser_brush(20)
    v.mousePressEvent(QtGuiMouseEvent(v, QtCore.Qt.LeftButton))
    far = QtGui.QMouseEvent(
        QtCore.QEvent.MouseMove,
        QtCore.QPointF(210.0, 10.0),  # a 200 px jump
        QtCore.Qt.LeftButton,
        QtCore.Qt.LeftButton,
        QtCore.Qt.NoModifier,
    )
    v.mouseMoveEvent(far)
    xs = [p.x() for p in v._eraser_path]
    assert len(v._eraser_path) > 5, "no interpolation happened"
    assert max(np.diff(sorted(xs))) <= 20, "gap wider than the brush remains"


# ---------------------------------------------------------------------------
# main_window: history, dirty state, close
# ---------------------------------------------------------------------------


def test_loading_a_second_file_clears_the_redo_stack(qapp, tmp_path):
    """Redo stayed enabled across a load and popped a cloud from the *previous*
    file into current_pcd, while current_path still named the new one."""
    a = tmp_path / "a.pcd"
    b = tmp_path / "b.pcd"
    pcd_ops.save_pcd(_cloud(500, seed=1), str(a))
    pcd_ops.save_pcd(_cloud(300, seed=2, offset=100.0), str(b))

    w = _window(qapp, _cloud())
    w.history.append(_cloud(seed=3))
    w.redo_stack.append(_cloud(seed=4))
    assert w.redo_stack

    w.load_path(str(a))
    _drain(qapp, w)
    assert w.history == []
    assert w.redo_stack == []
    assert not w._dirty


def test_history_is_capped(qapp):
    w = _window(qapp, _cloud())
    for i in range(HISTORY_DEPTH + 6):
        w.current_pcd = _cloud(seed=i)
        w._push_history()
    assert len(w.history) == HISTORY_DEPTH


def test_edits_mark_the_document_dirty_and_saving_clears_it(qapp, tmp_path):
    w = _window(qapp, _cloud())
    assert not w._dirty
    cur = w.current_pcd
    w._apply_async("Voxel", lambda: pcd_ops.voxel_downsample(cur, 0.3))
    _drain(qapp, w)
    assert w._dirty
    w._on_saved(str(tmp_path / "x.pcd"), "Saved to")
    assert not w._dirty


def test_close_prompts_when_dirty_and_respects_cancel(qapp, monkeypatch):
    w = MainWindow()
    w.current_pcd = _cloud()
    w._dirty = True
    asked = []
    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "warning",
        lambda *a, **k: (asked.append(True), QtWidgets.QMessageBox.Cancel)[1],
    )
    ev = QtGui.QCloseEvent()
    w.closeEvent(ev)
    assert asked, "no prompt shown for unsaved edits"
    assert not ev.isAccepted()

    asked.clear()
    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "warning",
        lambda *a, **k: (asked.append(True), QtWidgets.QMessageBox.Discard)[1],
    )
    ev2 = QtGui.QCloseEvent()
    w.closeEvent(ev2)
    assert asked


def test_close_does_not_prompt_when_clean(qapp, monkeypatch):
    w = MainWindow()
    w.current_pcd = _cloud()
    w._dirty = False
    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "warning",
        lambda *a, **k: pytest.fail("prompted with no unsaved changes"),
    )
    w.closeEvent(QtGui.QCloseEvent())


# ---------------------------------------------------------------------------
# main_window: busy gating and mode exclusivity
# ---------------------------------------------------------------------------


def test_menu_actions_are_disabled_while_busy(qapp):
    """The loading overlay only eats mouse events on the canvas, so menu items
    and their shortcuts stayed live during a long operation."""
    w = _window(qapp, _cloud())
    assert all(a.isEnabled() for a in w._menu_actions)
    w._busy = True
    w._refresh_state()
    assert not any(a.isEnabled() for a in w._menu_actions)
    w._busy = False
    w._refresh_state()
    assert all(a.isEnabled() for a in w._menu_actions)


def test_enabling_crop_unchecks_the_eraser_button(qapp):
    """The viewer already cleared eraser state; the button kept its checked
    look, so the toolbar showed two active tools."""
    w = _window(qapp, _cloud())
    w._a_eraser.setChecked(True)
    w._on_eraser_toggled()
    assert w._a_eraser.isChecked()
    w._a_crop.setChecked(True)
    w._on_crop_toggled()
    assert not w._a_eraser.isChecked()
    assert not w.viewer._eraser_mode


def test_marker_spin_range_keeps_a_marker_placed_outside_the_cloud(qapp):
    """setRange used to clamp the marker inside the cloud bbox, so editing yaw
    alone teleported a marker parked just outside the scanned area."""
    w = _window(qapp, _cloud())
    w.viewer.set_marker_position(50.0, -50.0)
    w._refresh_state()
    assert w.marker_x_spin.minimum() <= 50.0 <= w.marker_x_spin.maximum()
    assert w.marker_y_spin.minimum() <= -50.0 <= w.marker_y_spin.maximum()
    assert w.viewer.marker_position()[0] == pytest.approx(50.0)


def test_slab_scope_uses_the_viewer_bounds_not_the_raw_spins(qapp):
    """The viewer normalises an inverted low/high pair; crop and eraser read the
    raw spin boxes, so an inverted slab highlighted points it could not edit."""
    w = _window(qapp, _cloud())
    w._a_zslab.setChecked(True)
    w.viewer.set_zslab_visible(True)
    w.viewer.set_zslab_bounds(1.5, 0.5)  # deliberately inverted
    lo, hi = w.viewer.slab_bounds()
    assert lo < hi, "viewer should normalise the pair"


# ---------------------------------------------------------------------------
# occupancy_preview
# ---------------------------------------------------------------------------


def test_preview_follows_the_cloud_it_is_given(qapp):
    """The dialog is non-modal, so the cloud can change under it; it used to
    keep exporting the snapshot it was constructed with."""
    from alchemycloud.occupancy_preview import OccupancyPreviewDialog

    first = _cloud(600, seed=1)
    dlg = OccupancyPreviewDialog(first)
    assert dlg._pcd is first
    second = _cloud(600, seed=2, offset=10.0)
    dlg.set_cloud(second)
    assert dlg._pcd is second
    assert dlg._last_grid is None  # forced to rebuild
    dlg.close()


def test_preview_refuses_to_save_a_stale_grid_after_a_failed_build(qapp, monkeypatch):
    """A failed rebuild left _last_grid holding the previous success, and the
    pending-save path wrote that and reported "Saved"."""
    from alchemycloud.occupancy_preview import OccupancyPreviewDialog

    dlg = OccupancyPreviewDialog(_cloud(600))
    dlg._last_grid = object()
    dlg._last_params = object()
    dlg._last_error = "No points in the requested Z range"
    monkeypatch.setattr(
        QtWidgets.QFileDialog,
        "getSaveFileName",
        lambda *a, **k: pytest.fail("should not have reached the file dialog"),
    )
    dlg._do_save()
    assert "Not saved" in dlg.status_label.text()
    dlg.close()


def test_preview_close_stops_the_debounce_and_pending_save(qapp):
    """The old 50 ms polling chain kept firing after the dialog closed and could
    pop a file dialog from a dead window."""
    from alchemycloud.occupancy_preview import OccupancyPreviewDialog

    dlg = OccupancyPreviewDialog(_cloud(600))
    dlg._schedule()
    dlg._save_pending = True
    dlg.close()
    assert not dlg._debounce.isActive()
    assert not dlg._save_pending


def test_binary_and_tristate_backgrounds_have_separate_controls(qapp):
    """One shared boolean flipped the binary default from FREE to UNKNOWN,
    exporting a map with zero free cells."""
    from alchemycloud.occupancy_preview import OccupancyPreviewDialog

    dlg = OccupancyPreviewDialog(_cloud(600))
    dlg.tristate_check.setChecked(False)
    dlg._on_tristate_toggled(False)
    params = dlg._current_params()
    assert params.height_threshold is None
    assert params.unknown_outside_footprint is False  # binary default is FREE
    assert dlg.binary_bg_combo.isEnabled()
    assert not dlg.tristate_bg_combo.isEnabled()

    dlg.tristate_check.setChecked(True)
    dlg._on_tristate_toggled(True)
    params = dlg._current_params()
    assert params.tristate_empty_unknown is True  # tristate default is UNKNOWN
    assert dlg.tristate_bg_combo.isEnabled()
    assert not dlg.binary_bg_combo.isEnabled()
    dlg.close()


# ---------------------------------------------------------------------------
# packaging / entry point
# ---------------------------------------------------------------------------


def test_version_is_single_sourced():
    """__version__ must agree with pyproject.toml, whether installed or not."""
    from alchemycloud import __version__, _version_from_source_tree

    declared = _version_from_source_tree()
    assert declared, "could not read version from pyproject.toml"
    assert __version__ == declared


def test_excepthook_logs_and_is_installed(qapp, monkeypatch):
    from alchemycloud import app as app_mod

    shown = []
    monkeypatch.setattr(QtWidgets.QMessageBox, "critical", lambda *a, **k: shown.append(a[2]))
    original = __import__("sys").excepthook
    try:
        app_mod.install_excepthook(None)
        hook = __import__("sys").excepthook
        assert hook is not original
        try:
            raise RuntimeError("boom")
        except RuntimeError:
            import sys as _s

            hook(*_s.exc_info())
        assert shown and "boom" in shown[0]
    finally:
        __import__("sys").excepthook = original


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _drain(qapp, w, timeout_s: float = 5.0):
    import time

    t0 = time.monotonic()
    while w._busy and time.monotonic() - t0 < timeout_s:
        qapp.processEvents()
        time.sleep(0.02)
    qapp.processEvents()
