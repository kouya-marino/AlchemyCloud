"""Bugs from the November review (#9).

* File -> Open over unsaved edits asks first, like closing the window does.
* The dropped non-finite count comes from the load itself; nothing re-reads
  the file on the GUI thread afterwards.
* A closed occupancy preview is deleted and stops following edits.
* The preview's home-cell lookup cannot index past the map edge.
* Viewer tools ignore the mouse while an operation runs, before the delayed
  overlay appears.
* ``ALCHEMYCLOUD_GPU=false`` / ``off`` / ``no`` keep the GPU off.
"""

from __future__ import annotations

import threading
import time

import numpy as np
import open3d as o3d
import pytest
from PyQt5 import QtCore, QtGui, QtTest, QtWidgets

from alchemycloud import occupancy, pcd_ops
from alchemycloud.main_window import MainWindow
from alchemycloud.occupancy_preview import OccupancyPreviewDialog


def _cloud(n=5000, seed=0):
    pts = np.random.default_rng(seed).uniform(0, 5, (n, 3))
    return o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))


def _window(pcd=None):
    w = MainWindow()
    w.prompt_unsaved_on_close = False
    if pcd is not None:
        w.current_pcd = pcd
        w.viewer.set_cloud(pcd)
        w._refresh_state()
    return w


def _pump(qapp, seconds):
    t0 = time.monotonic()
    while time.monotonic() - t0 < seconds:
        qapp.processEvents()
        time.sleep(0.005)


def _run_loop(ms):
    """Spin a real event loop: deleteLater() only fires inside one."""
    loop = QtCore.QEventLoop()
    QtCore.QTimer.singleShot(ms, loop.quit)
    loop.exec_()


def _wait_idle(qapp, w, timeout_s=10.0):
    t0 = time.monotonic()
    while w._busy and time.monotonic() - t0 < timeout_s:
        _pump(qapp, 0.02)
    assert not w._busy


# ---------------------------------------------------------------------------
# 1. File -> Open over unsaved edits
# ---------------------------------------------------------------------------


@pytest.fixture
def open_spies(monkeypatch):
    calls = {"prompt": 0, "file_dialog": 0}
    answer = {"button": QtWidgets.QMessageBox.Cancel}

    def fake_warning(*_a, **_k):
        calls["prompt"] += 1
        return answer["button"]

    def fake_open(*_a, **_k):
        calls["file_dialog"] += 1
        return "", ""

    monkeypatch.setattr(QtWidgets.QMessageBox, "warning", staticmethod(fake_warning))
    monkeypatch.setattr(QtWidgets.QFileDialog, "getOpenFileName", staticmethod(fake_open))
    return calls, answer


def test_open_over_unsaved_edits_asks_and_cancel_keeps_them(qapp, open_spies):
    calls, answer = open_spies
    w = _window(_cloud())
    w.prompt_unsaved_on_close = True
    w._dirty = True
    answer["button"] = QtWidgets.QMessageBox.Cancel
    w.action_open()
    assert calls == {"prompt": 1, "file_dialog": 0}
    assert w._dirty
    w.prompt_unsaved_on_close = False
    w.close()


def test_open_over_unsaved_edits_discard_goes_on_to_the_file_dialog(qapp, open_spies):
    calls, answer = open_spies
    w = _window(_cloud())
    w.prompt_unsaved_on_close = True
    w._dirty = True
    answer["button"] = QtWidgets.QMessageBox.Discard
    w.action_open()
    assert calls == {"prompt": 1, "file_dialog": 1}
    w.prompt_unsaved_on_close = False
    w.close()


def test_open_without_unsaved_edits_does_not_ask(qapp, open_spies):
    calls, _ = open_spies
    w = _window(_cloud())
    w.prompt_unsaved_on_close = True
    w._dirty = False
    w.action_open()
    assert calls == {"prompt": 0, "file_dialog": 1}
    w.close()


# ---------------------------------------------------------------------------
# 2. Dropped-point count without re-reading the file
# ---------------------------------------------------------------------------


def _write_pcd_with_nans(path, n_good=200, n_nan=7):
    rng = np.random.default_rng(3)
    rows = [f"{x:.4f} {y:.4f} {z:.4f}" for x, y, z in rng.uniform(0, 2, (n_good, 3))]
    rows += ["nan nan nan"] * n_nan
    header = (
        "# .PCD v0.7\nVERSION 0.7\nFIELDS x y z\nSIZE 4 4 4\nTYPE F F F\nCOUNT 1 1 1\n"
        f"WIDTH {len(rows)}\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\nPOINTS {len(rows)}\nDATA ascii\n"
    )
    path.write_text(header + "\n".join(rows) + "\n")


def test_load_pcd_counted_reports_what_non_finite_count_does(tmp_path):
    p = tmp_path / "nans.pcd"
    _write_pcd_with_nans(p)
    pcd, dropped = pcd_ops.load_pcd_counted(str(p))
    assert len(pcd.points) == 200
    assert dropped == 7 == pcd_ops.non_finite_count(str(p))
    assert len(pcd_ops.load_pcd(str(p)).points) == 200  # wrapper unchanged


def test_load_pcd_counted_still_rejects_a_truncated_file(tmp_path, monkeypatch):
    # The header promises more points than the body holds, and none of the
    # missing ones were non-finite. (Real short files rarely reach this: Open3D
    # pads a short ASCII body with zeros and returns nothing for a short binary
    # one, so the header count is faked here.)
    p = tmp_path / "short.pcd"
    o3d.io.write_point_cloud(str(p), _cloud(200), write_ascii=False)
    monkeypatch.setattr(pcd_ops, "_declared_point_count", lambda _p: 250)
    with pytest.raises(ValueError, match="truncated"):
        pcd_ops.load_pcd_counted(str(p))


def test_load_reports_dropped_points_without_rereading_the_file(qapp, tmp_path, monkeypatch):
    p = tmp_path / "nans.pcd"
    _write_pcd_with_nans(p)

    def reread(*_a, **_k):
        raise AssertionError("non_finite_count re-reads the file on the GUI thread")

    monkeypatch.setattr(pcd_ops, "non_finite_count", reread)
    w = _window()
    w.load_path(str(p))
    _wait_idle(qapp, w)
    assert len(w.current_pcd.points) == 200
    assert "dropped 7 non-finite" in w.statusBar().currentMessage()
    w.close()


# ---------------------------------------------------------------------------
# 3. Closed occupancy previews are deleted and stop following edits
# ---------------------------------------------------------------------------


@pytest.fixture
def build_calls(monkeypatch):
    calls = []
    real = occupancy.build
    monkeypatch.setattr(occupancy, "build", lambda *a, **k: calls.append(1) or real(*a, **k))
    return calls


def _previews(w):
    return w.findChildren(OccupancyPreviewDialog)


@pytest.mark.parametrize("how", ["close_button", "escape", "reopen"])
def test_closed_preview_is_deleted_and_stops_rebuilding(qapp, build_calls, how):
    qapp.setQuitOnLastWindowClosed(False)
    pcd = _cloud()
    w = _window(pcd)
    for _ in range(3):
        w.action_export_map()
        _run_loop(200)
        dlg = w._occupancy_dialogs[0]
        if how == "close_button":
            dlg.close_btn.click()
        elif how == "escape":
            QtTest.QTest.keyClick(dlg, QtCore.Qt.Key_Escape)
        _run_loop(100)
    if how == "reopen":  # each export closed the one before; close the last
        w._occupancy_dialogs[0].close()
        _run_loop(100)
    assert _previews(w) == []
    assert w._occupancy_dialogs == []
    build_calls.clear()
    w.cloudChanged.emit(pcd)
    _run_loop(400)
    assert build_calls == []
    w.close()


def test_closing_a_preview_mid_build_is_safe(qapp, monkeypatch):
    qapp.setQuitOnLastWindowClosed(False)
    release = threading.Event()
    real = occupancy.build

    def slow_build(*a, **k):
        release.wait(5)
        return real(*a, **k)

    monkeypatch.setattr(occupancy, "build", slow_build)
    w = _window(_cloud())
    w.action_export_map()
    _run_loop(100)
    assert w._occupancy_dialogs[0]._busy  # a build is in flight
    w._occupancy_dialogs[0].close()
    _run_loop(50)
    release.set()  # the detached build finishes after its dialog is gone
    _run_loop(300)
    assert _previews(w) == []
    w.close()


# ---------------------------------------------------------------------------
# 4. Home-cell lookup at the map edge
# ---------------------------------------------------------------------------


def _grid(origin, size=10, res=1.0):
    return occupancy.OccupancyGrid(image=np.full((size, size), occupancy.FREE, np.uint8), resolution=res, origin=origin)


def test_home_in_the_last_half_cell_does_not_index_past_the_map(qapp):
    # Old check: col = 9.7 passes `< 10`, then round(9.7) = 10 -> IndexError.
    # World x = 0 is 10.2 cells from the left edge, so it is outside the map.
    g = _grid((-10.2, -5.0, 0.0))
    assert OccupancyPreviewDialog.home_cell(g) is None
    dlg = OccupancyPreviewDialog(_cloud())
    dlg._update_status(g)
    assert "OUTSIDE" in dlg.status_label.text()
    dlg.close()


@pytest.mark.parametrize(
    ("origin", "want"),
    [
        ((-9.6, -5.0, 0.0), (9, 4)),  # last column
        ((-0.4, -0.4, 0.0), (0, 9)),  # first column, bottom row
        ((-5.0, -9.6, 0.0), (5, 0)),  # top row
    ],
)
def test_home_cell_is_the_cell_containing_world_origin(origin, want):
    assert OccupancyPreviewDialog.home_cell(_grid(origin)) == want


def test_home_cell_agrees_with_pixel_to_world():
    g = _grid((-3.3, -6.7, 0.0), size=20, res=0.5)
    col, row = OccupancyPreviewDialog.home_cell(g)
    # Lower-left corner of that cell in world coordinates, map_server convention.
    x0 = g.origin[0] + col * g.resolution
    y0 = g.origin[1] + (g.height - 1 - row) * g.resolution
    assert x0 <= 0.0 < x0 + g.resolution
    assert y0 <= 0.0 < y0 + g.resolution


# ---------------------------------------------------------------------------
# 5. Viewer tools are locked while an operation runs
# ---------------------------------------------------------------------------


def _press(widget, pos, button):
    ev = QtGui.QMouseEvent(QtCore.QEvent.MouseButtonPress, QtCore.QPointF(pos), button, button, QtCore.Qt.NoModifier)
    QtWidgets.QApplication.sendEvent(widget, ev)
    rel = QtGui.QMouseEvent(
        QtCore.QEvent.MouseButtonRelease, QtCore.QPointF(pos), button, QtCore.Qt.NoButton, QtCore.Qt.NoModifier
    )
    QtWidgets.QApplication.sendEvent(widget, rel)


def test_marker_cannot_be_dragged_before_the_overlay_shows(qapp):
    pcd = _cloud()
    w = _window(pcd)
    w.show()
    _pump(qapp, 0.1)
    w._a_marker.setChecked(True)
    w._on_marker_toggled()
    release = threading.Event()
    w._apply_async("Voxel", lambda: (release.wait(5), pcd_ops.voxel_downsample(pcd, 0.05))[1])
    assert w._busy and not w.loading.isVisible() and w.viewer.tools_locked()
    before = w.viewer.marker_position()
    pos = QtCore.QPoint(w.viewer.width() // 3, w.viewer.height() * 2 // 3)
    _press(w.viewer, pos, QtCore.Qt.LeftButton)
    assert w.viewer.marker_position() == before
    release.set()
    _wait_idle(qapp, w)
    assert not w.viewer.tools_locked()
    _press(w.viewer, pos, QtCore.Qt.LeftButton)
    assert w.viewer.marker_position() != before  # works again once idle
    w.close()


def test_eraser_keeps_working_while_its_own_edits_commit(qapp):
    """The edit queue sets `_busy` without the overlay; it must not lock the
    eraser, or strokes made while earlier ones commit would be dropped."""
    w = _window(_cloud())
    w._a_eraser.setChecked(True)
    w._on_eraser_toggled()
    w._submit_mask_edit("Eraser", lambda p: p[:, 0] < 0.5)
    assert w._mask_active and w._busy
    assert not w.viewer.tools_locked()
    _wait_idle(qapp, w)
    w.close()


def test_locking_abandons_a_drag_in_progress(qapp):
    w = _window(_cloud())
    v = w.viewer
    v.set_crop_mode(True)
    ev = QtGui.QMouseEvent(
        QtCore.QEvent.MouseButtonPress,
        QtCore.QPointF(10, 10),
        QtCore.Qt.LeftButton,
        QtCore.Qt.LeftButton,
        QtCore.Qt.NoModifier,
    )
    QtWidgets.QApplication.sendEvent(v, ev)
    assert v._crop_origin is not None
    v.set_tools_locked(True)
    assert v._crop_origin is None
    v.set_tools_locked(False)
    w.close()


# ---------------------------------------------------------------------------
# 6. GPU opt-in parsing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("value", ["false", "False", "OFF", "no", " 0 "])
def test_gpu_stays_off_for_false_like_values(monkeypatch, value):
    monkeypatch.setenv(pcd_ops.GPU_ENV, value)
    monkeypatch.setattr(pcd_ops, "cuda_available", lambda: True)
    assert pcd_ops.cuda_device() is None


@pytest.mark.parametrize("value", ["1", "true", "yes", "on"])
def test_gpu_turns_on_for_true_like_values(monkeypatch, value):
    monkeypatch.setenv(pcd_ops.GPU_ENV, value)
    monkeypatch.setattr(pcd_ops, "cuda_available", lambda: True)
    assert pcd_ops.cuda_device() is not None
