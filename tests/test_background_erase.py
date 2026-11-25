"""Eraser / crop run in the background without locking the window.

An eraser stroke on a 1.8 M-point cloud used to freeze the whole UI for ~2 s.
Point-removal edits now hide the points on the display at once and commit the
full-cloud edit on a worker, one queued job at a time. These tests pin the
invariants that design depends on.
"""

from __future__ import annotations

import time

import numpy as np
import open3d as o3d
import pytest
from PyQt5 import QtCore, QtGui

from alchemycloud import pcd_ops
from alchemycloud.embedded_viewer import _project_columns, project_points
from alchemycloud.main_window import MainWindow


def _cloud(n=2000, seed=0, colors=True, normals=True):
    rng = np.random.default_rng(seed)
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(rng.uniform(0, 2, (n, 3))))
    if colors:
        pcd.colors = o3d.utility.Vector3dVector(rng.uniform(0, 1, (n, 3)))
    if normals:
        nrm = rng.normal(size=(n, 3))
        pcd.normals = o3d.utility.Vector3dVector(nrm / np.linalg.norm(nrm, axis=1, keepdims=True))
    return pcd


def _window(qapp, cloud):
    w = MainWindow()
    w.prompt_unsaved_on_close = False
    w.current_pcd = cloud
    w.original_pcd = cloud
    w.viewer.set_cloud(cloud)
    w._refresh_state()
    return w


def _drain(qapp, w, timeout_s=10.0):
    t0 = time.monotonic()
    while w._busy and time.monotonic() - t0 < timeout_s:
        qapp.processEvents()
        time.sleep(0.01)
    qapp.processEvents()
    assert not w._busy, "edit queue did not drain"


def _rows_in(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """For each row of `a`, is it (bit-exactly) a row of `b`?"""
    a = np.ascontiguousarray(a, dtype=np.float64)
    b = np.ascontiguousarray(b, dtype=np.float64)
    view = np.dtype((np.void, a.dtype.itemsize * 3))
    return np.isin(a.view(view).ravel(), b.view(view).ravel())


def _x_below(t):
    return lambda pts: pts[:, 0] < t


def _y_above(t):
    return lambda pts: pts[:, 1] > t


# ---------------------------------------------------------------------------
# pure helpers
# ---------------------------------------------------------------------------


def test_projection_of_a_subset_matches_the_full_cloud_bit_for_bit():
    """The display hide and the worker must classify a shared point the same
    way; a matmul may round differently by array size, element-wise must not."""
    rng = np.random.default_rng(3)
    pts = rng.uniform(-20, 20, (50_001, 3))
    mvp = rng.normal(size=(4, 4))
    mvp[3] = [0.01, -0.02, 0.5, 3.0]
    full = _project_columns(pts, mvp, 1600, 1000)
    sub_idx = np.sort(rng.choice(len(pts), 7_777, replace=False))
    sub = _project_columns(pts[sub_idx], mvp, 1600, 1000)
    for f, s in zip(full, sub):
        assert np.array_equal(f[sub_idx], s)


def test_project_points_keeps_its_shape_contract():
    px, in_front = project_points(np.zeros((5, 3)), np.eye(4), 100, 100)
    assert px.shape == (5, 2) and in_front.shape == (5,) and in_front.dtype == bool


def test_select_by_mask_keeps_points_colours_and_normals_exactly():
    pcd = _cloud(500)
    keep = np.random.default_rng(1).random(500) > 0.3
    out = pcd_ops.select_by_mask(pcd, keep)
    for attr in ("points", "colors", "normals"):
        assert np.array_equal(np.asarray(getattr(out, attr)), np.asarray(getattr(pcd, attr))[keep])
    assert len(pcd.points) == 500, "source must not be mutated"
    with pytest.raises(ValueError):
        pcd_ops.select_by_mask(pcd, keep[:-1])


def test_stats_bounds_with_and_without_non_finite_points():
    pcd = _cloud(300, colors=False, normals=False)
    pts = np.asarray(pcd.points)
    s = pcd_ops.stats(pcd)
    assert np.array_equal(s.min_bound, pts.min(axis=0))
    assert np.array_equal(s.max_bound, pts.max(axis=0))
    bad = np.vstack([pts, [[np.nan, 0, 0], [np.inf, 5e9, -5e9]]])
    s2 = pcd_ops.stats(o3d.geometry.PointCloud(o3d.utility.Vector3dVector(bad)))
    assert np.array_equal(s2.min_bound, pts.min(axis=0))
    assert np.array_equal(s2.max_bound, pts.max(axis=0))
    assert s2.num_points == 302


# ---------------------------------------------------------------------------
# main window queue
# ---------------------------------------------------------------------------


def test_points_vanish_from_the_display_before_the_worker_commits(qapp):
    w = _window(qapp, _cloud())
    before = w.viewer.display_point_count()
    w._submit_mask_edit("Eraser", _x_below(0.5))
    # No event processing yet: the worker cannot have committed.
    assert w._mask_active and w._busy
    assert w.viewer.display_point_count() < before
    assert (np.asarray(w.viewer._display_pcd.points)[:, 0] >= 0.5).all()
    _drain(qapp, w)
    w.close()


def test_rapid_strokes_commit_in_order_as_separate_undo_steps(qapp):
    cloud = _cloud()
    w = _window(qapp, cloud)
    w._submit_mask_edit("Eraser", _x_below(0.5))
    w._submit_mask_edit("Eraser", _y_above(1.5))
    _drain(qapp, w)

    pts0 = np.asarray(cloud.points)
    expected = pts0[(pts0[:, 0] >= 0.5) & (pts0[:, 1] <= 1.5)]
    assert np.array_equal(np.asarray(w.current_pcd.points), expected)
    assert len(w.history) == 2
    assert w.history[0] is cloud, "first undo step must be the untouched original"
    mid = np.asarray(w.history[1].points)
    assert np.array_equal(mid, pts0[pts0[:, 0] >= 0.5])
    assert w._dirty
    w.close()


def test_undo_restores_points_colours_and_normals_exactly(qapp):
    cloud = _cloud()
    ref = {a: np.asarray(getattr(cloud, a)).copy() for a in ("points", "colors", "normals")}
    w = _window(qapp, cloud)
    w._submit_mask_edit("Eraser", _x_below(1.0))
    _drain(qapp, w)
    assert len(w.current_pcd.points) < len(ref["points"])
    w.action_undo()
    _drain(qapp, w)
    for a, arr in ref.items():
        assert np.array_equal(np.asarray(getattr(w.current_pcd, a)), arr)
    w.close()


@pytest.mark.parametrize("lod", [True, False])
@pytest.mark.parametrize("slab", [False, True])
def test_display_stays_a_subset_of_the_cloud(qapp, lod, slab):
    w = _window(qapp, _cloud(3000))
    w.viewer.DISPLAY_MAX = 800  # force a real LOD subset on a small cloud
    w.viewer.set_display_lod_enabled(lod)
    w.viewer.set_cloud(w.current_pcd)
    if slab:
        w._a_zslab.setChecked(True)
        w.viewer.set_zslab_visible(True)
        w.viewer.set_zslab_bounds(0.5, 1.5)
    for fn in (_x_below(0.4), _y_above(1.7), lambda p: (p[:, 2] > 0.9) & (p[:, 0] > 1.2)):
        w._submit_mask_edit("Eraser", fn)
    _drain(qapp, w)
    disp = np.asarray(w.viewer._display_pcd.points)
    cloud = np.asarray(w.current_pcd.points)
    assert len(disp) > 0
    assert _rows_in(disp, cloud).all(), "a ghost point survived on screen"
    if w.viewer._source_rgb is not None:
        assert len(w.viewer._source_rgb) == len(disp)
    assert len(w.viewer._cached_pts) == len(disp)
    w.close()


def test_failed_job_resyncs_the_display_and_drops_the_queue(qapp, monkeypatch):
    cloud = _cloud()
    w = _window(qapp, cloud)
    warned = []
    monkeypatch.setattr(w, "_warn", lambda *a: warned.append(a))

    def boom(*_a, **_k):
        raise RuntimeError("simulated failure")

    monkeypatch.setattr(pcd_ops, "select_by_mask", boom)
    w._submit_mask_edit("Eraser", _x_below(0.5))
    w._submit_mask_edit("Eraser", _y_above(1.5))
    _drain(qapp, w)
    assert warned and "Eraser" in warned[0][0]
    assert w.current_pcd is cloud and w.history == []
    assert w.viewer.display_point_count() == len(cloud.points), "hidden points must come back"
    assert w._mask_queue == []
    w.close()


def test_edit_that_would_empty_the_cloud_is_refused(qapp, monkeypatch):
    cloud = _cloud(400)
    w = _window(qapp, cloud)
    warned = []
    monkeypatch.setattr(w, "_warn", lambda *a: warned.append(a))
    w._submit_mask_edit("Crop (keep)", lambda p: np.ones(len(p), dtype=bool))
    _drain(qapp, w)
    assert warned and "empty" in warned[0][1]
    assert w.current_pcd is cloud
    assert w.viewer.display_point_count() == 400
    w.close()


def test_undo_pressed_mid_queue_runs_after_the_drain(qapp):
    cloud = _cloud()
    w = _window(qapp, cloud)
    w._submit_mask_edit("Eraser", _x_below(0.5))
    assert w._a_undo.isEnabled(), "Undo must stay pressable while edits are queued"
    w.action_undo()  # deferred, not dropped
    _drain(qapp, w)
    _drain(qapp, w)  # the deferred undo is itself async
    assert np.array_equal(np.asarray(w.current_pcd.points), np.asarray(cloud.points))
    assert len(w.redo_stack) == 1
    w.close()


def test_eraser_stays_usable_but_other_tools_are_locked_while_queued(qapp):
    w = _window(qapp, _cloud())
    w._submit_mask_edit("Eraser", _x_below(0.5))
    assert w._a_eraser.isEnabled() and w.eraser_size_spin.isEnabled()
    assert not w._a_voxel.isEnabled() and not w._a_crop.isEnabled()
    _drain(qapp, w)
    assert w._a_voxel.isEnabled()
    w.close()


def test_closing_with_a_pending_queue_is_safe(qapp):
    w = _window(qapp, _cloud(20_000))
    for t in (0.2, 0.4, 0.6):
        w._submit_mask_edit("Eraser", _x_below(t))
    w.close()  # headless: abandons the queue, must not crash
    assert w._mask_queue == []
    for _ in range(50):
        qapp.processEvents()
        time.sleep(0.01)


def test_close_is_refused_while_edits_are_committing_when_interactive(qapp):
    w = _window(qapp, _cloud())
    w.prompt_unsaved_on_close = True
    w._submit_mask_edit("Eraser", _x_below(0.5))
    ev = QtGui.QCloseEvent()
    w.closeEvent(ev)
    assert not ev.isAccepted()
    _drain(qapp, w)
    w.prompt_unsaved_on_close = False
    w.close()


def test_crop_goes_through_the_queue(qapp, monkeypatch):
    cloud = _cloud()
    w = _window(qapp, cloud)
    monkeypatch.setattr(w, "_warn", lambda *a: None)
    w.viewer.resize(800, 600)
    w.viewer.set_cloud(cloud, reset_view=True)
    w.crop_mode_combo.setCurrentText("Remove inside")
    w._on_crop_requested(QtCore.QRect(0, 0, 400, 600))  # left half of the canvas
    assert not w._a_crop.isChecked()
    _drain(qapp, w)
    assert 0 < len(w.current_pcd.points) < len(cloud.points), "crop removed nothing"
    disp = np.asarray(w.viewer._display_pcd.points)
    assert _rows_in(disp, np.asarray(w.current_pcd.points)).all()
    w.close()


def test_failure_cancels_a_pending_undo(qapp, monkeypatch):
    """Stroke 1 commits, stroke 2 fails, Ctrl+Z was pressed meanwhile: the
    undo must not run and silently remove stroke 1 as well."""
    cloud = _cloud()
    w = _window(qapp, cloud)
    monkeypatch.setattr(w, "_warn", lambda *a: None)
    real = pcd_ops.select_by_mask
    calls = []

    def second_fails(pcd, keep):
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError("simulated failure")
        return real(pcd, keep)

    monkeypatch.setattr(pcd_ops, "select_by_mask", second_fails)
    w._submit_mask_edit("Eraser", _x_below(0.5))
    w._submit_mask_edit("Eraser", _y_above(1.5))
    w.action_undo()
    _drain(qapp, w)
    _drain(qapp, w)
    pts0 = np.asarray(cloud.points)
    assert np.array_equal(np.asarray(w.current_pcd.points), pts0[pts0[:, 0] >= 0.5]), "stroke 1 must survive"
    assert len(w.history) == 1 and w._deferred_action is None
    w.close()


def test_heavy_erasing_refills_the_display_from_the_cloud(qapp):
    w = _window(qapp, _cloud(6000))
    w.viewer.DISPLAY_MAX = 1000
    w.viewer.set_cloud(w.current_pcd)
    assert w.viewer.display_point_count() == 1000
    refills = []
    orig = w._on_refill_finished
    w._on_refill_finished = lambda: (refills.append(1), orig())[1]
    w._submit_mask_edit("Eraser", _x_below(1.4))  # ~70 % of the cloud
    _drain(qapp, w)
    assert refills, "display fell below half its budget but was not re-sampled"
    assert w.viewer.display_point_count() == 1000  # 1800 points left > budget
    assert _rows_in(np.asarray(w.viewer._display_pcd.points), np.asarray(w.current_pcd.points)).all()
    w.close()


def test_stroke_made_during_a_refill_is_still_hidden(qapp):
    w = _window(qapp, _cloud(6000))
    w.viewer.DISPLAY_MAX = 1000
    w.viewer.set_cloud(w.current_pcd)
    orig_refill = w._maybe_refill_display

    fired = []

    def refill_then_stroke():
        orig_refill()
        if w._mask_worker_ref is not None and not fired:  # a refill is now running
            fired.append(1)
            w._submit_mask_edit("Eraser", _y_above(1.5))

    w._maybe_refill_display = refill_then_stroke
    w._submit_mask_edit("Eraser", _x_below(1.4))
    _drain(qapp, w)
    assert fired, "test did not reach the mid-refill window"
    cloud = np.asarray(w.current_pcd.points)
    assert (cloud[:, 1] <= 1.5).all(), "the mid-refill stroke was not committed"
    disp = np.asarray(w.viewer._display_pcd.points)
    assert _rows_in(disp, cloud).all(), "the mid-refill stroke left ghosts on screen"
    w.close()
