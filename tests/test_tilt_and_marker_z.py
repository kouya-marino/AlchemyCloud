"""Marker Z control and the manual tilt trim.

Both exist because `level_ground` can only align the *dominant* plane. When a
floor and a ceiling disagree — measured 2.74 deg apart on a real 180 m survey —
no single rigid rotation levels both, and the remaining correction is a
judgement call the operator makes by eye.
"""

from __future__ import annotations

import numpy as np
import open3d as o3d
import pytest
from PyQt5 import QtWidgets

from alchemycloud import pcd_ops
from alchemycloud.main_window import MainWindow


def _cloud(arr):
    return o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.asarray(arr, dtype=np.float64)))


def _slab(z=1.2, n=4000, span=4.0, seed=0, tilt_deg=0.0):
    """A flat slab at height `z`, optionally tilted about +Y."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, span, n)
    y = rng.uniform(0, span, n)
    zz = np.full(n, z) + x * np.tan(np.radians(tilt_deg)) + rng.normal(0, 0.005, n)
    return np.c_[x, y, zz]


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
# pcd_ops.apply_tilt
# ---------------------------------------------------------------------------


def test_apply_tilt_removes_a_known_tilt():
    """A slab tilted 3 deg about +Y is levelled by a -3 deg pitch trim."""
    pcd = _cloud(_slab(tilt_deg=3.0))
    _n, before, _ = pcd_ops.surface_tilt(pcd, percentile=50.0)
    assert before == pytest.approx(3.0, abs=0.2)
    out = pcd_ops.apply_tilt(pcd, roll_deg=0.0, pitch_deg=3.0)
    _n2, after, _ = pcd_ops.surface_tilt(out, percentile=50.0)
    assert after < 0.2, f"trim left {after:.3f} deg"


def test_apply_tilt_about_centroid_keeps_the_cloud_in_place():
    """Rotating about the world origin swings a distant cloud through space.

    Note a rigid rotation cannot change the point set's extent whichever centre
    is used — only its position. That position is the whole point: a survey
    sitting 150 m from the origin should not fly off-screen when nudged.
    """
    pts = _slab(z=0.0) + np.array([150.0, 0.0, 0.0])
    pcd = _cloud(pts)
    c0 = np.asarray(pcd.points).mean(axis=0)

    centred = pcd_ops.apply_tilt(pcd, 0.0, 5.0, about="centroid")
    origin = pcd_ops.apply_tilt(pcd, 0.0, 5.0, about="origin")

    assert np.allclose(np.asarray(centred.points).mean(axis=0), c0, atol=1e-9)
    assert not np.allclose(np.asarray(origin.points).mean(axis=0), c0, atol=0.5)
    # Same shape either way.
    assert np.ptp(np.asarray(centred.points)[:, 2]) == pytest.approx(np.ptp(np.asarray(origin.points)[:, 2]), abs=1e-9)


def test_apply_tilt_zero_is_a_no_op():
    pcd = _cloud(_slab())
    out = pcd_ops.apply_tilt(pcd, 0.0, 0.0)
    assert np.allclose(np.asarray(out.points), np.asarray(pcd.points))


def test_apply_tilt_rejects_a_bad_centre():
    with pytest.raises(ValueError, match="centroid"):
        pcd_ops.apply_tilt(_cloud(_slab()), 1.0, 0.0, about="middle")


# ---------------------------------------------------------------------------
# pcd_ops.surface_tilt
# ---------------------------------------------------------------------------


def test_surface_tilt_separates_a_tilted_floor_from_a_level_ceiling():
    """The real failure mode: levelling the floor tilts the ceiling.

    Reproduces the shape of a real survey whose floor and ceiling were 2.74 deg
    apart, so flattening one necessarily tilts the other. Exaggerated here so
    the assertion has margin.
    """
    floor = _slab(z=0.0, tilt_deg=4.5, seed=1)
    ceiling = _slab(z=6.0, tilt_deg=0.0, seed=2)
    pcd = _cloud(np.vstack([floor, ceiling]))

    nf, tf, _ = pcd_ops.surface_tilt(pcd, percentile=2.0)
    nc, tc, _ = pcd_ops.surface_tilt(pcd, percentile=98.0)
    assert tf == pytest.approx(4.5, abs=0.5)
    assert tc < 0.5
    between = np.degrees(np.arccos(abs(float(nf @ nc))))
    assert between == pytest.approx(4.5, abs=0.5), "the disagreement must be visible"


def test_surface_tilt_needs_points():
    with pytest.raises(ValueError, match="at least 3"):
        pcd_ops.surface_tilt(_cloud(np.zeros((2, 3))), percentile=50.0)


# ---------------------------------------------------------------------------
# Marker Z
# ---------------------------------------------------------------------------


def test_marker_z_spin_drives_the_viewer(qapp):
    """The transform always subtracted Z; there was simply no way to set it."""
    w = _window(qapp, _cloud(_slab(z=1.2)))
    assert w.viewer.marker_position()[2] == pytest.approx(0.0)
    w.marker_z_spin.setValue(1.2)
    assert w.viewer.marker_position()[2] == pytest.approx(1.2)


def test_marker_z_round_trips_back_into_the_spin(qapp):
    w = _window(qapp, _cloud(_slab(z=1.2)))
    w.viewer.set_marker_position(0.5, 0.5, 0.9)
    qapp.processEvents()
    assert w.marker_z_spin.value() == pytest.approx(0.9, abs=1e-6)


def test_marker_z_range_never_clamps_the_marker_out_of_reach(qapp):
    """Same guard as X/Y: a marker above the cloud must stay representable."""
    w = _window(qapp, _cloud(_slab(z=1.2)))
    w.viewer.set_marker_position(0.0, 0.0, 9.0)
    w._refresh_state()
    assert w.marker_z_spin.minimum() <= 9.0 <= w.marker_z_spin.maximum()
    assert w.viewer.marker_position()[2] == pytest.approx(9.0)


def test_applying_a_marker_with_z_lands_the_height_on_zero():
    """End to end: the height typed into Z becomes the new z = 0."""
    pts = _slab(z=1.2)
    out = pcd_ops.transform_to_marker_frame(_cloud(pts), np.array([2.0, 1.5, 1.2]), 0.0)
    assert np.asarray(out.points)[:, 2].mean() == pytest.approx(0.0, abs=0.02)


# ---------------------------------------------------------------------------
# Window wiring
# ---------------------------------------------------------------------------


def test_tilt_controls_are_gated_on_a_loaded_cloud(qapp):
    w = MainWindow()
    w.prompt_unsaved_on_close = False
    assert not w.tilt_roll_spin.isEnabled()
    assert not w._a_tilt_trim.isEnabled()
    assert not w._a_snap_z.isEnabled()

    cloud = _cloud(_slab())
    w.current_pcd = cloud
    w.original_pcd = cloud
    w.viewer.set_cloud(cloud)
    w._refresh_state()
    assert w.tilt_roll_spin.isEnabled()
    assert w._a_tilt_trim.isEnabled()
    assert w._a_snap_z.isEnabled()


def test_apply_trim_resets_the_boxes_so_clicks_do_not_compound(qapp, busy_wait):
    w = _window(qapp, _cloud(_slab(tilt_deg=3.0)))
    w.tilt_pitch_spin.setValue(3.0)
    w.action_tilt_trim()
    assert busy_wait(lambda: not w._busy, timeout_s=10.0)
    assert w.tilt_pitch_spin.value() == pytest.approx(0.0)
    assert w.tilt_roll_spin.value() == pytest.approx(0.0)
    _n, after, _ = pcd_ops.surface_tilt(w.current_pcd, percentile=50.0)
    assert after < 0.3


def test_apply_trim_with_zero_does_nothing(qapp):
    w = _window(qapp, _cloud(_slab()))
    before = len(w.history)
    w.action_tilt_trim()
    assert len(w.history) == before
    assert "nothing to apply" in w.statusBar().currentMessage()


def test_snap_z_sets_the_marker_to_the_floor(qapp, busy_wait):
    w = _window(qapp, _cloud(_slab(z=1.2)))
    assert w.viewer.marker_position()[2] == pytest.approx(0.0)
    w.action_snap_marker_z()
    assert busy_wait(lambda: not w._busy, timeout_s=15.0)
    qapp.processEvents()
    assert w.viewer.marker_position()[2] == pytest.approx(1.2, abs=0.05)
    assert w.marker_z_spin.value() == pytest.approx(1.2, abs=0.05)


def test_measure_tilt_reports_the_disagreement(qapp, busy_wait, monkeypatch):
    floor = _slab(z=0.0, tilt_deg=4.5, seed=1)
    ceiling = _slab(z=6.0, tilt_deg=0.0, seed=2)
    w = _window(qapp, _cloud(np.vstack([floor, ceiling])))
    shown = []
    monkeypatch.setattr(QtWidgets.QMessageBox, "warning", lambda *a, **k: shown.append(a[2]))
    w.action_measure_tilt()
    assert busy_wait(lambda: not w._busy, timeout_s=15.0)
    qapp.processEvents()
    assert shown, "no report shown"
    assert "no single rotation levels both" in shown[0]
