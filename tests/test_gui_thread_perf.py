"""Work that used to freeze the GUI thread is gone, and nothing it relied on
broke.

* `colorize_by_axis` hands Open3D a contiguous array (a strided view took
  ~0.5 s per 500k points) — the colours must be unchanged.
* `apply_prepared_cloud` reuses the colours `prepare_cloud` computed on the
  worker instead of recolouring on the GUI thread — but only while they are
  still valid for the current colour mode.
* Undo/redo history and `original_pcd` hold references, not deep copies. That
  is only sound while no operation modifies a cloud in place, which the
  tests at the bottom pin.
"""

from __future__ import annotations

import time

import numpy as np
import open3d as o3d
import pytest
from matplotlib import colormaps

from alchemycloud import occupancy, pcd_ops
from alchemycloud.embedded_viewer import EmbeddedViewer
from alchemycloud.main_window import MainWindow


def _cloud(n=3000, seed=0, colors=True, normals=True):
    rng = np.random.default_rng(seed)
    pts = rng.uniform(0, 2, (n, 3))
    pts[: n // 3, 2] = rng.normal(0.0, 0.005, n // 3)  # a floor for the RANSAC ops
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
    if colors:
        pcd.colors = o3d.utility.Vector3dVector(rng.uniform(0, 1, (n, 3)))
    if normals:
        nrm = rng.normal(size=(n, 3))
        pcd.normals = o3d.utility.Vector3dVector(nrm / np.linalg.norm(nrm, axis=1, keepdims=True))
    return pcd


def _snapshot(pcd):
    return tuple(np.asarray(a).copy() for a in (pcd.points, pcd.colors, pcd.normals))


def _assert_same(pcd, snap):
    for got, want in zip(_snapshot(pcd), snap, strict=True):
        np.testing.assert_array_equal(got, want)


def _wait(qapp, w, timeout_s=20.0):
    t0 = time.monotonic()
    while w._busy and time.monotonic() - t0 < timeout_s:
        qapp.processEvents()
        time.sleep(0.01)
    qapp.processEvents()
    assert not w._busy, "operation did not finish"


# ---------------------------------------------------------------------------
# colour conversion
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("axis", ["x", "y", "z"])
def test_colorize_by_axis_colours_are_unchanged(axis):
    pcd = _cloud(colors=False)
    vals = np.asarray(pcd.points)[:, {"x": 0, "y": 1, "z": 2}[axis]]
    normed = (vals - vals.min()) / (vals.max() - vals.min())
    want = colormaps["viridis"](normed)[:, :3]
    pcd_ops.colorize_by_axis(pcd, axis)
    np.testing.assert_array_equal(np.asarray(pcd.colors), want)


# ---------------------------------------------------------------------------
# apply_prepared_cloud reuses the worker's colours
# ---------------------------------------------------------------------------


def _scatter_rgba(viewer):
    return np.asarray(viewer._scatter.color)


def test_apply_does_not_recolour_on_the_gui_thread(qapp, monkeypatch):
    v = EmbeddedViewer()
    prepared = v.prepare_cloud(_cloud())
    calls = []
    real = pcd_ops.colorize_by_axis
    monkeypatch.setattr(pcd_ops, "colorize_by_axis", lambda *a, **k: calls.append(1) or real(*a, **k))
    v.apply_prepared_cloud(*prepared)
    assert calls == []
    np.testing.assert_array_equal(_scatter_rgba(v), prepared[2])


def test_colour_mode_changed_between_prepare_and_apply_is_honoured(qapp):
    """The display refill prepares on a worker while the colour controls stay
    live; colours computed for the old mode must not be shown."""
    cloud = _cloud()
    v = EmbeddedViewer()
    v.set_color_axis("z")
    prepared = v.prepare_cloud(cloud)
    v.set_color_axis("x")
    v.apply_prepared_cloud(*prepared)

    ref = EmbeddedViewer()
    ref.set_color_axis("x")
    ref.set_cloud(cloud)
    np.testing.assert_allclose(_scatter_rgba(v), _scatter_rgba(ref), atol=1e-6)
    assert v._legend._axis == "X"
    pts = np.asarray(prepared[0].points)
    assert v._legend._vmin == pytest.approx(pts[:, 0].min(), abs=1e-5)
    assert v._legend._vmax == pytest.approx(pts[:, 0].max(), abs=1e-5)


def test_tone_settings_changed_between_prepare_and_apply_are_honoured(qapp):
    cloud = _cloud()
    v = EmbeddedViewer()
    v.set_color_axis("rgb")
    prepared = v.prepare_cloud(cloud)
    v.set_rgb_gamma(2.0)
    v.apply_prepared_cloud(*prepared)
    want = v._tone_map(np.asarray(prepared[0].colors, dtype=np.float32))
    np.testing.assert_allclose(_scatter_rgba(v)[:, :3], want, atol=1e-6)


def test_overlapping_prepares_keep_each_clouds_own_rgb(qapp):
    """Two prepares before either apply (a refill racing a resync) must not
    hand one cloud the other's source colours."""
    a, b = _cloud(seed=1), _cloud(seed=2)
    v = EmbeddedViewer()
    v.set_color_axis("z")
    prep_a = v.prepare_cloud(a)
    v.prepare_cloud(b)
    v.apply_prepared_cloud(*prep_a)
    v.set_color_axis("rgb")
    want = v._tone_map(np.asarray(a.colors, dtype=np.float32))
    np.testing.assert_allclose(_scatter_rgba(v)[:, :3], want, atol=1e-6)


def test_slab_view_uploads_only_in_slab_points(qapp):
    v = EmbeddedViewer()
    v.set_zslab_visible(True)
    v.set_zslab_bounds(0.5, 1.5)
    v.set_cloud(_cloud())
    pos = np.asarray(v._scatter.pos)
    assert len(pos) > 0
    assert pos[:, 2].min() >= 0.5 and pos[:, 2].max() <= 1.5


# ---------------------------------------------------------------------------
# history by reference: nothing may modify a cloud in place
# ---------------------------------------------------------------------------

OPS = {
    "voxel": lambda p: pcd_ops.voxel_downsample(p, 0.1),
    "sor": lambda p: pcd_ops.statistical_outlier_removal(p, 20, 2.0),
    "radius": lambda p: pcd_ops.radius_outlier_removal(p, 3, 0.2),
    "z_clip": lambda p: pcd_ops.z_clip(p, 0.2, 1.5),
    "crop_inside": lambda p: pcd_ops.crop_with_box(p, (0, 0, 0), (1, 1, 1), True),
    "crop_outside": lambda p: pcd_ops.crop_with_box(p, (0, 0, 0), (1, 1, 1), False),
    "ground": lambda p: pcd_ops.remove_ground_ransac(p)[0],
    "level": lambda p: pcd_ops.level_ground(p)[0],
    "tilt": lambda p: pcd_ops.apply_tilt(p, 2.0, -1.0),
    "marker": lambda p: pcd_ops.transform_to_marker_frame(p, (0.5, 0.5, 0.0), 0.3),
    "select_mask": lambda p: pcd_ops.select_by_mask(p, np.asarray(p.points)[:, 0] < 1.0),
    # Read-only consumers matter too: `original_pcd` is now the loaded cloud
    # itself, so a write here would corrupt Reset and every undo entry.
    "save_binary": lambda p: pcd_ops.save_pcd(p, "b.pcd"),
    "save_ascii": lambda p: pcd_ops.save_pcd(p, "a.pcd", ascii_format=True),
    "occupancy_projection": lambda p: occupancy.build(p, occupancy.GridParams()),
    "occupancy_voxel3d": lambda p: occupancy.build(p, occupancy.GridParams(method="voxel3d")),
    "surface_tilt": lambda p: pcd_ops.surface_tilt(p, 2.0),
    "fit_ground_plane": lambda p: pcd_ops.fit_ground_plane(p),
    "stats": pcd_ops.stats,
    "display_downsample": lambda p: pcd_ops.display_downsample(p, 500),
}


@pytest.mark.parametrize("name", sorted(OPS))
def test_operations_never_modify_their_input(name, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # the save ops write relative paths
    pcd = _cloud()
    snap = _snapshot(pcd)
    OPS[name](pcd)
    _assert_same(pcd, snap)


@pytest.mark.parametrize("lod", [True, False])
@pytest.mark.parametrize("axis", ["z", "rgb"])
def test_undo_redo_reset_restore_exact_clouds_without_copying(qapp, lod, axis):
    """LOD off makes the display start as the cloud itself — the case where
    recolouring could leak into a history entry if the viewer did not copy."""
    cloud = _cloud()
    snap = _snapshot(cloud)
    w = MainWindow()
    w.prompt_unsaved_on_close = False
    w.viewer.set_display_lod_enabled(lod)
    w.viewer.set_color_axis(axis)
    w.current_pcd = cloud
    w.original_pcd = cloud
    w.viewer.set_cloud(cloud)
    w._refresh_state()

    w._apply_async("Voxel", lambda: pcd_ops.voxel_downsample(cloud, 0.1))
    _wait(qapp, w)
    voxeled = w.current_pcd
    assert w.history[-1] is cloud  # a reference, not a copy

    w.viewer.set_color_axis("x" if axis != "x" else "y")
    w.action_undo()
    _wait(qapp, w)
    assert w.current_pcd is cloud
    assert w.redo_stack[-1] is voxeled
    _assert_same(cloud, snap)

    w.action_redo()
    _wait(qapp, w)
    assert w.current_pcd is voxeled

    w.action_reset()
    _wait(qapp, w)
    assert w.current_pcd is cloud
    _assert_same(cloud, snap)
    _assert_same(w.original_pcd, snap)
    w.close()
