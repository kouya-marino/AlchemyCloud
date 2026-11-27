"""Faster background operations give exactly the results they used to.

* Radius outlier removal can run on the GPU (opt-in, ``ALCHEMYCLOUD_GPU=1``)
  and must keep the points the CPU filter keeps; any GPU failure falls back
  to it. The tensor-filter adaptation is also checked on the CPU device, so
  CI (no GPU) still guards it.
* Z-clip uses a NumPy gather instead of ``select_by_index``.
* Occupancy voxel3d dedupes voxels via packed int64 keys instead of row-wise
  ``np.unique``.
* The busy overlay only appears for jobs slower than a short delay.
"""

from __future__ import annotations

import threading
import time

import numpy as np
import open3d as o3d
import pytest

from alchemycloud import occupancy, pcd_ops
from alchemycloud.loading_overlay import LoadingOverlay
from alchemycloud.main_window import MainWindow

HAS_CUDA = pcd_ops.cuda_available()


def _cloud(n=20_000, seed=0):
    rng = np.random.default_rng(seed)
    pts = np.concatenate([rng.normal(0, 0.3, (n - n // 10, 3)), rng.uniform(-3, 3, (n // 10, 3))])
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
    pcd.colors = o3d.utility.Vector3dVector(rng.uniform(0, 1, (n, 3)))
    nrm = rng.normal(size=(n, 3))
    pcd.normals = o3d.utility.Vector3dVector(nrm / np.linalg.norm(nrm, axis=1, keepdims=True))
    return pcd


def _assert_same_cloud(a, b):
    for attr in ("points", "colors", "normals"):
        np.testing.assert_array_equal(np.asarray(getattr(a, attr)), np.asarray(getattr(b, attr)))


# ---------------------------------------------------------------------------
# radius outlier removal: GPU path, CPU fallback
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not HAS_CUDA, reason="needs a CUDA GPU")
@pytest.mark.parametrize(("nb", "radius"), [(30, 0.10), (10, 0.05), (5, 0.2), (1, 0.02)])
def test_gpu_radius_filter_keeps_exactly_what_the_cpu_keeps(nb, radius, monkeypatch):
    monkeypatch.setenv(pcd_ops.GPU_ENV, "1")
    assert pcd_ops.cuda_device() is not None
    pcd = _cloud()
    cpu, _ = pcd.remove_radius_outlier(nb_points=nb, radius=radius)
    _assert_same_cloud(pcd_ops.radius_outlier_removal(pcd, nb, radius), cpu)


def test_radius_filter_falls_back_to_the_cpu_when_the_gpu_fails(monkeypatch):
    pcd = _cloud()
    cpu, _ = pcd.remove_radius_outlier(nb_points=10, radius=0.05)

    def boom(*_a, **_k):
        raise RuntimeError("CUDA runtime error: out of memory")

    monkeypatch.setattr(pcd_ops, "cuda_device", lambda: object())  # any non-None device
    monkeypatch.setattr(pcd_ops, "_radius_keep_mask_tensor", boom)
    _assert_same_cloud(pcd_ops.radius_outlier_removal(pcd, 10, 0.05), cpu)


@pytest.mark.parametrize("value", [None, "", "0"])
def test_gpu_is_off_unless_opted_in(monkeypatch, value):
    if value is None:
        monkeypatch.delenv(pcd_ops.GPU_ENV, raising=False)
    else:
        monkeypatch.setenv(pcd_ops.GPU_ENV, value)
    assert pcd_ops.cuda_device() is None


@pytest.mark.parametrize(("nb", "radius"), [(30, 0.10), (5, 0.05), (1, 0.02)])
def test_tensor_filter_adaptation_matches_legacy_on_the_cpu_device(nb, radius):
    """Guards the nb_points + 1 adaptation where CI can run it (no GPU there).

    Exact-tie lattices are deliberately not used: the legacy filter itself
    keeps a varying number of points from run to run when neighbours sit at
    exactly the radius."""
    pts = np.asarray(_cloud().points)
    legacy = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
    want = np.zeros(len(pts), bool)
    want[legacy.remove_radius_outlier(nb_points=nb, radius=radius)[1]] = True
    got = pcd_ops._radius_keep_mask_tensor(pts, nb, radius, o3d.core.Device("CPU:0"))
    np.testing.assert_array_equal(got, want)


# ---------------------------------------------------------------------------
# z-clip
# ---------------------------------------------------------------------------


def test_z_clip_matches_select_by_index_exactly():
    pcd = _cloud()
    pts = np.asarray(pcd.points)
    want = pcd.select_by_index(np.where((pts[:, 2] >= -0.2) & (pts[:, 2] <= 0.5))[0])
    _assert_same_cloud(pcd_ops.z_clip(pcd, -0.2, 0.5), want)


# ---------------------------------------------------------------------------
# occupancy voxel3d
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_unique_voxels_matches_row_wise_unique(seed):
    rng = np.random.default_rng(seed)
    # Different span per axis, so a mixed-up stride in the packing shows up.
    lo, hi = np.array([-40, -7, 3]), np.array([25, 90, 11])
    v = rng.integers(lo, hi, (50_000, 3), dtype=np.int64)
    np.testing.assert_array_equal(occupancy._unique_voxels(v), np.unique(v, axis=0))


def test_unique_voxels_falls_back_when_the_packed_key_would_overflow():
    v = np.array([[-(2**40), 0, 0], [2**40, 5, 2**30], [0, 0, 0], [0, 0, 0]], dtype=np.int64)
    np.testing.assert_array_equal(occupancy._unique_voxels(v), np.unique(v, axis=0))


@pytest.mark.parametrize("height_threshold", [None, 0.3])
def test_voxel3d_grid_is_unchanged(height_threshold, monkeypatch):
    pcd = _cloud()
    params = occupancy.GridParams(
        resolution=0.05, z_min=-1.0, z_max=1.0, method="voxel3d", height_threshold=height_threshold
    )
    fast = occupancy.build(pcd, params)
    monkeypatch.setattr(occupancy, "_unique_voxels", lambda v: np.unique(v, axis=0))
    slow = occupancy.build(pcd, params)
    np.testing.assert_array_equal(fast.image, slow.image)
    assert fast.origin == slow.origin


# ---------------------------------------------------------------------------
# delayed busy overlay
# ---------------------------------------------------------------------------


def _pump(qapp, seconds):
    t0 = time.monotonic()
    while time.monotonic() - t0 < seconds:
        qapp.processEvents()
        time.sleep(0.005)


def test_overlay_never_shows_for_a_job_faster_than_the_delay(qapp):
    from PyQt5 import QtWidgets

    parent = QtWidgets.QWidget()
    parent.show()
    ov = LoadingOverlay(parent)
    ov.start("x", delay_ms=200)
    _pump(qapp, 0.05)
    assert not ov.isVisible()
    ov.stop()
    _pump(qapp, 0.3)
    assert not ov.isVisible()


def test_overlay_shows_once_a_job_outlasts_the_delay(qapp):
    from PyQt5 import QtWidgets

    parent = QtWidgets.QWidget()
    parent.show()
    ov = LoadingOverlay(parent)
    ov.start("x", delay_ms=50)
    _pump(qapp, 0.3)
    assert ov.isVisible()
    ov.stop()
    assert not ov.isVisible()


def test_tools_lock_immediately_even_before_the_overlay_shows(qapp):
    w = MainWindow()
    w.prompt_unsaved_on_close = False
    pcd = _cloud(2000)
    w.current_pcd = pcd
    w.viewer.set_cloud(pcd)
    w._refresh_state()
    w.show()  # a hidden window's overlay can never be visible

    release = threading.Event()

    def slow():
        release.wait(10)
        return pcd_ops.voxel_downsample(pcd, 0.05)

    w._apply_async("Voxel", slow)
    assert w._busy
    assert not w._a_voxel.isEnabled()
    assert not w.loading.isVisible()
    t0 = time.monotonic()
    while not w.loading.isVisible() and time.monotonic() - t0 < 5:
        _pump(qapp, 0.01)
    shown_after = time.monotonic() - t0
    release.set()
    assert w.loading.isVisible()
    assert shown_after >= 0.2  # delayed, not immediate
    t0 = time.monotonic()
    while w._busy and time.monotonic() - t0 < 10:
        _pump(qapp, 0.02)
    assert not w._busy and not w.loading.isVisible()
    w.close()
