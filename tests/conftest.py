"""Shared pytest fixtures + offscreen Qt setup.

`QT_QPA_PLATFORM=offscreen` is set *before* any PyQt import so the
test process doesn't need an X server.  The pyqtgraph `Requires >=
OpenGL 2.1` warning printed under offscreen is harmless — pyqtgraph
falls back to a stub context for the geometry-only paths we test.
"""
from __future__ import annotations

import os
import sys
import time

# Must be set before anyone imports PyQt5 or pyqtgraph.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

# Make the project root importable so `from alchemycloud import ...` works
# even without an editable install (CI installs the package; this covers local runs).
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402
import open3d as o3d  # noqa: E402
import pytest  # noqa: E402


# ---------------------------------------------------------------------------
# Qt / GUI fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def qapp():
    """Session-scoped QApplication. Created lazily so non-GUI tests don't
    pay any cost."""
    from PyQt5 import QtWidgets

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(["pytest"])
    yield app


@pytest.fixture
def busy_wait(qapp):
    """Return a `wait_for(pred, timeout_s=5.0) -> bool` helper that drains
    Qt events while polling the predicate."""

    def _wait(predicate, timeout_s: float = 5.0) -> bool:
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout_s:
            if predicate():
                return True
            qapp.processEvents()
            time.sleep(0.02)
        return predicate()

    return _wait


# ---------------------------------------------------------------------------
# Synthetic point-cloud fixtures (no Qt — fast)
# ---------------------------------------------------------------------------

@pytest.fixture
def cube_cloud() -> o3d.geometry.PointCloud:
    """5 000 points uniformly distributed in [-1, 1]^3."""
    pts = np.random.default_rng(0).uniform(-1, 1, (5_000, 3))
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    return pcd


@pytest.fixture
def cluster_with_outliers() -> o3d.geometry.PointCloud:
    """Dense Gaussian cluster around origin + 50 isolated outliers far
    away. Designed so SOR / radius outlier removal can clear the outliers
    without touching the cluster."""
    rng = np.random.default_rng(0)
    cluster = rng.normal(0, 0.05, (5_000, 3))
    outliers = rng.uniform(5, 10, (50, 3))
    pts = np.vstack([cluster, outliers])
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    return pcd


@pytest.fixture
def floor_with_chair() -> o3d.geometry.PointCloud:
    """2 000-point floor at z=0 + 200-point 'chair top' at z=0.5.
    Used for occupancy tests: tristate with threshold=0.2 should mark
    the chair footprint occupied and the floor free."""
    rng = np.random.default_rng(1)
    floor = np.column_stack([
        rng.uniform(-2, 2, 2000),
        rng.uniform(-2, 2, 2000),
        np.zeros(2000),
    ])
    chair = np.column_stack([
        rng.uniform(-0.3, 0.3, 200),
        rng.uniform(-0.3, 0.3, 200),
        np.full(200, 0.5),
    ])
    pts = np.vstack([floor, chair])
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    return pcd


@pytest.fixture
def tilted_cloud() -> o3d.geometry.PointCloud:
    """12 000-point cloud (10 000 floor + 2 000 above) rotated +10° about
    +X and translated up by 1.5 m. Designed for `level_ground` tests."""
    rng = np.random.default_rng(2)
    floor = np.column_stack([
        rng.uniform(-2, 2, 10_000),
        rng.uniform(-2, 2, 10_000),
        rng.normal(0, 0.001, 10_000),
    ])
    extra = np.column_stack([
        rng.uniform(-2, 2, 2000),
        rng.uniform(-2, 2, 2000),
        rng.uniform(0.5, 2.0, 2000),
    ])
    pts = np.vstack([floor, extra])
    theta = np.radians(10.0)
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(theta), -np.sin(theta)],
        [0, np.sin(theta), np.cos(theta)],
    ])
    pts = (Rx @ pts.T).T + np.array([0.0, 0.0, 1.5])
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    return pcd
