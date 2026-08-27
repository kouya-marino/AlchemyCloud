"""Unit tests for pure-math operations in `pcd_ops`."""

from __future__ import annotations

import os
import tempfile

import numpy as np
import open3d as o3d
import pytest

from alchemycloud import pcd_ops

# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ext", [".pcd", ".ply"])
def test_load_save_roundtrip(cube_cloud, ext):
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as f:
        path = f.name
    try:
        pcd_ops.save_pcd(cube_cloud, path)
        loaded = pcd_ops.load_pcd(path)
        assert len(loaded.points) == len(cube_cloud.points)
        # Coordinates round-trip to within float precision.
        diff = np.asarray(loaded.points) - np.asarray(cube_cloud.points)
        assert np.max(np.abs(diff)) < 1e-5
    finally:
        os.unlink(path)


def test_load_pcd_rejects_empty(tmp_path):
    empty = tmp_path / "empty.ply"
    empty.write_text(
        "ply\nformat ascii 1.0\nelement vertex 0\nproperty float x\nproperty float y\nproperty float z\nend_header\n"
    )
    with pytest.raises(ValueError, match="No points"):
        pcd_ops.load_pcd(str(empty))


# ---------------------------------------------------------------------------
# Voxel downsample
# ---------------------------------------------------------------------------


def test_voxel_downsample_reduces_count(cube_cloud):
    out = pcd_ops.voxel_downsample(cube_cloud, 0.2)
    assert 0 < len(out.points) < len(cube_cloud.points)


def test_voxel_downsample_rejects_nonpositive(cube_cloud):
    with pytest.raises(ValueError):
        pcd_ops.voxel_downsample(cube_cloud, 0.0)
    with pytest.raises(ValueError):
        pcd_ops.voxel_downsample(cube_cloud, -0.1)


# ---------------------------------------------------------------------------
# Outlier removal — SOR + radius
# ---------------------------------------------------------------------------


def test_statistical_outlier_removal_drops_outliers(cluster_with_outliers):
    n_in = len(cluster_with_outliers.points)
    out = pcd_ops.statistical_outlier_removal(cluster_with_outliers)
    # Outliers should be gone; most of the dense cluster survives.
    assert len(out.points) < n_in
    assert len(out.points) >= n_in - 60  # at most ~outliers + a tiny slack


def test_radius_outlier_removal_drops_outliers(cluster_with_outliers):
    n_in = len(cluster_with_outliers.points)
    out = pcd_ops.radius_outlier_removal(cluster_with_outliers, nb_points=20, radius=0.1)
    assert len(out.points) < n_in
    assert len(out.points) >= n_in - 60


def test_radius_outlier_rejects_bad_args(cluster_with_outliers):
    with pytest.raises(ValueError):
        pcd_ops.radius_outlier_removal(cluster_with_outliers, nb_points=0, radius=0.1)
    with pytest.raises(ValueError):
        pcd_ops.radius_outlier_removal(cluster_with_outliers, nb_points=10, radius=0.0)


# ---------------------------------------------------------------------------
# Z-clip
# ---------------------------------------------------------------------------


def test_z_clip_inclusive_bounds():
    # Include points AT the boundary so inclusive vs strict comparison can
    # be detected by the test (uniform random samples never land on round
    # numbers exactly).
    pts = np.array(
        [
            [0.0, 0.0, -0.5],  # exactly at z_min — must be KEPT (inclusive)
            [0.0, 0.0, 0.5],  # exactly at z_max — must be KEPT (inclusive)
            [0.0, 0.0, 0.0],  # interior — kept
            [0.0, 0.0, -0.51],  # just below — dropped
            [0.0, 0.0, 0.51],  # just above — dropped
        ],
        dtype=np.float64,
    )
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    out = pcd_ops.z_clip(pcd, -0.5, 0.5)
    z = np.asarray(out.points)[:, 2]
    assert len(out.points) == 3
    assert np.all(z >= -0.5) and np.all(z <= 0.5)


def test_z_clip_rejects_inverted_range(cube_cloud):
    with pytest.raises(ValueError):
        pcd_ops.z_clip(cube_cloud, 1.0, 0.5)


# ---------------------------------------------------------------------------
# RANSAC ground
# ---------------------------------------------------------------------------


def test_remove_ground_ransac_finds_floor():
    rng = np.random.default_rng(3)
    floor = np.column_stack(
        [
            rng.uniform(-2, 2, 5000),
            rng.uniform(-2, 2, 5000),
            rng.normal(0, 0.005, 5000),
        ]
    )
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(floor)
    _, ground, plane = pcd_ops.remove_ground_ransac(pcd, distance_threshold=0.05)
    assert len(ground.points) > 4000
    a, b, c, d = plane
    # Normal should point near ±Z; constant d near 0.
    assert abs(abs(c) - 1.0) < 0.05
    assert abs(a) < 0.05 and abs(b) < 0.05
    assert abs(d) < 0.05


# ---------------------------------------------------------------------------
# transform_to_marker_frame
# ---------------------------------------------------------------------------


def test_transform_to_marker_frame_translation():
    pts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    out = pcd_ops.transform_to_marker_frame(pcd, [1.0, 2.0, 0.0], 0.0)
    expected = pts - [1.0, 2.0, 0.0]
    assert np.allclose(np.asarray(out.points), expected)


def test_transform_to_marker_frame_rotation():
    pts = np.array([[1, 0, 0]], dtype=np.float64)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    out = pcd_ops.transform_to_marker_frame(pcd, [0.0, 0.0, 0.0], np.radians(90.0))
    # world +X with marker yaw +90° → local -Y
    assert np.allclose(np.asarray(out.points)[0], [0.0, -1.0, 0.0], atol=1e-9)


# ---------------------------------------------------------------------------
# level_ground
# ---------------------------------------------------------------------------


def test_level_ground_corrects_10deg_tilt(tilted_cloud):
    out, plane_before, tilt_deg = pcd_ops.level_ground(tilted_cloud)
    assert abs(tilt_deg - 10.0) < 0.5
    floor = np.asarray(out.points)[:10_000]
    assert np.abs(floor[:, 2]).max() < 0.01
    # Plane normal returned with z >= 0 (we flip it upward in level_ground).
    assert plane_before[2] >= 0


def test_level_ground_idempotent(tilted_cloud):
    """A second pass must not meaningfully move an already-levelled cloud.

    Not asserted bitwise: `translate_to_zero` re-seats the floor on every call,
    and RANSAC's fit on a floor with 1 mm noise carries a residual offset of
    order 1e-8 m. Exact equality only held while the already-level branch
    skipped the translation entirely — which was the bug that let a level cloud
    whose floor sat at z=5 stay at z=5 (see the test below). 1 µm is far below
    any physical relevance for LiDAR.
    """
    out, _, _ = pcd_ops.level_ground(tilted_cloud)
    out2, _, tilt2 = pcd_ops.level_ground(out)
    assert tilt2 < 0.05
    moved = np.abs(np.asarray(out2.points) - np.asarray(out.points)).max()
    assert moved < 1e-6, f"second pass moved points by {moved:.2e} m"


def test_level_ground_translates_an_already_level_cloud_to_zero():
    """`translate_to_zero` must apply even when no rotation is needed."""
    rng = np.random.default_rng(11)
    pts = np.column_stack(
        [
            rng.uniform(-2, 2, 3000),
            rng.uniform(-2, 2, 3000),
            np.full(3000, 5.0),
        ]
    )
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
    out, _, tilt = pcd_ops.level_ground(pcd)
    assert tilt < 0.05  # already level, so no rotation
    assert abs(float(np.asarray(out.points)[:, 2].mean())) < 1e-6  # but still moved to z=0


def test_level_ground_recovers_the_floor_despite_a_denser_wall(tilted_cloud):
    """A wall that out-votes the floor must not be mistaken for ground.

    This is the ppt_15july.pcd failure mode: the winning plane had normal
    ~[0.99, 0.11, 0.06] with 383 k inliers against the floor's 210 k, and the
    old code aligned it to +Z — turning a 1.8 m-thick room scan into a 21.6 m
    one. Here the retry loop should discard the wall and land on the floor.
    """
    rng = np.random.default_rng(12)
    wall = np.column_stack(
        [  # dense vertical slab at x ~ 0
            rng.normal(0, 0.01, 40_000),
            rng.uniform(-5, 5, 40_000),
            rng.uniform(0, 3, 40_000),
        ]
    )
    floor = np.column_stack(
        [  # sparser true floor at z ~ 0
            rng.uniform(0.5, 5, 8000),
            rng.uniform(-5, 5, 8000),
            rng.normal(0, 0.01, 8000),
        ]
    )
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.vstack([wall, floor])))
    out, plane, tilt = pcd_ops.level_ground(pcd, floor_band=1.0)
    assert tilt < 15.0, f"levelled against a {tilt:.1f}° plane — that is a wall"
    assert abs(plane[2]) > 0.95
    z = np.asarray(out.points)[:, 2]
    assert np.ptp(z) < 4.0  # still a 3 m-tall room, not stood on its side


def test_level_ground_raises_when_there_is_no_floor():
    """Vertical structure only — refuse rather than rotate 90°."""
    rng = np.random.default_rng(14)
    wall = np.column_stack(
        [
            rng.normal(0, 0.01, 20_000),
            rng.uniform(-5, 5, 20_000),
            rng.uniform(0, 3, 20_000),
        ]
    )
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(wall))
    with pytest.raises(ValueError, match="near-horizontal"):
        pcd_ops.level_ground(pcd, floor_band=1.0, max_tilt_deg=30.0)


def test_fit_ground_plane_prefers_the_floor_over_a_denser_wall():
    rng = np.random.default_rng(13)
    wall = np.column_stack(
        [
            rng.normal(0, 0.01, 30_000),
            rng.uniform(-5, 5, 30_000),
            rng.uniform(0, 3, 30_000),
        ]
    )
    floor = np.column_stack(
        [
            rng.uniform(0, 5, 5000),
            rng.uniform(-5, 5, 5000),
            rng.normal(0, 0.01, 5000),
        ]
    )
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.vstack([wall, floor])))
    n, d = pcd_ops.fit_ground_plane(pcd)
    assert abs(n[2]) > 0.95, f"normal {n} is not near-vertical"
    assert abs(d) < 0.2


# ---------------------------------------------------------------------------
# colorize_by_axis
# ---------------------------------------------------------------------------


def test_colorize_by_axis_writes_normalised_colors(cube_cloud):
    pcd_ops.colorize_by_axis(cube_cloud, axis="z")
    cols = np.asarray(cube_cloud.colors)
    assert cols.shape == (len(cube_cloud.points), 3)
    assert cols.min() >= 0.0 and cols.max() <= 1.0


def test_colorize_by_axis_rejects_bad_axis(cube_cloud):
    with pytest.raises(ValueError):
        pcd_ops.colorize_by_axis(cube_cloud, axis="w")


# ---------------------------------------------------------------------------
# display_downsample
# ---------------------------------------------------------------------------


def test_display_downsample_under_target():
    pts = np.random.default_rng(0).uniform(-5, 5, (800_000, 3))
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    out = pcd_ops.display_downsample(pcd, max_points=200_000)
    assert len(out.points) <= 200_000 * 1.25
    assert len(out.points) < len(pcd.points)


def test_display_downsample_passthrough_for_small_clouds(cube_cloud):
    # cube_cloud has 5 000 points — should be returned as-is.
    out = pcd_ops.display_downsample(cube_cloud, max_points=200_000)
    assert out is cube_cloud


# ---------------------------------------------------------------------------
# stats
# ---------------------------------------------------------------------------


def test_stats_matches_input(cube_cloud):
    s = pcd_ops.stats(cube_cloud)
    pts = np.asarray(cube_cloud.points)
    assert s.num_points == len(pts)
    assert np.allclose(s.min_bound, pts.min(axis=0))
    assert np.allclose(s.max_bound, pts.max(axis=0))
    # Cube is [-1, 1]^3 → describe() should mention "5,000 points".
    assert "5,000 points" in s.describe()
