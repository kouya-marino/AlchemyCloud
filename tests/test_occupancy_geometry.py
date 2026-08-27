"""Occupancy-grid geometry and robustness regressions.

The existing `test_occupancy.py` checks cell *counts* and the PGM/YAML round
trip, but never asserts where anything actually is — hard-coding the YAML
origin to (0,0,0) or deleting both vertical flips still passed the whole suite.
These tests pin the geometry, plus the edge cases that used to crash or
silently produce a wrong map.
"""

from __future__ import annotations

import numpy as np
import open3d as o3d
import pytest
import yaml
from PIL import Image

from alchemycloud import occupancy


def cloud(arr: np.ndarray) -> o3d.geometry.PointCloud:
    return o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.asarray(arr, dtype=np.float64)))


def block(x0, x1, y0, y1, z0=0.5, z1=1.0, n=6000, seed=1):
    rng = np.random.default_rng(seed)
    return np.c_[rng.uniform(x0, x1, n), rng.uniform(y0, y1, n), rng.uniform(z0, z1, n)]


def pixel_to_world(col, row, height, origin, res):
    """ROS map_server convention: origin is the world position of the image's
    lower-left pixel, and row 0 is the top row."""
    return origin[0] + (col + 0.5) * res, origin[1] + (height - 1 - row + 0.5) * res


# ---------------------------------------------------------------------------
# Geometry: the property that matters most
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method", ["projection", "voxel3d"])
def test_occupied_cells_map_back_to_the_obstacle(method):
    """An asymmetric L must round-trip through (pixel, origin, resolution).

    Asymmetric on purpose: a symmetric shape survives a transpose or a mirror,
    so it cannot detect a flipud/origin mismatch.
    """
    res = 0.10
    long_arm = block(2.0, 6.0, 1.0, 1.3)  # runs along X at low Y
    short_arm = block(2.0, 2.3, 1.0, 3.0, seed=2)  # runs along Y at low X
    floor = block(1.0, 7.0, 0.5, 4.0, z0=0.0, z1=0.02, n=30000, seed=3)
    pcd = cloud(np.vstack([long_arm, short_arm, floor]))

    grid = occupancy.build(
        pcd,
        occupancy.GridParams(
            resolution=res,
            z_min=-0.1,
            z_max=2.0,
            method=method,
            height_threshold=0.3,
            padding_cells=5,
        ),
    )
    rows, cols = np.where(grid.image == occupancy.OCC)
    assert len(rows) > 50
    wx, wy = pixel_to_world(cols, rows, grid.height, grid.origin, res)

    tol = 2 * res
    assert abs(wx.min() - 2.0) < tol and abs(wx.max() - 6.0) < tol
    assert abs(wy.min() - 1.0) < tol and abs(wy.max() - 3.0) < tol

    # Orientation: the long arm lies along X at low Y; the stub along Y at low X.
    low_y, low_x = wy < 1.45, wx < 2.45
    assert (wx[low_y].max() - wx[low_y].min()) > 3.0
    assert (wy[low_x].max() - wy[low_x].min()) > 1.8
    # Not mirrored: the stub sits at the arm's low-X end.
    assert wx[low_x].mean() < wx[low_y].mean()


def test_yaml_origin_survives_the_pgm_round_trip(tmp_path):
    res = 0.10
    pcd = cloud(np.vstack([block(2.0, 6.0, 1.0, 1.3), block(2.0, 2.3, 1.0, 3.0, seed=2)]))
    params = occupancy.GridParams(resolution=res, z_min=-0.1, z_max=2.0, padding_cells=4)
    grid = occupancy.build(pcd, params)
    pgm, yml = occupancy.save(grid, str(tmp_path / "m.pgm"), params)

    meta = yaml.safe_load(yml.read_text())
    img = np.array(Image.open(pgm))
    assert meta["image"] == "m.pgm"
    assert meta["resolution"] == pytest.approx(res)
    assert meta["negate"] == 0
    assert meta["mode"] == "trinary"
    assert img.shape == grid.image.shape
    assert np.array_equal(img, grid.image)
    # A non-trivial origin is the whole point — (0,0,0) would pass a weaker test.
    assert meta["origin"][0] != 0.0 and meta["origin"][1] != 0.0

    rows, cols = np.where(img == occupancy.OCC)
    wx, wy = pixel_to_world(cols, rows, img.shape[0], meta["origin"], meta["resolution"])
    assert abs(wx.min() - 2.0) < 2 * res
    assert abs(wy.max() - 3.0) < 2 * res


def test_pgm_is_binary_p5_with_maxval_255(tmp_path):
    pcd = cloud(block(0, 3, 0, 3))
    params = occupancy.GridParams(resolution=0.1, z_min=-1, z_max=2)
    pgm, _ = occupancy.save(occupancy.build(pcd, params), str(tmp_path / "m.pgm"), params)
    head = pgm.read_bytes()[:16]
    assert head.startswith(b"P5")
    assert b"255" in head


def test_unknown_value_reads_as_unknown_under_map_server_maths():
    """UNKNOWN=205 must land above free_thresh, or map_server calls it free.

    p = (255 - value) / 255 for negate=0. The margin is only 7.8e-5, so this is
    worth pinning against a well-meaning tweak to either constant.
    """
    p = (255 - occupancy.UNKNOWN) / 255
    d = occupancy.GridParams()
    assert p > d.free_threshold
    assert p < d.occupied_threshold
    assert (255 - occupancy.FREE) / 255 < d.free_threshold
    assert (255 - occupancy.OCC) / 255 > d.occupied_threshold


# ---------------------------------------------------------------------------
# voxel3d integer indexing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("pad", [0, 3, 5, 7])
def test_voxel3d_leaves_no_holes_in_a_solid_wall(pad):
    """Float voxel centres used to re-derive the wrong cell for ~5 % of columns,
    punching one-cell holes through a wall, with the pattern depending on
    padding_cells."""
    rng = np.random.default_rng(5)
    wall = np.c_[rng.uniform(0, 4, 200_000), rng.uniform(1.0, 1.05, 200_000), rng.uniform(0.2, 1.5, 200_000)]
    grid = occupancy.build(
        cloud(wall),
        occupancy.GridParams(
            resolution=0.05,
            z_min=0,
            z_max=2,
            method="voxel3d",
            padding_cells=pad,
        ),
    )
    per_col = (grid.image == occupancy.OCC).sum(axis=0)
    lit = np.where(per_col > 0)[0]
    interior = per_col[lit.min() : lit.max() + 1]
    assert (interior == 0).sum() == 0, f"{(interior == 0).sum()} hole(s) with pad={pad}"


def test_voxel3d_origin_is_on_the_voxel_lattice():
    rng = np.random.default_rng(6)
    pts = np.c_[rng.uniform(1, 3, 5000), rng.uniform(2, 4, 5000), rng.uniform(0, 1, 5000)]
    res, pad = 0.05, 5
    grid = occupancy.build(
        cloud(pts),
        occupancy.GridParams(
            resolution=res,
            z_min=-1,
            z_max=2,
            method="voxel3d",
            padding_cells=pad,
        ),
    )
    # origin must be an exact integer multiple of the resolution
    for o in grid.origin[:2]:
        assert abs(o / res - round(o / res)) < 1e-9


def test_min_points_per_cell_does_not_blank_a_voxel3d_map():
    """It used to: 3D dedupe left one entry per column, so any value above 1
    produced a map of a few float-rounding collisions."""
    rng = np.random.default_rng(7)
    wall = np.c_[rng.uniform(0, 4, 150_000), rng.uniform(1.0, 1.05, 150_000), rng.uniform(0.2, 1.5, 150_000)]
    counts = [
        int(
            (
                occupancy.build(
                    cloud(wall),
                    occupancy.GridParams(
                        resolution=0.05,
                        z_min=0,
                        z_max=2,
                        method="voxel3d",
                        min_points_per_cell=m,
                    ),
                ).image
                == occupancy.OCC
            ).sum()
        )
        for m in (1, 2, 3)
    ]
    assert all(c > 50 for c in counts), counts


def test_min_points_per_cell_gates_the_tristate_occupied_test():
    """One stray return above the threshold should not plant an obstacle."""
    floor = block(0, 3, 0, 3, z0=0.0, z1=0.02, n=20000, seed=8)
    stray = np.array([[1.5, 1.5, 1.2]])
    pcd = cloud(np.vstack([floor, stray]))
    common = dict(resolution=0.1, z_min=-0.1, z_max=2.0, height_threshold=0.5)
    lenient = occupancy.build(pcd, occupancy.GridParams(min_points_per_cell=1, **common))
    strict = occupancy.build(pcd, occupancy.GridParams(min_points_per_cell=3, **common))
    assert (lenient.image == occupancy.OCC).sum() == 1
    assert (strict.image == occupancy.OCC).sum() == 0


# ---------------------------------------------------------------------------
# Robustness
# ---------------------------------------------------------------------------


def test_non_finite_points_are_filtered_not_fatal():
    rng = np.random.default_rng(9)
    pts = np.vstack([rng.uniform(0, 4, (2000, 3)), [[np.nan, 1, 0.5]], [[1, np.inf, 0.5]]])
    grid = occupancy.build(cloud(pts), occupancy.GridParams(resolution=0.1, z_min=-1, z_max=2))
    assert grid.image.shape == (50, 50)


def test_far_outlier_is_refused_with_an_actionable_message():
    """A finite 1e9 glitch is real data, so it is not silently dropped — but it
    must not be allowed to ask for a terabyte allocation either."""
    rng = np.random.default_rng(10)
    pts = np.vstack([rng.uniform(0, 4, (2000, 3)), [[1e9, 1, 0.5]]])
    with pytest.raises(ValueError, match="sensor glitches"):
        occupancy.build(cloud(pts), occupancy.GridParams(resolution=0.1, z_min=-1, z_max=2))


def test_resolution_too_fine_is_refused_before_allocating():
    rng = np.random.default_rng(11)
    pts = np.c_[rng.uniform(0, 200, 500), rng.uniform(0, 200, 500), rng.uniform(0, 1, 500)]
    with pytest.raises(ValueError, match="above the"):
        occupancy.build(cloud(pts), occupancy.GridParams(resolution=0.005, z_min=-1, z_max=2))


def test_zero_extent_axis_with_no_padding_does_not_crash():
    """`width = ceil(0/res) = 0` then `clip(ix, 0, -1)` gave index -1 into a
    zero-width array: IndexError."""
    line = np.c_[np.full(500, 1.0), np.linspace(0, 3, 500), np.full(500, 0.5)]
    grid = occupancy.build(
        cloud(line),
        occupancy.GridParams(
            resolution=0.1,
            z_min=0,
            z_max=1,
            padding_cells=0,
        ),
    )
    assert grid.width >= 1 and grid.height >= 1


def test_single_point_cloud_builds():
    grid = occupancy.build(cloud([[1.0, 2.0, 0.5]]), occupancy.GridParams(resolution=0.1, z_min=0, z_max=1))
    assert (grid.image == occupancy.OCC).sum() == 1


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------


def test_dotted_filename_is_not_truncated(tmp_path):
    """`with_suffix` turned "3F_map_v1.2" into "3F_map_v1.pgm"."""
    params = occupancy.GridParams(resolution=0.1, z_min=-1, z_max=2)
    grid = occupancy.build(cloud(block(0, 3, 0, 3)), params)
    pgm, yml = occupancy.save(grid, str(tmp_path / "3F_map_v1.2"), params)
    assert pgm.name == "3F_map_v1.2.pgm"
    assert yml.name == "3F_map_v1.2.yaml"


def test_negate_is_refused_because_pixels_are_written_un_negated(tmp_path):
    params = occupancy.GridParams(resolution=0.1, z_min=-1, z_max=2)
    grid = occupancy.build(cloud(block(0, 3, 0, 3)), params)
    with pytest.raises(ValueError, match="negate must be 0"):
        occupancy.save(grid, str(tmp_path / "m.pgm"), occupancy.GridParams(negate=1))


def test_save_to_a_missing_directory_raises(tmp_path):
    params = occupancy.GridParams(resolution=0.1, z_min=-1, z_max=2)
    grid = occupancy.build(cloud(block(0, 3, 0, 3)), params)
    with pytest.raises(FileNotFoundError):
        occupancy.save(grid, str(tmp_path / "nope" / "m.pgm"), params)


# ---------------------------------------------------------------------------
# The two background flags mean different things
# ---------------------------------------------------------------------------


def test_binary_unknown_background_erases_all_free_space():
    """Pins the asymmetry a shared UI control would hide: in binary there is no
    observed/unobserved distinction, so UNKNOWN background leaves zero FREE."""
    floor = block(0, 4, 0, 4, z0=0.0, z1=0.02, n=20000, seed=12)
    chair = block(1, 2, 1, 2, z0=0.0, z1=0.8, n=4000, seed=13)
    pcd = cloud(np.vstack([floor, chair]))
    common = dict(resolution=0.1, z_min=-0.1, z_max=2.0)

    binary_unknown = occupancy.build(pcd, occupancy.GridParams(unknown_outside_footprint=True, **common))
    assert (binary_unknown.image == occupancy.FREE).sum() == 0

    tristate_unknown = occupancy.build(
        pcd, occupancy.GridParams(height_threshold=0.2, tristate_empty_unknown=True, **common)
    )
    assert (tristate_unknown.image == occupancy.FREE).sum() > 0
    assert (tristate_unknown.image == occupancy.UNKNOWN).sum() > 0
