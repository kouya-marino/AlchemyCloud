"""The home-pose marker on the occupancy preview.

`home` is world (0, 0) — where "Apply Marker as Origin" puts the robot. It is
*not* the YAML `origin`, which is the world position of the image's lower-left
corner. Nothing in the preview showed where home landed, so a map whose SLAM
origin sits outside the surveyed floor exported silently with the robot
starting in unknown space.
"""

from __future__ import annotations

import numpy as np
import open3d as o3d
import pytest

from alchemycloud import occupancy
from alchemycloud.occupancy_preview import OccupancyPreviewDialog


def _cloud(arr):
    return o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.asarray(arr, dtype=np.float64)))


def _blob(cx, cy, n=4000, span=4.0, seed=0):
    rng = np.random.default_rng(seed)
    return np.c_[
        rng.uniform(cx - span, cx + span, n),
        rng.uniform(cy - span, cy + span, n),
        rng.uniform(0.0, 1.0, n),
    ]


def _grid(pts, **kw):
    params = dict(resolution=0.1, z_min=-1.0, z_max=2.0)
    params.update(kw)
    return occupancy.build(_cloud(pts), occupancy.GridParams(**params))


# ---------------------------------------------------------------------------
# The mapping itself
# ---------------------------------------------------------------------------


def test_home_pixel_inverts_the_map_server_convention():
    """home_pixel must be the exact inverse of pixel -> world."""
    grid = _grid(_blob(3.0, 2.0))
    col, row = OccupancyPreviewDialog.home_pixel(grid)
    ox, oy, _ = grid.origin
    res = grid.resolution
    wx = ox + (col + 0.5) * res
    wy = oy + (grid.height - 1 - row + 0.5) * res
    assert wx == pytest.approx(0.0, abs=1e-9)
    assert wy == pytest.approx(0.0, abs=1e-9)


def test_home_pixel_is_not_the_yaml_origin():
    """A regression guard: confusing the two is the whole point of this feature."""
    grid = _grid(_blob(3.0, 2.0))
    col, row = OccupancyPreviewDialog.home_pixel(grid)
    # The YAML origin is the lower-left corner, i.e. pixel (0, height-1).
    assert (round(col), round(row)) != (0, grid.height - 1)
    assert grid.origin[0] != 0.0 and grid.origin[1] != 0.0


def test_home_is_inside_a_map_that_straddles_the_origin():
    grid = _grid(_blob(0.0, 0.0))
    col, row = OccupancyPreviewDialog.home_pixel(grid)
    assert 0 <= col < grid.width
    assert 0 <= row < grid.height


def test_home_is_outside_a_map_far_from_the_origin():
    grid = _grid(_blob(100.0, 100.0))
    col, row = OccupancyPreviewDialog.home_pixel(grid)
    assert not (0 <= col < grid.width and 0 <= row < grid.height)


@pytest.mark.parametrize("res", [0.02, 0.05, 0.1, 0.25])
def test_home_pixel_is_resolution_independent_in_world_terms(res):
    grid = _grid(_blob(3.0, 2.0), resolution=res)
    col, row = OccupancyPreviewDialog.home_pixel(grid)
    ox, oy, _ = grid.origin
    wx = ox + (col + 0.5) * res
    wy = oy + (grid.height - 1 - row + 0.5) * res
    assert (wx, wy) == pytest.approx((0.0, 0.0), abs=1e-9)


# ---------------------------------------------------------------------------
# What the dialog shows
# ---------------------------------------------------------------------------


def test_status_reports_the_cell_under_home(qapp):
    pts = _blob(0.0, 0.0)
    dlg = OccupancyPreviewDialog(_cloud(pts))
    dlg._update_status(_grid(pts))
    text = dlg.status_label.text()
    assert "home (0,0) at px" in text
    dlg.close()


def test_status_warns_when_home_is_off_the_map(qapp):
    pts = _blob(100.0, 100.0)
    dlg = OccupancyPreviewDialog(_cloud(pts))
    dlg._update_status(_grid(pts))
    text = dlg.status_label.text()
    assert "OUTSIDE the map" in text
    assert "#d66" in text, "an off-map home must be flagged, not stated neutrally"
    dlg.close()


def test_status_flags_home_on_unknown_but_not_on_free(qapp):
    """Home in unknown space means the robot would localise into grey."""
    pts = _blob(0.0, 0.0)
    dlg = OccupancyPreviewDialog(_cloud(pts))

    # Binary policy with a FREE background puts home on free space.
    dlg._update_status(_grid(pts, unknown_outside_footprint=False))
    assert "#888" in dlg.status_label.text()

    # Tristate leaves never-observed cells UNKNOWN; home sits in one here
    # because the blob is hollow at the exact origin only by chance, so force
    # it with a cloud that avoids the origin but still contains it in-frame.
    ring = np.vstack([_blob(0.0, 6.0, seed=1), _blob(0.0, -6.0, seed=2)])
    dlg._update_status(_grid(ring, height_threshold=0.5, padding_cells=2))
    assert "#d66" in dlg.status_label.text()
    dlg.close()


def test_overlay_is_drawn_and_survives_downscaling(qapp):
    """A big grid scaled into a small label must still show the marker.

    Drawing on the full-resolution map first would shrink the cross with
    everything else; on a 2000+ cell grid it would disappear entirely.
    """
    pts = _blob(0.0, 0.0, n=20000, span=40.0)
    grid = _grid(pts, resolution=0.05)
    assert grid.width > 1000 and grid.height > 1000

    dlg = OccupancyPreviewDialog(_cloud(pts))
    dlg.preview_label.resize(300, 240)
    dlg._update_preview_pixmap(grid)
    pm = dlg.preview_label.pixmap()
    assert pm is not None and pm.width() > 0

    img = pm.toImage()
    # The map is strictly greyscale (0 / 205 / 254); the marker is cyan, so any
    # pixel with a channel spread is overlay.
    found = False
    for y in range(0, img.height(), 2):
        for x in range(0, img.width(), 2):
            c = img.pixelColor(x, y)
            if max(c.red(), c.green(), c.blue()) - min(c.red(), c.green(), c.blue()) > 40:
                found = True
                break
        if found:
            break
    assert found, "no coloured overlay pixel found in the scaled preview"
    dlg.close()


def test_no_overlay_when_home_is_off_the_map(qapp):
    pts = _blob(100.0, 100.0)
    grid = _grid(pts)
    dlg = OccupancyPreviewDialog(_cloud(pts))
    dlg.preview_label.resize(300, 240)
    dlg._update_preview_pixmap(grid)
    img = dlg.preview_label.pixmap().toImage()
    for y in range(0, img.height(), 3):
        for x in range(0, img.width(), 3):
            c = img.pixelColor(x, y)
            spread = max(c.red(), c.green(), c.blue()) - min(c.red(), c.green(), c.blue())
            assert spread <= 40, f"overlay drawn at ({x},{y}) for an off-map home"
    dlg.close()


def test_overlay_does_not_touch_the_exported_image(qapp, tmp_path):
    """The marker is preview-only; the PGM must stay a clean map_server image."""
    pts = _blob(0.0, 0.0)
    params = occupancy.GridParams(resolution=0.1, z_min=-1.0, z_max=2.0)
    grid = occupancy.build(_cloud(pts), params)
    before = grid.image.copy()

    dlg = OccupancyPreviewDialog(_cloud(pts))
    dlg.preview_label.resize(300, 240)
    dlg._update_preview_pixmap(grid)
    assert np.array_equal(grid.image, before), "preview mutated the grid"

    pgm, _yaml = occupancy.save(grid, str(tmp_path / "m.pgm"), params)
    from PIL import Image

    written = np.array(Image.open(pgm))
    assert set(np.unique(written).tolist()) <= {occupancy.OCC, occupancy.FREE, occupancy.UNKNOWN}
    dlg.close()


def test_preview_survives_a_label_resize(qapp):
    pts = _blob(0.0, 0.0)
    grid = _grid(pts)
    dlg = OccupancyPreviewDialog(_cloud(pts))
    dlg._last_grid = grid
    for size in ((320, 240), (640, 480), (120, 90)):
        dlg.preview_label.resize(*size)
        dlg._update_preview_pixmap(grid)
        assert dlg.preview_label.pixmap().width() > 0
    dlg.close()
