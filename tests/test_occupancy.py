"""Unit tests for `occupancy.build` / `save` — binary + tristate paths."""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest
import yaml
from PIL import Image

from alchemycloud import occupancy

# ---------------------------------------------------------------------------
# Binary policy
# ---------------------------------------------------------------------------


def test_grid_params_defaults():
    p = occupancy.GridParams()
    assert p.resolution == 0.05
    assert p.method == "projection"
    assert p.min_points_per_cell == 1
    assert p.unknown_outside_footprint is False
    assert p.height_threshold is None
    assert p.occupied_threshold == 0.65
    assert p.free_threshold == 0.196


def test_binary_projection_marks_floor_occupied(floor_with_chair):
    params = occupancy.GridParams(resolution=0.1, z_min=-0.1, z_max=1.0)
    grid = occupancy.build(floor_with_chair, params)
    # 2 000 random floor points + 200 chair points → with 0.1 m cells and a
    # 5-cell pad, the data footprint dominates. ~40-60 % of cells end up
    # occupied (the rest are padding or never-hit randomness).
    occ_frac = float((grid.image == occupancy.OCC).mean())
    assert occ_frac > 0.30
    assert grid.image.shape == (grid.height, grid.width)


def test_binary_voxel3d_produces_grid(floor_with_chair):
    params = occupancy.GridParams(resolution=0.1, z_min=-0.1, z_max=1.0, method="voxel3d")
    grid = occupancy.build(floor_with_chair, params)
    assert grid.image.size > 0
    assert (grid.image == occupancy.OCC).sum() > 0


def test_unknown_method_raises(floor_with_chair):
    params = occupancy.GridParams(method="bogus")
    with pytest.raises(ValueError):
        occupancy.build(floor_with_chair, params)


def test_empty_z_range_raises(floor_with_chair):
    # No points have z > 10 m.
    params = occupancy.GridParams(z_min=10.0, z_max=20.0)
    with pytest.raises(ValueError):
        occupancy.build(floor_with_chair, params)


# ---------------------------------------------------------------------------
# Tristate policy
# ---------------------------------------------------------------------------


def test_tristate_separates_free_and_occupied(floor_with_chair):
    """Floor at z=0, chair at z=0.5, threshold=0.2:
    * chair footprint → occupied (some Z > 0.2)
    * floor-only cells → free (have points but all <= 0.2)
    * empty (padding) cells → unknown
    """
    params = occupancy.GridParams(
        resolution=0.1,
        z_min=-0.1,
        z_max=1.0,
        height_threshold=0.2,
    )
    grid = occupancy.build(floor_with_chair, params)
    occ = int((grid.image == occupancy.OCC).sum())
    free = int((grid.image == occupancy.FREE).sum())
    unknown = int((grid.image == occupancy.UNKNOWN).sum())

    # Chair is 0.6×0.6 m → ~6×6 cells → ~36 occupied, plus randomness slack.
    assert 10 < occ < 100
    # Floor 4×4 m → ~1600 cells, minus the chair area.
    assert free > 100
    # Padding cells around the data → unknown.
    assert unknown > 0
    # All three classes account for the full image.
    assert occ + free + unknown == grid.image.size


def test_tristate_with_voxel3d(floor_with_chair):
    params = occupancy.GridParams(
        resolution=0.1,
        z_min=-0.1,
        z_max=1.0,
        method="voxel3d",
        height_threshold=0.2,
    )
    grid = occupancy.build(floor_with_chair, params)
    occ = int((grid.image == occupancy.OCC).sum())
    free = int((grid.image == occupancy.FREE).sum())
    assert occ > 0
    assert free > 100


# ---------------------------------------------------------------------------
# Save / YAML round-trip
# ---------------------------------------------------------------------------


def test_save_roundtrip_pgm_yaml(floor_with_chair):
    params = occupancy.GridParams(
        resolution=0.1,
        z_min=-0.1,
        z_max=1.0,
        height_threshold=0.2,
    )
    grid = occupancy.build(floor_with_chair, params)
    with tempfile.NamedTemporaryFile(suffix=".pgm", delete=False) as f:
        pgm_path = f.name
    yaml_path = pgm_path.replace(".pgm", ".yaml")
    try:
        out_pgm, out_yaml = occupancy.save(grid, pgm_path, params)
        # PGM round-trip via PIL
        loaded = np.array(Image.open(out_pgm))
        assert np.array_equal(loaded, grid.image)
        # YAML has the expected keys
        with open(out_yaml) as f:
            meta = yaml.safe_load(f)
        assert meta["image"] == os.path.basename(str(out_pgm))
        assert meta["resolution"] == params.resolution
        assert meta["occupied_thresh"] == params.occupied_threshold
        assert meta["free_thresh"] == params.free_threshold
        assert len(meta["origin"]) == 3
    finally:
        for p in (pgm_path, yaml_path):
            if os.path.exists(p):
                os.unlink(p)
