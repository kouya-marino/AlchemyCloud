"""Generate 2D occupancy grids from point clouds and save in ROS map_server format.

Two methods are supported:
  * "projection"   — top-down 2D projection of points whose Z is in [z_min, z_max]
  * "voxel3d"      — full 3D voxelization, then collapse to 2D along Z

Two occupancy policies:
  * **binary** (`height_threshold=None`, default) — a cell is occupied if
    ``min_points_per_cell`` or more points project into it; *every* other cell
    (observed or not) takes the background colour set by
    ``unknown_outside_footprint``.
  * **tristate** (`height_threshold` set) — for each in-slab cell:
        - if ``min_points_per_cell`` returns have z > threshold → occupied
        - else if it has any points (all ≤ threshold) → free
        - never-observed cells → the background set by ``tristate_empty_unknown``
    This is the policy navigation stacks expect (threshold ≈ robot height)
    and matches the ``occupancy_grid_tuner.py`` workflow.

Note the two background flags are **not** interchangeable. In tristate,
"background" means *never observed*, and observed floor is still FREE — so
UNKNOWN there marks genuinely unseen space. In binary there is no observed/
unobserved distinction at all: every non-occupied cell is background, so
``unknown_outside_footprint=True`` paints the navigable interior grey too and
the exported map contains zero FREE cells. A UI must not drive both from one
control.

Output is the standard ROS map_server PGM + YAML pair:
  occupied = 0
  free     = 254
  unknown  = 205
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import open3d as o3d
import yaml
from PIL import Image

OCC = 0
FREE = 254
UNKNOWN = 205

# Refuse to allocate beyond this many cells. The resolution spin box bottoms out
# at 0.005 m, which on a 200 m scan asks for 1.6 G cells — a 6.4 GB int64 count
# array plus a 1.6 GB image, i.e. swap-thrash or the OOM killer rather than a
# clean MemoryError. Both prototype scripts this app grew out of capped at 4 M.
MAX_CELLS = 25_000_000


@dataclass
class GridParams:
    resolution: float = 0.05  # meters per cell
    z_min: float = 0.1  # ignore points below this (drop ground)
    z_max: float = 2.0  # ignore points above this (drop ceiling)
    method: str = "projection"  # "projection" | "voxel3d"
    # Minimum returns needed to call a cell occupied. Applies to every
    # method/policy combination: raw points for "projection", occupied voxels
    # in the column for "voxel3d", and returns above the threshold in tristate.
    min_points_per_cell: int = 1
    # BINARY ONLY. True paints every non-occupied cell UNKNOWN — including
    # observed free floor, so the map ends up with no FREE cells at all. Not
    # the same meaning as `tristate_empty_unknown`; see the module docstring.
    unknown_outside_footprint: bool = False
    padding_cells: int = 5  # extra cells of margin around the data
    # Tristate policy: when set, switches free/occupied/unknown logic. Cells
    # with `min_points_per_cell` returns above height_threshold become OCCUPIED;
    # cells with points but none above become FREE; never-observed cells follow
    # `tristate_empty_unknown`.
    height_threshold: float | None = None
    # Tristate only: empty (unobserved) cells → UNKNOWN (grey) if True,
    # else FREE (white). Default True preserves the navigation-style output.
    tristate_empty_unknown: bool = True
    occupied_threshold: float = 0.65  # written into YAML
    free_threshold: float = 0.196  # written into YAML
    negate: int = 0  # written into YAML


@dataclass
class OccupancyGrid:
    image: np.ndarray  # uint8 HxW (already in map_server intensity convention)
    resolution: float
    origin: tuple[float, float, float]  # (x, y, yaw=0) of the lower-left pixel in world coords

    @property
    def height(self) -> int:
        return self.image.shape[0]

    @property
    def width(self) -> int:
        return self.image.shape[1]


def _filter_z(points: np.ndarray, z_min: float, z_max: float) -> np.ndarray:
    """Points whose Z lies in [z_min, z_max], excluding non-finite rows.

    The finite check matters: a single ``nan`` makes ``.astype(np.int64)`` raise
    "cannot convert float NaN to integer", and a single ``1e9`` glitch return
    (a specular reflection, say) inflates the grid extent enough to ask for a
    terabyte-scale allocation. Both are routine in raw LiDAR.
    """
    finite = np.isfinite(points).all(axis=1)
    in_band = (points[:, 2] >= z_min) & (points[:, 2] <= z_max)
    return points[finite & in_band]


@dataclass
class _Frame:
    """Grid geometry plus each input point's integer cell index."""

    x_min: float
    y_min: float
    width: int
    height: int
    ix: np.ndarray
    iy: np.ndarray


def _outlier_hint(xy: np.ndarray) -> str:
    """Extra sentence for the too-big-grid error when a few stray points, not
    the scan itself, are what inflated the extent.

    A single specular return at x=1e9 is finite, so it is real data we must not
    silently discard — but it is almost certainly a glitch, and saying so beats
    telling the user to crop a 4 m room.
    """
    if len(xy) < 20:
        return ""
    lo = np.percentile(xy, 1.0, axis=0)
    hi = np.percentile(xy, 99.0, axis=0)
    robust = float(np.max(hi - lo))
    full = float(np.max(xy.max(axis=0) - xy.min(axis=0)))
    if robust <= 0 or full <= 3.0 * robust:
        return ""
    outside = int((((xy < lo - robust) | (xy > hi + robust)).any(axis=1)).sum())
    return (
        f" Note: 99 % of the cloud spans only {robust:,.1f} m but the full extent is "
        f"{full:,.1f} m — {outside:,} point(s) lie far outside the bulk and are probably "
        "sensor glitches. Run Statistical Outlier Removal or Z-Clip first."
    )


def _check_size(width: int, height: int, params: GridParams, xy: np.ndarray | None = None) -> None:
    if width <= 0 or height <= 0:
        raise ValueError(
            f"Grid would be {width} x {height} cells. The cloud has no extent in one "
            "axis — increase 'Padding (cells)' above 0, or check the Z range is not "
            "selecting a single scan line."
        )
    cells = width * height
    if cells > MAX_CELLS:
        span_x = width * params.resolution
        span_y = height * params.resolution
        hint = _outlier_hint(xy) if xy is not None else ""
        raise ValueError(
            f"Grid would be {width:,} x {height:,} = {cells / 1e6:,.1f} M cells "
            f"({span_x:,.0f} x {span_y:,.0f} m at {params.resolution} m/cell), above the "
            f"{MAX_CELLS / 1e6:.0f} M limit. Raise the resolution, tighten the Z range, or "
            f"crop the cloud first.{hint}"
        )


def _frame_from_xy(xy: np.ndarray, params: GridParams) -> _Frame:
    """Frame derived from continuous world XY (the projection method)."""
    res = params.resolution
    pad = params.padding_cells
    if res <= 0:
        raise ValueError("resolution must be > 0")

    x_min = float(xy[:, 0].min()) - pad * res
    y_min = float(xy[:, 1].min()) - pad * res
    x_max = float(xy[:, 0].max()) + pad * res
    y_max = float(xy[:, 1].max()) + pad * res

    # At least one cell per axis: a cloud with zero extent in X or Y (a single
    # point, or one scan line) would otherwise give a zero-width array, and the
    # clip below would produce index -1 and an IndexError.
    width = max(int(np.ceil((x_max - x_min) / res)), 1)
    height = max(int(np.ceil((y_max - y_min) / res)), 1)
    _check_size(width, height, params, xy)

    ix = np.floor((xy[:, 0] - x_min) / res).astype(np.int64)
    iy = np.floor((xy[:, 1] - y_min) / res).astype(np.int64)
    np.clip(ix, 0, width - 1, out=ix)
    np.clip(iy, 0, height - 1, out=iy)
    return _Frame(x_min, y_min, width, height, ix, iy)


def _frame_from_voxel_index(vx: np.ndarray, vy: np.ndarray, params: GridParams) -> _Frame:
    """Frame derived from integer voxel indices (the voxel3d method).

    Exact by construction. The old path turned voxel indices back into float
    centres ``(k + 0.5) * res`` and re-derived the cell with
    ``floor((x - x_min) / res)``; that round trip is not exact in float64, so
    for a few percent of columns the expression evaluated to ``n - 1e-14`` and
    ``floor`` returned ``n - 1``. Two voxel columns then merged and column ``n``
    was left empty — a one-cell hole through a solid wall, with the pattern
    shifting as ``padding_cells`` changed. Staying in integers removes the
    class of bug entirely.
    """
    res = params.resolution
    pad = params.padding_cells
    if res <= 0:
        raise ValueError("resolution must be > 0")

    vx_min, vy_min = int(vx.min()), int(vy.min())
    width = int(vx.max()) - vx_min + 1 + 2 * pad
    height = int(vy.max()) - vy_min + 1 + 2 * pad
    _check_size(width, height, params)

    ix = (vx - vx_min + pad).astype(np.int64)
    iy = (vy - vy_min + pad).astype(np.int64)
    # Cell (pad, pad) is voxel (vx_min, vy_min), whose lower-left corner is at
    # vx_min * res — so the grid's own lower-left corner is pad cells before it.
    return _Frame((vx_min - pad) * res, (vy_min - pad) * res, width, height, ix, iy)


def _binary_image(frame: _Frame, params: GridParams, weights: np.ndarray | None = None) -> np.ndarray:
    """Binary policy: a cell is OCC once it accumulates `min_points_per_cell`."""
    counts = np.zeros((frame.height, frame.width), dtype=np.int64)
    if weights is None:
        np.add.at(counts, (frame.iy, frame.ix), 1)
    else:
        np.add.at(counts, (frame.iy, frame.ix), weights)
    background = UNKNOWN if params.unknown_outside_footprint else FREE
    image = np.full((frame.height, frame.width), background, dtype=np.uint8)
    image[counts >= max(int(params.min_points_per_cell), 1)] = OCC
    # Row 0 is the top row in map_server; world Y grows upward, so flip.
    return np.flipud(image)


def _tristate_image(frame: _Frame, params: GridParams, z: np.ndarray) -> np.ndarray:
    """Tristate policy: OCC where enough returns sit above `height_threshold`,
    FREE where a cell was observed but only below it, background elsewhere."""
    thr = float(params.height_threshold)
    min_hits = max(int(params.min_points_per_cell), 1)

    observed = np.zeros((frame.height, frame.width), dtype=bool)
    observed[frame.iy, frame.ix] = True

    above = z > thr
    # Count, not just flag: min_points_per_cell now gates the occupied test too,
    # so a single stray return above the threshold no longer plants an obstacle.
    occ_hits = np.zeros((frame.height, frame.width), dtype=np.int64)
    np.add.at(occ_hits, (frame.iy[above], frame.ix[above]), 1)

    background = UNKNOWN if params.tristate_empty_unknown else FREE
    image = np.full((frame.height, frame.width), background, dtype=np.uint8)
    image[observed] = FREE
    image[occ_hits >= min_hits] = OCC
    return np.flipud(image)


def from_projection(pcd: o3d.geometry.PointCloud, params: GridParams) -> OccupancyGrid:
    pts = _filter_z(np.asarray(pcd.points), params.z_min, params.z_max)
    if len(pts) == 0:
        raise ValueError("No points in the requested Z range — cannot build a grid.")
    frame = _frame_from_xy(pts[:, :2], params)
    if params.height_threshold is not None:
        image = _tristate_image(frame, params, pts[:, 2])
    else:
        image = _binary_image(frame, params)
    return OccupancyGrid(image=image, resolution=params.resolution, origin=(frame.x_min, frame.y_min, 0.0))


def from_voxel3d(pcd: o3d.geometry.PointCloud, params: GridParams) -> OccupancyGrid:
    """Voxelize at `resolution`, then collapse occupied voxels to 2D.

    ``min_points_per_cell`` counts *occupied voxels* in a column here, not raw
    points — the 3D dedupe means every column would otherwise hold exactly one
    entry, so any value above 1 produced a blank map.
    """
    res = params.resolution
    if res <= 0:
        raise ValueError("resolution must be > 0")
    pts = _filter_z(np.asarray(pcd.points), params.z_min, params.z_max)
    if len(pts) == 0:
        raise ValueError("No points in the requested Z range — cannot build a grid.")

    vx = np.floor(pts[:, 0] / res).astype(np.int64)
    vy = np.floor(pts[:, 1] / res).astype(np.int64)
    vz = np.floor(pts[:, 2] / res).astype(np.int64)
    voxels = np.unique(np.stack([vx, vy, vz], axis=1), axis=0)

    frame = _frame_from_voxel_index(voxels[:, 0], voxels[:, 1], params)
    if params.height_threshold is not None:
        # Voxel centre height, so a voxel straddling the threshold resolves the
        # same way it would for its midpoint.
        z_centers = (voxels[:, 2].astype(np.float64) + 0.5) * res
        image = _tristate_image(frame, params, z_centers)
    else:
        image = _binary_image(frame, params)
    return OccupancyGrid(image=image, resolution=res, origin=(frame.x_min, frame.y_min, 0.0))


def build(pcd: o3d.geometry.PointCloud, params: GridParams) -> OccupancyGrid:
    if params.method == "projection":
        return from_projection(pcd, params)
    if params.method == "voxel3d":
        return from_voxel3d(pcd, params)
    raise ValueError(f"Unknown method: {params.method!r}")


def save(grid: OccupancyGrid, pgm_path: str, params: GridParams) -> tuple[Path, Path]:
    """Write the map_server PGM + YAML pair. Returns the two paths written.

    ``params`` must be the params the grid was *built* with — the thresholds and
    ``negate`` written to the YAML come from here while the geometry comes from
    ``grid``, so passing a different object silently describes the image wrongly.
    """
    pgm_path = Path(pgm_path)
    # Append rather than replace: `with_suffix` would turn "3F_map_v1.2" into
    # "3F_map_v1.pgm", quietly dropping part of the name the user typed.
    if pgm_path.suffix.lower() != ".pgm":
        pgm_path = pgm_path.with_name(pgm_path.name + ".pgm")
    yaml_path = pgm_path.with_suffix(".yaml")

    if not pgm_path.parent.exists():
        raise FileNotFoundError(f"Directory does not exist: {pgm_path.parent}")
    if int(params.negate) != 0:
        # map_server reads occupancy as (255 - value)/255 when negate is 0 and
        # value/255 when it is 1. Pixels here are always written in the former
        # convention, so a non-zero negate would invert the whole map.
        raise ValueError(
            "negate must be 0: pixels are written as occupied=0 / free=254, which "
            "map_server only interprets correctly with negate: 0."
        )

    Image.fromarray(grid.image, mode="L").save(pgm_path)

    meta = {
        "image": pgm_path.name,
        "resolution": float(grid.resolution),
        "origin": [float(grid.origin[0]), float(grid.origin[1]), float(grid.origin[2])],
        "negate": int(params.negate),
        "occupied_thresh": float(params.occupied_threshold),
        "free_thresh": float(params.free_threshold),
        # nav2 defaults to trinary when absent; stating it removes the ambiguity
        # and is ignored by ROS 1 map_server.
        "mode": "trinary",
    }
    with yaml_path.open("w") as f:
        yaml.safe_dump(meta, f, sort_keys=False, allow_unicode=True)
    return pgm_path, yaml_path
