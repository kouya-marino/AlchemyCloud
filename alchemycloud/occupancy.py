"""Generate 2D occupancy grids from point clouds and save in ROS map_server format.

Two methods are supported:
  * "projection"   — top-down 2D projection of points whose Z is in [z_min, z_max]
  * "voxel3d"      — full 3D voxelization, then collapse to 2D along Z

Two occupancy policies:
  * **binary** (`height_threshold=None`, default) — a cell is occupied if
    enough points project into it; everything else is free or unknown
    depending on ``unknown_outside_footprint``.
  * **tristate** (`height_threshold` set) — for each in-slab cell:
        - if ANY point has z > threshold → occupied
        - else if it has any points (all ≤ threshold) → free
        - empty cells → unknown
    This is the policy navigation stacks expect (threshold ≈ robot height)
    and matches the ``occupancy_grid_tuner.py`` workflow.

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


@dataclass
class GridParams:
    resolution: float = 0.05  # meters per cell
    z_min: float = 0.1  # ignore points below this (drop ground)
    z_max: float = 2.0  # ignore points above this (drop ceiling)
    method: str = "projection"  # "projection" | "voxel3d"
    # Binary projection mode only — ignored by voxel3d (which dedupes voxels
    # via np.unique before reaching the cell-count step).
    min_points_per_cell: int = 1
    unknown_outside_footprint: bool = False  # binary only: empty cells UNKNOWN vs FREE
    padding_cells: int = 5  # extra cells of margin around the data
    # Tristate policy: when set, switches free/occupied/unknown logic. Cells
    # with any point z > height_threshold become OCCUPIED; cells with points
    # but none above become FREE; cells with no points stay UNKNOWN.
    height_threshold: float | None = None
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
    return points[(points[:, 2] >= z_min) & (points[:, 2] <= z_max)]


def _make_grid_from_xy(xy: np.ndarray, params: GridParams) -> OccupancyGrid:
    if len(xy) == 0:
        raise ValueError("No points in the requested Z range — cannot build a grid.")

    res = params.resolution
    pad = params.padding_cells

    x_min = float(xy[:, 0].min()) - pad * res
    y_min = float(xy[:, 1].min()) - pad * res
    x_max = float(xy[:, 0].max()) + pad * res
    y_max = float(xy[:, 1].max()) + pad * res

    width = int(np.ceil((x_max - x_min) / res))
    height = int(np.ceil((y_max - y_min) / res))

    ix = np.floor((xy[:, 0] - x_min) / res).astype(np.int64)
    iy = np.floor((xy[:, 1] - y_min) / res).astype(np.int64)
    np.clip(ix, 0, width - 1, out=ix)
    np.clip(iy, 0, height - 1, out=iy)

    counts = np.zeros((height, width), dtype=np.int32)
    np.add.at(counts, (iy, ix), 1)

    background = UNKNOWN if params.unknown_outside_footprint else FREE
    image = np.full((height, width), background, dtype=np.uint8)
    image[counts >= params.min_points_per_cell] = OCC

    # Image's row 0 = top in map_server; world Y grows upward, so flip vertically.
    image = np.flipud(image)

    return OccupancyGrid(image=image, resolution=res, origin=(x_min, y_min, 0.0))


def _make_tristate_grid(xyz: np.ndarray, params: GridParams) -> OccupancyGrid:
    """Tristate (free / occupied / unknown) grid keyed off `height_threshold`."""
    if len(xyz) == 0:
        raise ValueError("No points in the requested Z range — cannot build a grid.")
    if params.height_threshold is None:
        raise ValueError("height_threshold must be set for tristate grids.")

    res = params.resolution
    pad = params.padding_cells
    thr = float(params.height_threshold)

    x_min = float(xyz[:, 0].min()) - pad * res
    y_min = float(xyz[:, 1].min()) - pad * res
    x_max = float(xyz[:, 0].max()) + pad * res
    y_max = float(xyz[:, 1].max()) + pad * res

    width = int(np.ceil((x_max - x_min) / res))
    height = int(np.ceil((y_max - y_min) / res))

    ix = np.floor((xyz[:, 0] - x_min) / res).astype(np.int64)
    iy = np.floor((xyz[:, 1] - y_min) / res).astype(np.int64)
    np.clip(ix, 0, width - 1, out=ix)
    np.clip(iy, 0, height - 1, out=iy)

    has_any = np.zeros((height, width), dtype=bool)
    has_occ = np.zeros((height, width), dtype=bool)
    has_any[iy, ix] = True
    above = xyz[:, 2] > thr
    has_occ[iy[above], ix[above]] = True

    image = np.full((height, width), UNKNOWN, dtype=np.uint8)
    image[has_any] = FREE
    image[has_occ] = OCC
    image = np.flipud(image)
    return OccupancyGrid(image=image, resolution=res, origin=(x_min, y_min, 0.0))


def from_projection(pcd: o3d.geometry.PointCloud, params: GridParams) -> OccupancyGrid:
    pts = np.asarray(pcd.points)
    pts = _filter_z(pts, params.z_min, params.z_max)
    if params.height_threshold is not None:
        return _make_tristate_grid(pts[:, :3], params)
    return _make_grid_from_xy(pts[:, :2], params)


def from_voxel3d(pcd: o3d.geometry.PointCloud, params: GridParams) -> OccupancyGrid:
    """Voxelize at `resolution`, then collapse occupied voxels to 2D."""
    res = params.resolution
    pts = np.asarray(pcd.points)
    pts = _filter_z(pts, params.z_min, params.z_max)
    if len(pts) == 0:
        raise ValueError("No points in the requested Z range — cannot build a grid.")

    # Discretize XYZ into integer voxel indices and dedupe in 3D.
    ix = np.floor(pts[:, 0] / res).astype(np.int64)
    iy = np.floor(pts[:, 1] / res).astype(np.int64)
    iz = np.floor(pts[:, 2] / res).astype(np.int64)
    voxels = np.unique(np.stack([ix, iy, iz], axis=1), axis=0)

    if params.height_threshold is not None:
        # Tristate: keep each voxel's Z so the threshold rule applies.
        xyz_centers = (voxels.astype(np.float64) + 0.5) * res
        return _make_tristate_grid(xyz_centers, params)

    # Binary: collapse to unique XY voxel centres.
    xy_voxels = np.unique(voxels[:, :2], axis=0)
    xy_centers = (xy_voxels + 0.5) * res
    return _make_grid_from_xy(xy_centers, params)


def build(pcd: o3d.geometry.PointCloud, params: GridParams) -> OccupancyGrid:
    if params.method == "projection":
        return from_projection(pcd, params)
    if params.method == "voxel3d":
        return from_voxel3d(pcd, params)
    raise ValueError(f"Unknown method: {params.method!r}")


def save(grid: OccupancyGrid, pgm_path: str, params: GridParams) -> tuple[Path, Path]:
    pgm_path = Path(pgm_path)
    if pgm_path.suffix.lower() != ".pgm":
        pgm_path = pgm_path.with_suffix(".pgm")
    yaml_path = pgm_path.with_suffix(".yaml")

    Image.fromarray(grid.image, mode="L").save(pgm_path)

    meta = {
        "image": pgm_path.name,
        "resolution": float(grid.resolution),
        "origin": [float(grid.origin[0]), float(grid.origin[1]), float(grid.origin[2])],
        "negate": int(params.negate),
        "occupied_thresh": float(params.occupied_threshold),
        "free_thresh": float(params.free_threshold),
    }
    with yaml_path.open("w") as f:
        yaml.safe_dump(meta, f, sort_keys=False)
    return pgm_path, yaml_path
