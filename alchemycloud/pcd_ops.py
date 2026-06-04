"""Point-cloud editing operations built on Open3D."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import open3d as o3d


def load_pcd(path: str) -> o3d.geometry.PointCloud:
    pcd = o3d.io.read_point_cloud(path)
    if len(pcd.points) == 0:
        raise ValueError(f"No points read from {path}")
    return pcd


def save_pcd(pcd: o3d.geometry.PointCloud, path: str, ascii_format: bool = False) -> None:
    o3d.io.write_point_cloud(path, pcd, write_ascii=ascii_format)


def voxel_downsample(pcd: o3d.geometry.PointCloud, voxel_size: float) -> o3d.geometry.PointCloud:
    if voxel_size <= 0:
        raise ValueError("voxel_size must be > 0")
    return pcd.voxel_down_sample(voxel_size=voxel_size)


def statistical_outlier_removal(
    pcd: o3d.geometry.PointCloud, nb_neighbors: int = 30, std_ratio: float = 1.5
) -> o3d.geometry.PointCloud:
    """Remove statistical outliers (matches PCL ``pcl_outlier_removal
    -method statistical``). Defaults follow the friend's working values
    (`-mean_k 30 -std_dev_mul 1.5`)."""
    filtered, _ = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    return filtered


def radius_outlier_removal(
    pcd: o3d.geometry.PointCloud, nb_points: int = 30, radius: float = 0.10
) -> o3d.geometry.PointCloud:
    """Remove points whose neighbourhood is too sparse (matches PCL
    ``pcl_outlier_removal -method radius``). A point survives if it has
    at least ``nb_points`` neighbours within ``radius`` metres."""
    if radius <= 0 or nb_points <= 0:
        raise ValueError("radius and nb_points must be positive")
    filtered, _ = pcd.remove_radius_outlier(nb_points=nb_points, radius=radius)
    return filtered


def z_clip(
    pcd: o3d.geometry.PointCloud, z_min: float, z_max: float
) -> o3d.geometry.PointCloud:
    if z_max <= z_min:
        raise ValueError("z_max must be > z_min")
    pts = np.asarray(pcd.points)
    mask = (pts[:, 2] >= z_min) & (pts[:, 2] <= z_max)
    out = pcd.select_by_index(np.where(mask)[0])
    return out


def remove_ground_ransac(
    pcd: o3d.geometry.PointCloud,
    distance_threshold: float = 0.05,
    ransac_n: int = 3,
    num_iterations: int = 1000,
) -> tuple[o3d.geometry.PointCloud, o3d.geometry.PointCloud, list[float]]:
    """Return (non_ground, ground, plane_model[a,b,c,d])."""
    plane_model, inliers = pcd.segment_plane(
        distance_threshold=distance_threshold,
        ransac_n=ransac_n,
        num_iterations=num_iterations,
    )
    ground = pcd.select_by_index(inliers)
    non_ground = pcd.select_by_index(inliers, invert=True)
    return non_ground, ground, list(plane_model)


def crop_with_box(
    pcd: o3d.geometry.PointCloud,
    min_bound: tuple[float, float, float],
    max_bound: tuple[float, float, float],
    keep_inside: bool = True,
) -> o3d.geometry.PointCloud:
    bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=min_bound, max_bound=max_bound)
    if keep_inside:
        return pcd.crop(bbox)
    pts = np.asarray(pcd.points)
    lo = np.asarray(min_bound)
    hi = np.asarray(max_bound)
    inside = np.all((pts >= lo) & (pts <= hi), axis=1)
    return pcd.select_by_index(np.where(~inside)[0])


def transform_to_marker_frame(
    pcd: o3d.geometry.PointCloud,
    marker_pos,
    marker_yaw: float,
) -> o3d.geometry.PointCloud:
    """Return a deep-copied cloud transformed so that a marker placed at
    `marker_pos` (m) with yaw `marker_yaw` (radians, about +Z) becomes the new
    origin with no rotation.

    p_local = R(-yaw) · (p_world - marker_pos)
    """
    import copy as _copy

    m = np.asarray(marker_pos, dtype=np.float64).reshape(3)
    c, s = float(np.cos(marker_yaw)), float(np.sin(marker_yaw))
    # R(-yaw) about Z:
    R = np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = -R @ m

    out = _copy.deepcopy(pcd)
    out.transform(T)
    return out


def level_ground(
    pcd: o3d.geometry.PointCloud,
    distance_threshold: float = 0.05,
    ransac_n: int = 3,
    num_iterations: int = 1000,
    translate_to_zero: bool = True,
    tilt_eps_deg: float = 0.05,
) -> tuple[o3d.geometry.PointCloud, list[float], float]:
    """Rotate (and optionally translate) the cloud so the dominant RANSAC
    plane becomes horizontal — its normal aligned with +Z. Returns
    ``(new_pcd, plane_before, tilt_degrees)``.

    - ``plane_before`` is the fitted plane equation ``[a, b, c, d]`` *before*
      the transform was applied (with the normal flipped so ``c >= 0``).
    - ``tilt_degrees`` is how far the plane normal was from +Z. If it is
      below ``tilt_eps_deg``, the cloud is returned unchanged — any
      "correcting" rotation at that scale is just float drift.
    """
    import copy as _copy
    from scipy.spatial.transform import Rotation

    plane_model, _ = pcd.segment_plane(
        distance_threshold=distance_threshold,
        ransac_n=ransac_n,
        num_iterations=num_iterations,
    )
    a, b, c, d = (float(x) for x in plane_model)
    n = np.array([a, b, c], dtype=np.float64)
    norm = float(np.linalg.norm(n))
    if norm < 1e-12:
        raise ValueError("RANSAC returned a degenerate plane (zero normal).")
    n = n / norm
    d = d / norm
    # Make the normal point "up".
    if n[2] < 0:
        n = -n
        d = -d
    plane_before = [float(n[0]), float(n[1]), float(n[2]), float(d)]

    cos_t = float(np.clip(n[2], -1.0, 1.0))
    tilt_deg = float(np.degrees(np.arccos(cos_t)))
    if tilt_deg < tilt_eps_deg:
        # Already level — no-op.
        return _copy.deepcopy(pcd), plane_before, tilt_deg

    rot, _rmsd = Rotation.align_vectors(
        np.array([[0.0, 0.0, 1.0]]),
        n.reshape(1, 3),
    )
    R = rot.as_matrix()

    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    if translate_to_zero:
        # After rotation the plane sits at z = -d; lift it to z = 0.
        T[:3, 3] = np.array([0.0, 0.0, d], dtype=np.float64)

    out = _copy.deepcopy(pcd)
    out.transform(T)
    return out, plane_before, tilt_deg


def display_downsample(
    pcd: o3d.geometry.PointCloud, max_points: int = 500_000
) -> o3d.geometry.PointCloud:
    """Return a voxel-downsampled clone targeting roughly `max_points` points
    for fast on-screen rendering. The full cloud is left untouched."""
    n = len(pcd.points)
    if n <= max_points:
        return pcd
    pts = np.asarray(pcd.points)
    extent = float(np.max(pts.max(axis=0) - pts.min(axis=0)))
    if extent <= 0:
        return pcd
    voxel = extent / np.cbrt(max_points) * 1.5
    out = pcd.voxel_down_sample(voxel_size=voxel)
    # If our heuristic over-shoots, tighten the voxel until we hit the budget.
    for _ in range(3):
        if len(out.points) <= max_points * 1.25:
            break
        voxel *= 1.3
        out = pcd.voxel_down_sample(voxel_size=voxel)
    return out


@dataclass
class PCDStats:
    num_points: int
    min_bound: np.ndarray
    max_bound: np.ndarray
    has_colors: bool

    def describe(self) -> str:
        extent = self.max_bound - self.min_bound
        return (
            f"{self.num_points:,} points | "
            f"X[{self.min_bound[0]:.2f},{self.max_bound[0]:.2f}] "
            f"Y[{self.min_bound[1]:.2f},{self.max_bound[1]:.2f}] "
            f"Z[{self.min_bound[2]:.2f},{self.max_bound[2]:.2f}] | "
            f"extent {extent[0]:.2f}x{extent[1]:.2f}x{extent[2]:.2f} m"
        )


_AXIS = {"x": 0, "y": 1, "z": 2}


def colorize_by_axis(
    pcd: o3d.geometry.PointCloud, axis: str = "z", cmap_name: str = "viridis"
) -> o3d.geometry.PointCloud:
    """Set per-point colors based on the chosen axis. Mutates `pcd` in place
    and also returns it for chaining."""
    axis = axis.lower()
    if axis not in _AXIS:
        raise ValueError(f"axis must be one of x/y/z, got {axis!r}")
    pts = np.asarray(pcd.points)
    if len(pts) == 0:
        return pcd
    vals = pts[:, _AXIS[axis]]
    v_min, v_max = float(vals.min()), float(vals.max())
    if v_max - v_min < 1e-9:
        normed = np.zeros_like(vals)
    else:
        normed = (vals - v_min) / (v_max - v_min)
    # matplotlib is pulled in by open3d, safe to import here.
    from matplotlib import colormaps

    rgba = colormaps[cmap_name](normed)
    pcd.colors = o3d.utility.Vector3dVector(rgba[:, :3])
    return pcd


def stats(pcd: o3d.geometry.PointCloud) -> PCDStats:
    pts = np.asarray(pcd.points)
    return PCDStats(
        num_points=len(pts),
        min_bound=pts.min(axis=0) if len(pts) else np.zeros(3),
        max_bound=pts.max(axis=0) if len(pts) else np.zeros(3),
        has_colors=pcd.has_colors(),
    )
