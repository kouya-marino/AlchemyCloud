"""Point-cloud editing operations built on Open3D."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import open3d as o3d

log = logging.getLogger(__name__)

# Text header of a PCD/PLY is ASCII even when the payload is binary, so the
# declared point count can be read without parsing the body.
_PCD_POINTS = re.compile(rb"^POINTS\s+(\d+)\s*$", re.M)
_PLY_VERTEX = re.compile(rb"^element\s+vertex\s+(\d+)\s*$", re.M)


def _declared_point_count(path: str) -> int | None:
    """Point count from the file's own header, or None if unavailable."""
    try:
        with open(path, "rb") as f:
            head = f.read(8192)
    except OSError:
        return None
    for pattern in (_PCD_POINTS, _PLY_VERTEX):
        m = pattern.search(head)
        if m:
            return int(m.group(1))
    return None


def load_pcd(path: str) -> o3d.geometry.PointCloud:
    """Read a point cloud, dropping non-finite points.

    Raw LiDAR routinely contains ``nan nan nan`` rows for invalid returns and
    occasional ``inf`` spikes. Left in, a single one poisons every ``min``/
    ``max`` downstream: the cloud renders black, the status bar reads ``nan``,
    and occupancy export dies with "cannot convert float NaN to integer" or
    tries to allocate a grid of astronomical size. Open3D can strip them at
    read time, so do that here rather than defending in every kernel.

    Raises ``ValueError`` when the file yields no usable points, and when the
    body holds fewer points than the header declares (a truncated file — Open3D
    reports success and returns the short cloud).
    """
    pcd = o3d.io.read_point_cloud(path, remove_nan_points=True, remove_infinite_points=True)
    n = len(pcd.points)
    if n == 0:
        raise ValueError(f"No points read from {path}")

    declared = _declared_point_count(path)
    if declared is not None:
        finite = int(np.isfinite(np.asarray(pcd.points)).all(axis=1).sum())
        dropped = declared - n
        if dropped > 0 and finite == n:
            # Ambiguous on its own: non-finite rows we deliberately removed look
            # the same as a truncated body. Re-read raw to tell them apart.
            raw = o3d.io.read_point_cloud(path)
            if len(raw.points) < declared:
                raise ValueError(
                    f"{Path(path).name} is truncated: header declares {declared:,} points "
                    f"but the file only contains {len(raw.points):,}."
                )
            log.info("%s: dropped %d non-finite point(s) of %d", path, dropped, declared)
    return pcd


def non_finite_count(path: str) -> int:
    """How many points ``load_pcd`` would drop from `path` as non-finite.

    Useful for telling the user their file was cleaned, without making
    ``load_pcd`` return a tuple.
    """
    declared = _declared_point_count(path)
    if declared is None:
        return 0
    kept = len(o3d.io.read_point_cloud(path, remove_nan_points=True, remove_infinite_points=True).points)
    raw = len(o3d.io.read_point_cloud(path).points)
    return max(min(raw, declared) - kept, 0)


def save_pcd(pcd: o3d.geometry.PointCloud, path: str, ascii_format: bool = False) -> None:
    """Write a cloud, raising if the write did not happen.

    ``o3d.io.write_point_cloud`` returns ``False`` and logs to stderr rather
    than raising — an unwritable directory, a missing parent, or an extension
    Open3D cannot infer a format from all fail silently. Swallowing that return
    value means the UI reports "Saved" for a file that was never created.
    """
    target = Path(path)
    if target.suffix == "":
        raise ValueError(
            f"Cannot infer a point-cloud format from {target.name!r} — give the file a .pcd, .ply or .xyz extension."
        )
    if not target.parent.exists():
        raise FileNotFoundError(f"Directory does not exist: {target.parent}")
    if not o3d.io.write_point_cloud(path, pcd, write_ascii=ascii_format):
        raise OSError(
            f"Open3D could not write {path} — check the path is writable and the "
            f"extension ({target.suffix}) is a supported point-cloud format."
        )


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


def z_clip(pcd: o3d.geometry.PointCloud, z_min: float, z_max: float) -> o3d.geometry.PointCloud:
    if z_max <= z_min:
        raise ValueError("z_max must be > z_min")
    pts = np.asarray(pcd.points)
    mask = (pts[:, 2] >= z_min) & (pts[:, 2] <= z_max)
    out = pcd.select_by_index(np.where(mask)[0])
    return out


def _normalised_up(plane_model) -> tuple[np.ndarray, float]:
    """Unit normal (forced to point +Z) and matching offset for `plane_model`."""
    a, b, c, d = (float(x) for x in plane_model)
    n = np.array([a, b, c], dtype=np.float64)
    norm = float(np.linalg.norm(n))
    if norm < 1e-12:
        raise ValueError("RANSAC returned a degenerate plane (zero normal).")
    n, d = n / norm, d / norm
    if n[2] < 0:
        n, d = -n, -d
    return n, d


def fit_ground_plane(
    pcd: o3d.geometry.PointCloud,
    distance_threshold: float = 0.05,
    ransac_n: int = 3,
    num_iterations: int = 1000,
    max_tilt_deg: float = 30.0,
    floor_band: float = 0.30,
    max_attempts: int = 5,
) -> tuple[np.ndarray, float]:
    """Fit the dominant **near-horizontal** plane. Returns (unit_normal, d).

    ``segment_plane`` alone returns whichever plane has the most inliers, and in
    an indoor scan a long flat wall very often beats the floor — on a 24 x 36 x
    1.8 m room scan the winning plane had normal ~[0.99, 0.11, 0.06] and 383 k
    inliers against the floor's 210 k. Aligning that to +Z stands the room on
    its side.

    Two guards, together:
      * candidates are restricted to the lowest ``floor_band`` fraction of the
        Z range, which is where a floor lives;
      * any fit whose normal is more than ``max_tilt_deg`` from vertical is
        rejected, its inliers dropped, and the fit retried.

    Raises ``ValueError`` if no near-horizontal plane is found, rather than
    silently applying a nonsensical rotation.
    """
    if distance_threshold <= 0:
        raise ValueError("distance_threshold must be > 0")
    pts = np.asarray(pcd.points)
    if len(pts) < ransac_n:
        raise ValueError(f"Need at least {ransac_n} points to fit a plane.")

    z = pts[:, 2]
    z_lo, z_hi = float(z.min()), float(z.max())
    band = z_lo + max(float(floor_band), 0.0) * (z_hi - z_lo)
    idx = np.where(z <= max(band, z_lo + distance_threshold))[0]
    # Fall back to the whole cloud if the band is too thin to fit anything.
    working = pcd.select_by_index(idx.tolist()) if len(idx) >= max(10 * ransac_n, 100) else pcd

    best_tilt = None
    for _ in range(max(int(max_attempts), 1)):
        if len(working.points) < max(ransac_n, 3):
            break
        plane_model, inliers = working.segment_plane(
            distance_threshold=distance_threshold,
            ransac_n=ransac_n,
            num_iterations=num_iterations,
        )
        n, d = _normalised_up(plane_model)
        tilt = float(np.degrees(np.arccos(float(np.clip(n[2], -1.0, 1.0)))))
        if tilt <= max_tilt_deg:
            return n, d
        best_tilt = tilt if best_tilt is None else min(best_tilt, tilt)
        # Not a floor — drop it and look again in what is left.
        working = working.select_by_index(inliers, invert=True)

    got = f"closest was {best_tilt:.1f}°" if best_tilt is not None else "no plane could be fitted"
    raise ValueError(
        f"No near-horizontal plane found within {max_tilt_deg:.0f}° of vertical ({got}). "
        "The dominant structure looks like a wall or ceiling rather than a floor — "
        "crop to the floor area first, or raise max_tilt_deg if the scan really is that steep."
    )


def remove_ground_ransac(
    pcd: o3d.geometry.PointCloud,
    distance_threshold: float = 0.05,
    ransac_n: int = 3,
    num_iterations: int = 1000,
    max_tilt_deg: float = 30.0,
    floor_band: float = 0.30,
) -> tuple[o3d.geometry.PointCloud, o3d.geometry.PointCloud, list[float]]:
    """Return (non_ground, ground, plane_model[a,b,c,d]).

    Only a near-horizontal plane is accepted as "ground" — see
    ``fit_ground_plane``. Inliers are then selected against the **full** cloud
    by point-to-plane distance, so the band restriction used for fitting does
    not limit what gets removed.
    """
    n, d = fit_ground_plane(
        pcd,
        distance_threshold=distance_threshold,
        ransac_n=ransac_n,
        num_iterations=num_iterations,
        max_tilt_deg=max_tilt_deg,
        floor_band=floor_band,
    )
    pts = np.asarray(pcd.points)
    on_plane = np.abs(pts @ n + d) <= distance_threshold
    ground_idx = np.where(on_plane)[0]
    ground = pcd.select_by_index(ground_idx.tolist())
    non_ground = pcd.select_by_index(ground_idx.tolist(), invert=True)
    return non_ground, ground, [float(n[0]), float(n[1]), float(n[2]), float(d)]


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
    max_tilt_deg: float = 30.0,
    floor_band: float = 0.30,
) -> tuple[o3d.geometry.PointCloud, list[float], float]:
    """Rotate (and optionally translate) the cloud so the dominant *floor*
    plane becomes horizontal — its normal aligned with +Z. Returns
    ``(new_pcd, plane_before, tilt_degrees)``.

    - ``plane_before`` is the fitted plane equation ``[a, b, c, d]`` *before*
      the transform was applied (with the normal flipped so ``c >= 0``).
    - ``tilt_degrees`` is how far the plane normal was from +Z. Below
      ``tilt_eps_deg`` no rotation is applied — any "correcting" rotation at
      that scale is just float drift — but ``translate_to_zero`` still holds.

    Only a near-horizontal plane is accepted; see ``fit_ground_plane``. A scan
    whose dominant plane is a wall raises ``ValueError`` instead of being
    rotated onto its side.
    """
    import copy as _copy

    from scipy.spatial.transform import Rotation

    n, d = fit_ground_plane(
        pcd,
        distance_threshold=distance_threshold,
        ransac_n=ransac_n,
        num_iterations=num_iterations,
        max_tilt_deg=max_tilt_deg,
        floor_band=floor_band,
    )
    plane_before = [float(n[0]), float(n[1]), float(n[2]), float(d)]

    cos_t = float(np.clip(n[2], -1.0, 1.0))
    tilt_deg = float(np.degrees(np.arccos(cos_t)))

    T = np.eye(4, dtype=np.float64)
    if tilt_deg >= tilt_eps_deg:
        rot, _rmsd = Rotation.align_vectors(
            np.array([[0.0, 0.0, 1.0]]),
            n.reshape(1, 3),
        )
        T[:3, :3] = rot.as_matrix()
    if translate_to_zero:
        # After rotation the plane sits at z = -d; lift it to z = 0. With no
        # rotation (already level) the same offset still puts it at z = 0.
        T[:3, 3] = np.array([0.0, 0.0, d], dtype=np.float64)

    out = _copy.deepcopy(pcd)
    if not np.allclose(T, np.eye(4)):
        out.transform(T)
    return out, plane_before, tilt_deg


def display_downsample(
    pcd: o3d.geometry.PointCloud, max_points: int = 500_000, seed: int = 0
) -> o3d.geometry.PointCloud:
    """Return a clone of at most `max_points` points for fast rendering.

    Uses uniform random subsampling. The previous voxel heuristic sized the
    voxel as ``extent / cbrt(max_points)``, which assumes points fill a cube;
    a LiDAR floor scan is a *slab*, so occupancy grows with area, not volume,
    and the voxel came out one to two orders of magnitude too coarse. On a
    2.2 M-point 24 x 36 x 1.8 m room scan that yielded **1,580** points — 0.32 %
    of a 500 k budget — and the correction loop could only decimate further,
    never recover.

    Random sampling hits the budget exactly, is O(n), keeps real source points
    rather than voxel centroids, and needs no assumption about cloud shape.
    Returns the input object itself when it is already within budget, so
    callers that intend to mutate the result must copy it (see
    ``EmbeddedViewer.prepare_cloud``).
    """
    n = len(pcd.points)
    if max_points <= 0:
        raise ValueError("max_points must be > 0")
    if n <= max_points:
        return pcd
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=int(max_points), replace=False)
    idx.sort()  # keep source order so the view stays stable between frames
    return pcd.select_by_index(idx.tolist())


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
    # nan-safe: a single non-finite coordinate would otherwise make v_min/v_max
    # NaN, normed all-NaN, and every point render as the colormap's "bad"
    # colour (black). load_pcd strips non-finite points, but a cloud can also
    # arrive from a caller that did not go through it.
    finite = np.isfinite(vals)
    if not finite.any():
        normed = np.zeros(len(vals), dtype=np.float64)
    else:
        v_min = float(vals[finite].min())
        v_max = float(vals[finite].max())
        if v_max - v_min < 1e-9:
            normed = np.zeros(len(vals), dtype=np.float64)
        else:
            normed = (np.nan_to_num(vals, nan=v_min, posinf=v_max, neginf=v_min) - v_min) / (v_max - v_min)
    # matplotlib is pulled in by open3d, safe to import here.
    from matplotlib import colormaps

    rgba = colormaps[cmap_name](np.clip(normed, 0.0, 1.0))
    pcd.colors = o3d.utility.Vector3dVector(rgba[:, :3])
    return pcd


def stats(pcd: o3d.geometry.PointCloud) -> PCDStats:
    """Point count, bounds and colour presence. Bounds ignore non-finite
    coordinates, so one stray NaN cannot turn every reported bound into nan."""
    pts = np.asarray(pcd.points)
    if len(pts) == 0:
        return PCDStats(0, np.zeros(3), np.zeros(3), pcd.has_colors())
    keep = np.isfinite(pts).all(axis=1)
    usable = pts[keep] if keep.any() else np.zeros((1, 3))
    return PCDStats(
        num_points=len(pts),
        min_bound=usable.min(axis=0),
        max_bound=usable.max(axis=0),
        has_colors=pcd.has_colors(),
    )
