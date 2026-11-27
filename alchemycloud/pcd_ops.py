"""Point-cloud editing operations built on Open3D."""

from __future__ import annotations

import logging
import os
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
    """Read a point cloud, dropping non-finite points. See ``load_pcd_counted``."""
    return load_pcd_counted(path)[0]


def load_pcd_counted(path: str) -> tuple[o3d.geometry.PointCloud, int]:
    """Read a point cloud, dropping non-finite points. Returns ``(cloud,
    n_dropped)`` so a caller can report the clean-up without re-reading the
    file (``non_finite_count`` reads it twice more: seconds on a large scan).

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
    dropped = 0
    if declared is not None and declared > n:
        # Ambiguous on its own: non-finite rows we deliberately removed look
        # the same as a truncated body. Re-read raw to tell them apart.
        raw = len(o3d.io.read_point_cloud(path).points)
        if raw < declared:
            raise ValueError(
                f"{Path(path).name} is truncated: header declares {declared:,} points "
                f"but the file only contains {raw:,}."
            )
        dropped = declared - n
        log.info("%s: dropped %d non-finite point(s) of %d", path, dropped, declared)
    return pcd, dropped


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


# Opt-in: set to 1 to run GPU-capable operations on CUDA. Off by default
# because the GPU's floating-point rounding can classify a neighbour at
# *exactly* the radius differently from the CPU, so results would depend on
# the machine (seen only on clouds with rounded coordinates, never on scans).
GPU_ENV = "ALCHEMYCLOUD_GPU"
_GPU_OFF = frozenset({"", "0", "false", "no", "off"})
_cuda_available: bool | None = None


def cuda_available() -> bool:
    """Whether Open3D can use a CUDA GPU here (cached)."""
    global _cuda_available
    if _cuda_available is None:
        try:
            _cuda_available = bool(o3d.core.cuda.is_available())
        except Exception:  # CPU-only Open3D build, or no driver
            _cuda_available = False
    return _cuda_available


def cuda_device() -> o3d.core.Device | None:
    """The CUDA device to use, or None to stay on the CPU (the default:
    ``ALCHEMYCLOUD_GPU`` unset, empty, 0/false/no/off, or no usable GPU)."""
    if os.environ.get(GPU_ENV, "").strip().lower() in _GPU_OFF or not cuda_available():
        return None
    return o3d.core.Device("CUDA:0")


def _radius_keep_mask_tensor(points: np.ndarray, nb_points: int, radius: float, device) -> np.ndarray:
    """Radius-outlier keep mask from Open3D's tensor filter on `device`, made
    to match the legacy CPU filter: the tensor filter counts the query point
    among its own neighbours (hence nb_points + 1) and includes neighbours at
    exactly `radius`, which the legacy one excludes (hence the next float
    below `radius`)."""
    t = o3d.t.geometry.PointCloud(device)
    t.point.positions = o3d.core.Tensor(points, o3d.core.float64, device)
    _, mask = t.remove_radius_outliers(nb_points + 1, float(np.nextafter(radius, 0.0)))
    return mask.cpu().numpy().astype(bool)


def radius_outlier_removal(
    pcd: o3d.geometry.PointCloud, nb_points: int = 30, radius: float = 0.10
) -> o3d.geometry.PointCloud:
    """Remove points whose neighbourhood is too sparse (matches PCL
    ``pcl_outlier_removal -method radius``). A point survives if it has
    at least ``nb_points`` neighbours within ``radius`` metres.

    With ``ALCHEMYCLOUD_GPU=1`` and a CUDA GPU, the neighbour search runs on
    the GPU (~3x faster on a 1.8 M-point scan, same points kept on real scans;
    see GPU_ENV for the rounding caveat). Any GPU failure, e.g. running out of
    GPU memory for a large radius, falls back to the CPU filter."""
    if radius <= 0 or nb_points <= 0:
        raise ValueError("radius and nb_points must be positive")
    device = cuda_device()
    if device is not None and len(pcd.points):
        try:
            keep = _radius_keep_mask_tensor(np.asarray(pcd.points), nb_points, radius, device)
            return select_by_mask(pcd, keep)
        except Exception as exc:
            log.warning("GPU radius outlier removal failed (%s); using the CPU", exc)
        finally:
            try:  # hand cached GPU memory back to other applications
                o3d.core.cuda.release_cache()
            except Exception:
                pass
    filtered, _ = pcd.remove_radius_outlier(nb_points=nb_points, radius=radius)
    return filtered


def z_clip(pcd: o3d.geometry.PointCloud, z_min: float, z_max: float) -> o3d.geometry.PointCloud:
    if z_max <= z_min:
        raise ValueError("z_max must be > z_min")
    pts = np.asarray(pcd.points)
    mask = (pts[:, 2] >= z_min) & (pts[:, 2] <= z_max)
    return select_by_mask(pcd, mask)


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


def surface_tilt(
    pcd: o3d.geometry.PointCloud,
    percentile: float,
    band: float = 0.30,
) -> tuple[np.ndarray, float, int]:
    """Tilt of a horizontal surface selected by height percentile.

    ``percentile`` picks which surface: a low value (say 2) lands on the floor,
    a high one (98) on the ceiling. Returns ``(unit_normal, tilt_degrees,
    n_points)``.

    Useful because a floor and a ceiling that disagree cannot both be levelled
    by one rigid rotation, and knowing the disagreement up front tells you
    whether auto-levelling will help or just move the problem. On a real
    survey this measured floor 4.66 deg against ceiling 0.75 deg — a 4.13 deg
    inconsistency that no single rotation can remove.
    """
    pts = np.asarray(pcd.points)
    pts = pts[np.isfinite(pts).all(axis=1)]
    if len(pts) < 3:
        raise ValueError("Need at least 3 finite points to measure a surface.")

    def fit(sel: np.ndarray) -> tuple[np.ndarray, float]:
        """Total-least-squares plane; the smallest singular direction is the normal."""
        centre = sel.mean(axis=0)
        _u, _s, vt = np.linalg.svd(sel - centre, full_matrices=False)
        nrm = vt[2] / np.linalg.norm(vt[2])
        if nrm[2] < 0:
            nrm = -nrm
        return nrm, float(-nrm @ centre)

    # Seed on a horizontal slice. A *wide* one on purpose: a tilted surface
    # spans Z by itself, so a narrow percentile window cuts a thin stripe across
    # it and the plane fit on that stripe is degenerate — which is exactly the
    # case this function exists to measure.
    z = pts[:, 2]
    lo = np.percentile(z, max(percentile - 12.0, 0.0))
    hi = np.percentile(z, min(percentile + 12.0, 100.0))
    sel = pts[(z >= lo) & (z <= hi)]
    if len(sel) < 3:
        sel = pts

    n, d = fit(sel)
    # Then let the selection follow the fitted plane rather than a Z slice, so
    # the whole tilted surface is recovered. Converges in a couple of passes.
    for _ in range(4):
        keep = np.abs(pts @ n + d) <= band
        if keep.sum() < max(3, len(pts) // 1000):
            break
        n_new, d_new = fit(pts[keep])
        if abs(float(n_new @ n)) > 1.0 - 1e-9:
            n, d = n_new, d_new
            break
        n, d = n_new, d_new
        sel = pts[keep]

    tilt = float(np.degrees(np.arccos(float(np.clip(n[2], -1.0, 1.0)))))
    return n, tilt, int(np.abs(pts @ n + d).__le__(band).sum())


def apply_tilt(
    pcd: o3d.geometry.PointCloud,
    roll_deg: float,
    pitch_deg: float,
    about: str = "centroid",
) -> o3d.geometry.PointCloud:
    """Rotate the cloud by `roll_deg` about +X then `pitch_deg` about +Y.

    A manual trim for what RANSAC cannot settle. ``level_ground`` aligns the
    *dominant* plane and stops; when a floor and ceiling disagree, or when a
    0.2 deg residual still drags a costmap slab across the map (0.2 deg over a
    180 m site is 64 cm of height drift corner to corner), the remaining
    correction is a judgement call the operator has to make by eye.

    Rotation is about the cloud's centroid by default, not the world origin.
    This does not change the cloud's shape or its Z extent — a rigid rotation
    cannot — but it keeps the cloud where the camera is already looking instead
    of swinging it through space, which matters when the survey sits 150 m from
    the origin. Pass ``about="origin"`` for the raw rotation.
    """
    import copy as _copy

    if about not in {"centroid", "origin"}:
        raise ValueError(f"about must be 'centroid' or 'origin', got {about!r}")

    r, p = float(np.radians(roll_deg)), float(np.radians(pitch_deg))
    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p), np.sin(p)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=np.float64)
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=np.float64)
    R = Ry @ Rx

    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    if about == "centroid":
        pts = np.asarray(pcd.points)
        c = pts[np.isfinite(pts).all(axis=1)].mean(axis=0) if len(pts) else np.zeros(3)
        T[:3, 3] = c - R @ c

    out = _copy.deepcopy(pcd)
    if not np.allclose(T, np.eye(4)):
        out.transform(T)
    return out


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
    return select_by_indices(pcd, idx)


def select_by_indices(pcd: o3d.geometry.PointCloud, idx: np.ndarray) -> o3d.geometry.PointCloud:
    """Fast ``pcd.select_by_index(idx)`` for an integer index array.

    Open3D's ``select_by_index`` needs a Python ``list`` — converting a
    multi-million-element array to one and back costs ~0.7 s on a 1.8 M-point
    cloud. This gathers points, colours and normals with ``np.take`` instead.
    Values are copied bit-for-bit, so a selected point is exactly equal to its
    source point (the eraser's optimistic display edit relies on that).
    """
    idx = np.asarray(idx, dtype=np.int64)
    out = o3d.geometry.PointCloud()
    out.points = o3d.utility.Vector3dVector(np.take(np.asarray(pcd.points), idx, axis=0))
    if pcd.has_colors():
        out.colors = o3d.utility.Vector3dVector(np.take(np.asarray(pcd.colors), idx, axis=0))
    if pcd.has_normals():
        out.normals = o3d.utility.Vector3dVector(np.take(np.asarray(pcd.normals), idx, axis=0))
    return out


def select_by_mask(pcd: o3d.geometry.PointCloud, keep: np.ndarray) -> o3d.geometry.PointCloud:
    """Return a new cloud holding the points where boolean `keep` is True
    (points, colours and normals). Never mutates `pcd`."""
    keep = np.asarray(keep, dtype=bool)
    if len(keep) != len(pcd.points):
        raise ValueError(f"mask has {len(keep)} entries for {len(pcd.points)} points")
    return select_by_indices(pcd, np.flatnonzero(keep))


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
    # Contiguous copy is essential: Vector3dVector walks a strided view such as
    # `rgba[:, :3]` element by element (~0.5 s for 500k points) but block-copies
    # a contiguous (N, 3) float64 array (~8 ms).
    pcd.colors = o3d.utility.Vector3dVector(np.ascontiguousarray(rgba[:, :3], dtype=np.float64))
    return pcd


def stats(pcd: o3d.geometry.PointCloud) -> PCDStats:
    """Point count, bounds and colour presence. Bounds ignore non-finite
    coordinates, so one stray NaN cannot turn every reported bound into nan."""
    pts = np.asarray(pcd.points)
    if len(pts) == 0:
        return PCDStats(0, np.zeros(3), np.zeros(3), pcd.has_colors())
    # Runs on the main thread after every edit, so it must stay cheap: a
    # whole-array finiteness check plus per-column reductions. numpy's
    # axis-0 reduction over an (N, 3) array is ~6x slower than three
    # per-column reductions (~0.7 s vs ~0.1 s on 1.8 M points).
    if np.isfinite(pts).all():
        usable = pts
    else:
        keep = np.isfinite(pts).all(axis=1)
        usable = pts[keep] if keep.any() else np.zeros((1, 3))
    return PCDStats(
        num_points=len(pts),
        min_bound=np.array([usable[:, i].min() for i in range(3)]),
        max_bound=np.array([usable[:, i].max() for i in range(3)]),
        has_colors=pcd.has_colors(),
    )
