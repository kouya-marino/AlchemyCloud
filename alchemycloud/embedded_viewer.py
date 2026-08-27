"""In-window 3D point-cloud viewer (pyqtgraph.GLViewWidget).

The widget owns a single `GLScatterPlotItem`, applies a display-LOD policy for
very large clouds (full cloud is still what filters/export use), supports
color-by-axis (X/Y/Z, viridis), and implements a Qt-native screen-space
rectangle crop. When crop mode is on, mouse drags draw a dashed rectangle on
a child overlay widget; on release the viewer emits `cropRequested(rect)`
with the widget-pixel rectangle. The actual point-membership math is exposed
as module-level helpers so the main window can do the crop against the full
(not display-LOD) cloud.
"""

from __future__ import annotations

import copy

import numpy as np
import open3d as o3d
import pyqtgraph.opengl as gl
from PyQt5 import QtCore, QtGui, QtWidgets

from . import pcd_ops
from .widgets import ColorbarLegend

# ---------------------------------------------------------------------------
# Overlay widget — draws the crop rectangle on top of the GL view
# ---------------------------------------------------------------------------


class _CropOverlay(QtWidgets.QWidget):
    """Translucent overlay used for the crop rectangle, the eraser brush
    ring, and the eraser stroke trail."""

    def __init__(self, parent: QtWidgets.QWidget):
        super().__init__(parent)
        self.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents, True)
        self.setAttribute(QtCore.Qt.WA_NoSystemBackground, True)
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground, True)
        self._rect: QtCore.QRect | None = None
        self._brush_center: QtCore.QPoint | None = None
        self._brush_radius_px: int = 30
        self._brush_shape: str = "circle"  # "circle" | "square" | "rectangle"
        self._brush_aspect: float = 2.0  # rectangle only — width = aspect × height
        self._stroke: list[QtCore.QPoint] = []

    def set_rect(self, rect: QtCore.QRect | None) -> None:
        self._rect = rect
        self.update()

    def set_brush(
        self,
        center: QtCore.QPoint | None,
        radius_px: int,
        shape: str = "circle",
        aspect: float = 2.0,
    ) -> None:
        self._brush_center = center
        self._brush_radius_px = int(radius_px)
        self._brush_shape = shape if shape in {"circle", "square", "rectangle"} else "circle"
        self._brush_aspect = float(aspect) if aspect > 0 else 1.0
        self.update()

    def set_stroke(self, points: list[QtCore.QPoint]) -> None:
        self._stroke = list(points) if points else []
        self.update()

    def clear_stroke(self) -> None:
        self._stroke = []
        self._brush_center = None
        self.update()

    def paintEvent(self, _ev):
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing, True)
        if self._rect is not None and not self._rect.isEmpty():
            p.setPen(QtGui.QPen(QtGui.QColor(255, 200, 0), 1.5, QtCore.Qt.DashLine))
            p.setBrush(QtGui.QColor(255, 200, 0, 50))
            p.drawRect(self._rect)
        # Eraser stroke trail (translucent red marks along the dragged path)
        if self._stroke:
            p.setPen(QtCore.Qt.NoPen)
            p.setBrush(QtGui.QColor(255, 80, 80, 60))
            for pt in self._stroke:
                self._draw_brush_shape(p, pt)
        # Brush cursor outline (live)
        if self._brush_center is not None:
            p.setBrush(QtCore.Qt.NoBrush)
            p.setPen(QtGui.QPen(QtGui.QColor(255, 80, 80, 220), 1.8))
            self._draw_brush_shape(p, self._brush_center)
            p.setPen(QtGui.QPen(QtGui.QColor(0, 0, 0, 160), 1.0))
            self._draw_brush_shape(p, self._brush_center, inflate=1)
        p.end()

    def _draw_brush_shape(
        self,
        p: QtGui.QPainter,
        center: QtCore.QPoint,
        *,
        inflate: int = 0,
    ) -> None:
        r = self._brush_radius_px + inflate
        cx, cy = center.x(), center.y()
        if self._brush_shape == "circle":
            p.drawEllipse(center, r, r)
        elif self._brush_shape == "square":
            p.drawRect(QtCore.QRectF(cx - r, cy - r, 2 * r, 2 * r))
        else:  # rectangle
            w = r * self._brush_aspect
            p.drawRect(QtCore.QRectF(cx - w, cy - r, 2 * w, 2 * r))


# ---------------------------------------------------------------------------
# Embedded GLViewWidget
# ---------------------------------------------------------------------------


class EmbeddedViewer(gl.GLViewWidget):
    """In-window 3D viewer. Use `set_cloud(pcd)` to display a cloud and
    `set_color_axis('x'|'y'|'z'|'rgb')` to recolor — 'rgb' shows the cloud's
    own colours (tone-mapped via `set_rgb_gamma` / `set_rgb_autolevel`) rather
    than a colormap. Emits `cropRequested(QRect)` when the user finishes a
    drag in crop mode."""

    cropRequested = QtCore.pyqtSignal(QtCore.QRect)
    # x, y, z, yaw_degrees
    markerMoved = QtCore.pyqtSignal(float, float, float, float)
    # Eraser drag finished: emits (path_xy_list, brush_radius_px)
    eraserStrokeFinished = QtCore.pyqtSignal(object, int)

    DISPLAY_MAX = 500_000  # points beyond this get voxel-downsampled for display

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(400, 300)
        self.setBackgroundColor(QtGui.QColor(20, 22, 28))
        # Enable hover-tracking so eraser-brush updates without holding a
        # button. (Plain orbit / pan modes ignore the extra events.)
        self.setMouseTracking(True)

        self._point_size = 1.5  # rendered scatter point size (px); user-adjustable
        self._scatter = gl.GLScatterPlotItem(
            pos=np.zeros((0, 3), dtype=np.float32),
            color=np.zeros((0, 4), dtype=np.float32),
            size=self._point_size,
            pxMode=True,
            # pyqtgraph defaults GLScatterPlotItem to 'additive', which means
            # GL_DEPTH_TEST off and glBlendFunc(SRC_ALPHA, ONE): overlapping
            # points SUM instead of occluding. Dense regions blow past white
            # (measured ~7 % of lit pixels on a 178 k-point room scan, up to 29
            # points landing on one pixel) and nothing hides anything, so the
            # cloud reads as haze rather than surfaces. Points here are fully
            # opaque, so depth-test them properly.
            glOptions="opaque",
        )
        self.addItem(self._scatter)

        # Ground reference grid; size/position are updated whenever a cloud loads
        self._grid = gl.GLGridItem()
        self._grid.setSize(10, 10, 1)
        self._grid.setSpacing(1, 1, 1)
        self.addItem(self._grid)

        # Origin marker: a 1-point scatter (centre dot) + three coloured axis
        # lines (X red, Y green, Z blue). Size scales with the loaded cloud.
        # The marker carries a yaw rotation about Z; X/Y axes rotate with it.
        self._marker_pos = np.zeros(3, dtype=np.float64)
        self._marker_yaw = 0.0  # radians
        self._marker_size = 1.0
        self._marker_dot = gl.GLScatterPlotItem(
            pos=np.zeros((1, 3), dtype=np.float32),
            color=np.array([[1.0, 1.0, 1.0, 1.0]], dtype=np.float32),
            size=12,
            pxMode=True,
        )
        self._marker_axes = gl.GLLinePlotItem(
            pos=np.zeros((6, 3), dtype=np.float32),
            color=np.zeros((6, 4), dtype=np.float32),
            width=3,
            antialias=True,
            mode="lines",
        )
        self.addItem(self._marker_axes)
        self.addItem(self._marker_dot)
        self._refresh_marker_geometry()

        self._color_axis = "z"
        self._display_lod_enabled = True
        self._display_pcd: o3d.geometry.PointCloud | None = None
        # The display cloud's *own* RGB, captured before colorize_by_axis
        # overwrites it. None when the source cloud carries no colour. Without
        # this, switching RGB -> Z -> RGB would come back viridis, because
        # colorize_by_axis mutates `_display_pcd.colors` in place.
        # `_pending_src_rgb` is written by prepare_cloud (worker thread) and
        # promoted in apply_prepared_cloud (main thread), so the cached colours
        # can never belong to a different cloud than `_display_pcd`.
        self._source_rgb: np.ndarray | None = None
        self._pending_src_rgb: np.ndarray | None = None
        # RGB display tone mapping. Indoor colour scans are typically very
        # under-exposed -- a mean channel value around 0.5 is normal -- so raw
        # values render close to black. Auto-level stretches each channel to
        # its 1-99 % range; gamma below 1.0 then lifts the midtones.
        self._rgb_gamma = 0.62
        self._rgb_autolevel = True
        # Cached per-point arrays for the *current* display cloud + axis.
        # Slab bound changes can then just slice these without re-running
        # the matplotlib colormap on every spin tick.
        self._cached_pts: np.ndarray | None = None
        self._cached_rgba: np.ndarray | None = None
        # XY bbox of the cached cloud, inflated 5 % on each side. Used to
        # size the Z-Slab planes; doesn't change with slab bounds.
        self._cached_xy_bbox: tuple[float, float, float, float] | None = None
        self.setCameraPosition(distance=10, elevation=30, azimuth=-60)

        # Crop mode state
        self._crop_mode = False
        self._crop_origin: QtCore.QPoint | None = None
        self._crop_current: QtCore.QPoint | None = None

        # Z-Slab editor state
        self._slab_visible = False
        self._slab_low = 0.0
        self._slab_high = 1.0
        self._slab_top = gl.GLMeshItem(
            vertexes=np.zeros((4, 3), dtype=np.float32),
            faces=np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32),
            color=(1.0, 0.85, 0.2, 0.18),
            smooth=False,
            drawEdges=True,
            edgeColor=(1.0, 0.85, 0.2, 0.85),
            glOptions="translucent",
        )
        self._slab_bottom = gl.GLMeshItem(
            vertexes=np.zeros((4, 3), dtype=np.float32),
            faces=np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32),
            color=(1.0, 0.85, 0.2, 0.18),
            smooth=False,
            drawEdges=True,
            edgeColor=(1.0, 0.85, 0.2, 0.85),
            glOptions="translucent",
        )
        self.addItem(self._slab_bottom)
        self.addItem(self._slab_top)
        self._slab_top.setVisible(False)
        self._slab_bottom.setVisible(False)

        # Eraser tool state
        self._eraser_mode = False
        self._eraser_brush_px = 30
        self._eraser_brush_shape = "circle"  # "circle" | "square" | "rectangle"
        self._eraser_brush_aspect = 2.0  # only used by "rectangle"
        self._eraser_path: list[QtCore.QPoint] | None = None

        # Marker drag state
        self._marker_mode = False
        self._marker_drag_origin: QtCore.QPoint | None = None
        self._marker_drag_start_yaw: float = 0.0
        self._marker_drag_button: QtCore.Qt.MouseButton | None = None

        self._overlay = _CropOverlay(self)
        self._overlay.resize(self.size())
        self._overlay.raise_()

        self._legend = ColorbarLegend(self)
        self._position_legend()

    # -- public API --------------------------------------------------------

    def clear_cloud(self) -> None:
        self._display_pcd = None
        self._source_rgb = None
        self._pending_src_rgb = None
        self._cached_pts = None
        self._cached_rgba = None
        self._cached_xy_bbox = None
        self._scatter.setData(
            pos=np.zeros((0, 3), dtype=np.float32),
            color=np.zeros((0, 4), dtype=np.float32),
        )
        self._legend.hide_legend()
        self.update()

    def _update_legend_range(self) -> None:
        """Recompute the visible (vmin, vmax) along the current colour axis
        and push it to the legend. When Z-Slab is on, the range is clamped
        to in-slab points so the legend matches what's actually rendered.
        Reads from the cached float32 pts to avoid re-materialising
        Open3D's Vector3dVector each call."""
        if self._color_axis == "rgb":
            # The colourbar maps a scalar axis to a colormap; in RGB mode the
            # colour carries no scalar, so there is nothing to legend.
            self._legend.hide_legend()
            return
        if self._cached_pts is None:
            self._rebuild_color_cache()
        if self._cached_pts is None or len(self._cached_pts) == 0:
            self._legend.hide_legend()
            return
        axis_idx = {"x": 0, "y": 1, "z": 2}[self._color_axis]
        pts = self._cached_pts
        if self._slab_visible:
            slab_mask = self._compute_slab_mask()
            if slab_mask is not None and slab_mask.any():
                pts = pts[slab_mask]
        vals = pts[:, axis_idx]
        if vals.size == 0:
            self._legend.hide_legend()
            return
        self._legend.set_range(self._color_axis, float(vals.min()), float(vals.max()))
        self._legend.show_legend()
        self._position_legend()

    def is_display_lod_enabled(self) -> bool:
        return self._display_lod_enabled

    def set_display_lod_enabled(self, enabled: bool) -> None:
        """Enable / disable the display-LOD. Off => render every point of
        the full cloud (can be slow on multi-million-point clouds)."""
        self._display_lod_enabled = bool(enabled)

    def point_size(self) -> float:
        return self._point_size

    def set_point_size(self, px: float) -> None:
        """Set the rendered scatter point size (px). Size lives in its own
        VBO, so updating it alone is cheap and carries no pos/colour desync
        risk. Persists across reloads/filters via the size used in
        apply_prepared_cloud / _refresh_scatter_colors."""
        self._point_size = max(float(px), 0.1)
        if self._display_pcd is not None:
            self._scatter.setData(size=self._point_size)
            self.update()

    def _tone_map(self, cols: np.ndarray) -> np.ndarray:
        """Apply auto-level + gamma to source RGB for display.

        Raw indoor colour clouds are usually under-exposed and render close to
        black, so the defaults lift them. `auto_level=False, gamma=1.0` renders
        the file's colours completely unmodified.
        """
        out = np.clip(np.asarray(cols, dtype=np.float32), 0.0, 1.0)
        if self._rgb_autolevel and len(out) > 1:
            lo = np.percentile(out, 1.0, axis=0)
            hi = np.percentile(out, 99.0, axis=0)
            out = np.clip((out - lo) / np.maximum(hi - lo, 1e-6), 0.0, 1.0)
        gamma = float(self._rgb_gamma)
        if abs(gamma - 1.0) > 1e-6:
            out = out**gamma
        return out

    def _resolve_colors(self, display: o3d.geometry.PointCloud, src_rgb: np.ndarray | None, n: int) -> np.ndarray:
        """RGBA colour block for `n` points under the current colour mode.

        In "rgb" mode this is the cloud's own colour, tone-mapped; a cloud with
        no colour falls back to flat grey. In an axis mode it is whatever
        `colorize_by_axis` just wrote.
        """
        if self._color_axis == "rgb":
            if src_rgb is None or src_rgb.size == 0 or len(src_rgb) != n:
                return np.full((n, 3), 0.6, dtype=np.float32)
            return self._tone_map(src_rgb)
        cols = np.asarray(display.colors, dtype=np.float32)
        if cols.size and len(cols) == n:
            return cols
        return np.full((n, 3), 0.6, dtype=np.float32)

    def prepare_cloud(
        self, pcd: o3d.geometry.PointCloud
    ) -> tuple[o3d.geometry.PointCloud | None, np.ndarray, np.ndarray]:
        """Compute the display clone + RGBA arrays. Safe to call from a
        worker thread — touches no Qt or GL state."""
        if pcd is None or len(pcd.points) == 0:
            self._pending_src_rgb = None
            return None, np.zeros((0, 3), dtype=np.float32), np.zeros((0, 4), dtype=np.float32)

        if self._display_lod_enabled:
            display = pcd_ops.display_downsample(pcd, self.DISPLAY_MAX)
        else:
            display = pcd
        # display_downsample may return the input as-is for small clouds; we
        # need our own copy because colorize_by_axis mutates `pcd.colors`.
        if display is pcd:
            display = copy.deepcopy(display)

        # Snapshot the cloud's own RGB *before* colorize_by_axis overwrites it.
        # voxel_down_sample averages colours per voxel, so this stays meaningful
        # under display-LOD. Promoted to `_source_rgb` in apply_prepared_cloud.
        src = np.asarray(display.colors, dtype=np.float32)
        src = src.copy() if src.size else None
        self._pending_src_rgb = src

        if self._color_axis != "rgb":
            pcd_ops.colorize_by_axis(display, self._color_axis)

        pts = np.asarray(display.points, dtype=np.float32)
        rgba = np.empty((len(pts), 4), dtype=np.float32)
        rgba[:, :3] = self._resolve_colors(display, src, len(pts))
        rgba[:, 3] = 1.0
        return display, pts, rgba

    def apply_prepared_cloud(
        self,
        display: o3d.geometry.PointCloud | None,
        pts: np.ndarray,
        rgba: np.ndarray,
        reset_view: bool = False,
        reset_marker: bool = False,
    ) -> None:
        """Main-thread step: push the prepared arrays to the GL scatter item.

        `reset_marker` zeroes the marker pose. Pass it only when the cloud's
        frame genuinely changed — a file load, or applying the marker as the new
        origin. It used to be unconditional, which meant *every* operation
        (voxel, SOR, z-clip, crop, an eraser stroke, undo, even Reset Camera)
        silently threw away a marker the user had positioned, then reported
        "Marker is already at the origin — nothing to apply".
        """
        if display is None or pts.size == 0:
            self.clear_cloud()
            return
        self._display_pcd = display
        # Promote the source-RGB snapshot taken by prepare_cloud, so it always
        # belongs to the cloud we are about to show.
        self._source_rgb = self._pending_src_rgb
        self._pending_src_rgb = None
        # New cloud → invalidate the colour cache (rebuild on next access).
        self._cached_pts = None
        self._cached_rgba = None
        self._cached_xy_bbox = None
        # Explicit clear-then-set to force a clean VBO reallocation. Without
        # this, pyqtgraph 0.14 has been observed to leave the old (smaller)
        # colour VBO behind when the point count grows, so extra vertices
        # read past the buffer and end up drawn with the shader's default
        # constant colour (white). Clearing first guarantees both pos and
        # colour buffers are freshly allocated at the new size.
        self._scatter.setData(
            pos=np.zeros((0, 3), dtype=np.float32),
            color=np.zeros((0, 4), dtype=np.float32),
        )
        self._scatter.setData(pos=pts, color=rgba, size=self._point_size, pxMode=True)
        self._fit_grid(pts)
        # Rescaling the marker glyph to the new extent is always harmless;
        # zeroing its pose is not (see the docstring).
        self._fit_marker(pts)
        if reset_marker:
            self.reset_marker()
        else:
            self._refresh_marker_geometry()
        self._update_legend_range()
        # If Z-Slab editor was open, recompute its geometry/preview against
        # the new cloud (bounds are kept; main window may reset them).
        if self._slab_visible:
            self._refresh_slab_geometry()
            self._refresh_scatter_colors(slab_mask=self._compute_slab_mask())
        if reset_view:
            self._fit_camera(pts)
        self.update()

    def set_cloud(
        self,
        pcd: o3d.geometry.PointCloud,
        reset_view: bool = False,
        reset_marker: bool = False,
    ) -> None:
        """Synchronous convenience wrapper — fine for small clouds and tests.
        For large clouds prefer prepare_cloud() on a worker + apply_prepared_cloud()
        on the main thread (see MainWindow._apply_async)."""
        display, pts, rgba = self.prepare_cloud(pcd)
        self.apply_prepared_cloud(display, pts, rgba, reset_view=reset_view, reset_marker=reset_marker)

    def has_source_colors(self) -> bool:
        """True when the displayed cloud carries its own RGB (so "rgb" mode
        will show something other than flat grey)."""
        return self._source_rgb is not None and self._source_rgb.size > 0

    def rgb_gamma(self) -> float:
        return self._rgb_gamma

    def set_rgb_gamma(self, gamma: float) -> None:
        """Display gamma for "rgb" mode. Below 1.0 brightens, above darkens;
        1.0 leaves the file's colours untouched. No effect in axis modes."""
        self._rgb_gamma = float(np.clip(float(gamma), 0.05, 5.0))
        self._refresh_rgb()

    def rgb_autolevel(self) -> bool:
        return self._rgb_autolevel

    def set_rgb_autolevel(self, on: bool) -> None:
        """Per-channel 1-99 % contrast stretch before gamma, in "rgb" mode."""
        self._rgb_autolevel = bool(on)
        self._refresh_rgb()

    def _refresh_rgb(self) -> None:
        """Re-tone-map the current cloud in place. No-op outside "rgb" mode —
        geometry is untouched, only the colour VBO is re-uploaded."""
        if self._color_axis != "rgb" or self._display_pcd is None:
            return
        self._cached_rgba = None
        slab_mask = self._compute_slab_mask() if self._slab_visible else None
        self._refresh_scatter_colors(slab_mask=slab_mask)
        self.update()

    def set_color_axis(self, axis: str) -> None:
        axis = axis.lower()
        if axis not in {"x", "y", "z", "rgb"}:
            return
        self._color_axis = axis
        if self._display_pcd is None:
            return
        # New axis → invalidate the cache so the next refresh recomputes.
        self._cached_pts = None
        self._cached_rgba = None
        slab_mask = self._compute_slab_mask() if self._slab_visible else None
        self._refresh_scatter_colors(slab_mask=slab_mask)
        self._update_legend_range()
        self.update()

    # -- Z-Slab editor ----------------------------------------------------

    def set_zslab_visible(self, on: bool) -> None:
        self._slab_visible = bool(on)
        self._slab_top.setVisible(self._slab_visible)
        self._slab_bottom.setVisible(self._slab_visible)
        if not self._slab_visible:
            # Restore the full cloud display
            self._refresh_scatter_colors(slab_mask=None)
        else:
            self._refresh_slab_geometry()
            self._refresh_scatter_colors(slab_mask=self._compute_slab_mask())
        self._update_legend_range()
        self.update()

    def set_zslab_bounds(self, z_low: float, z_high: float) -> None:
        if z_high < z_low:
            z_low, z_high = z_high, z_low
        self._slab_low = float(z_low)
        self._slab_high = float(z_high)
        if self._slab_visible:
            self._refresh_slab_geometry()
            self._refresh_scatter_colors(slab_mask=self._compute_slab_mask())
            self._update_legend_range()
            self.update()

    def slab_bounds(self) -> tuple[float, float]:
        return self._slab_low, self._slab_high

    def _refresh_slab_geometry(self) -> None:
        if self._cached_pts is None:
            self._rebuild_color_cache()
        if self._cached_pts is None or len(self._cached_pts) == 0:
            return
        # Use the cached float32 pts and only recompute the XY bounding box
        # once per cloud — slab bounds don't change the cloud footprint.
        if not hasattr(self, "_cached_xy_bbox") or self._cached_xy_bbox is None:
            pts = self._cached_pts
            xmin = float(pts[:, 0].min())
            ymin = float(pts[:, 1].min())
            xmax = float(pts[:, 0].max())
            ymax = float(pts[:, 1].max())
            dx = max((xmax - xmin) * 0.05, 0.1)
            dy = max((ymax - ymin) * 0.05, 0.1)
            self._cached_xy_bbox = (xmin - dx, ymin - dy, xmax + dx, ymax + dy)
        xmin, ymin, xmax, ymax = self._cached_xy_bbox
        for plane, z in ((self._slab_top, self._slab_high), (self._slab_bottom, self._slab_low)):
            verts = np.array(
                [
                    [xmin, ymin, z],
                    [xmax, ymin, z],
                    [xmax, ymax, z],
                    [xmin, ymax, z],
                ],
                dtype=np.float32,
            )
            plane.setMeshData(
                vertexes=verts,
                faces=np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32),
            )

    def _compute_slab_mask(self) -> np.ndarray | None:
        if self._cached_pts is None:
            self._rebuild_color_cache()
        if self._cached_pts is None:
            return None
        z = self._cached_pts[:, 2]
        return (z >= self._slab_low) & (z <= self._slab_high)

    def _rebuild_color_cache(self) -> None:
        """Recompute and cache (pts, rgba) for the current display cloud and
        colour axis. Called when the cloud or axis changes; *not* on slab
        bound changes (those reuse the cache)."""
        if self._display_pcd is None or len(self._display_pcd.points) == 0:
            self._cached_pts = None
            self._cached_rgba = None
            return
        if self._color_axis != "rgb":
            pcd_ops.colorize_by_axis(self._display_pcd, self._color_axis)
        pts = np.asarray(self._display_pcd.points, dtype=np.float32)
        # Size from `pts`, not from the colour array: a colourless cloud gives
        # a zero-length colour array, which would otherwise yield an empty
        # rgba buffer for a non-empty cloud.
        rgba = np.empty((len(pts), 4), dtype=np.float32)
        rgba[:, :3] = self._resolve_colors(self._display_pcd, self._source_rgb, len(pts))
        rgba[:, 3] = 1.0
        self._cached_pts = pts
        self._cached_rgba = rgba

    def _refresh_scatter_colors(self, slab_mask: np.ndarray | None) -> None:
        """Re-upload the scatter. When `slab_mask` is provided, only in-slab
        points are uploaded — out-of-slab points are hidden from the canvas
        (they stay in `current_pcd` and can still be edited indirectly).
        Uses the cached colour buffer — slab bound changes are O(N) numpy
        slicing, no matplotlib recompute."""
        if self._cached_pts is None or self._cached_rgba is None:
            self._rebuild_color_cache()
        if self._cached_pts is None:
            return
        if slab_mask is None:
            pts = self._cached_pts
            rgba = self._cached_rgba
        else:
            pts = self._cached_pts[slab_mask]
            rgba = self._cached_rgba[slab_mask]
        # Always re-issue both pos AND color (see note in apply_prepared_cloud)
        self._scatter.setData(
            pos=np.zeros((0, 3), dtype=np.float32),
            color=np.zeros((0, 4), dtype=np.float32),
        )
        self._scatter.setData(pos=pts, color=rgba, size=self._point_size, pxMode=True)

    # -- Eraser tool ------------------------------------------------------

    def set_eraser_mode(self, on: bool) -> None:
        self._eraser_mode = bool(on)
        self._eraser_path = None
        self._overlay.set_rect(None)
        if on:
            # Make eraser mutually exclusive with crop & marker modes.
            self._crop_mode = False
            self._crop_origin = None
            self._crop_current = None
            self._marker_mode = False
            self._cancel_marker_drag()
            # Show the brush ring at the cursor immediately so the user
            # sees their tool — without this they'd have to move the mouse
            # before the overlay paints.
            cursor_pos = self.mapFromGlobal(QtGui.QCursor.pos())
            if not self.rect().contains(cursor_pos):
                cursor_pos = self.rect().center()
            self._overlay.set_brush(
                cursor_pos,
                self._eraser_brush_px,
                self._eraser_brush_shape,
                self._eraser_brush_aspect,
            )
        else:
            self._overlay.set_brush(
                None,
                self._eraser_brush_px,
                self._eraser_brush_shape,
                self._eraser_brush_aspect,
            )
        self._update_cursor()

    def set_eraser_brush(self, px: int) -> None:
        self._eraser_brush_px = max(int(px), 2)
        self._refresh_brush_cursor()

    def set_eraser_shape(self, shape: str) -> None:
        if shape not in {"circle", "square", "rectangle"}:
            return
        self._eraser_brush_shape = shape
        self._refresh_brush_cursor()

    def set_eraser_aspect(self, aspect: float) -> None:
        self._eraser_brush_aspect = max(float(aspect), 0.1)
        self._refresh_brush_cursor()

    def brush_shape(self) -> str:
        return self._eraser_brush_shape

    def brush_aspect(self) -> float:
        return self._eraser_brush_aspect

    def _refresh_brush_cursor(self) -> None:
        """Re-paint the brush ring with the current shape / aspect."""
        if not self._eraser_mode:
            return
        if self._eraser_path:
            center = self._eraser_path[-1]
        else:
            center = self.mapFromGlobal(QtGui.QCursor.pos())
            if not self.rect().contains(center):
                center = self.rect().center()
        self._overlay.set_brush(
            center,
            self._eraser_brush_px,
            self._eraser_brush_shape,
            self._eraser_brush_aspect,
        )

    def set_crop_mode(self, on: bool) -> None:
        self._crop_mode = bool(on)
        self._crop_origin = None
        self._crop_current = None
        self._overlay.set_rect(None)
        if on:
            self._marker_mode = False
            self._cancel_marker_drag()
            self._eraser_mode = False
            self._eraser_path = None
            self._overlay.set_brush(None, self._eraser_brush_px)
        self._update_cursor()

    def is_crop_mode(self) -> bool:
        return self._crop_mode

    # -- marker -----------------------------------------------------------

    def set_marker_mode(self, on: bool) -> None:
        self._marker_mode = bool(on)
        self._cancel_marker_drag()
        if on:
            self._crop_mode = False
            self._crop_origin = None
            self._crop_current = None
            self._overlay.set_rect(None)
            self._eraser_mode = False
            self._eraser_path = None
            self._overlay.set_brush(None, self._eraser_brush_px)
        self._update_cursor()

    def is_marker_mode(self) -> bool:
        return self._marker_mode

    def marker_position(self) -> tuple[float, float, float]:
        return tuple(map(float, self._marker_pos))

    def marker_yaw_degrees(self) -> float:
        return float(np.degrees(self._marker_yaw))

    def set_marker_position(self, x: float, y: float, z: float | None = None) -> None:
        self._marker_pos[0] = float(x)
        self._marker_pos[1] = float(y)
        if z is not None:
            self._marker_pos[2] = float(z)
        self._refresh_marker_geometry()
        self._emit_marker_moved()

    def set_marker_yaw(self, yaw_radians: float) -> None:
        # Normalize to (-π, π]. Pure modulo gives -π for an input of +π,
        # which flips the displayed sign confusingly — clamp back to +π.
        y = (float(yaw_radians) + np.pi) % (2 * np.pi) - np.pi
        if y == -np.pi:
            y = np.pi
        self._marker_yaw = y
        self._refresh_marker_geometry()
        self._emit_marker_moved()

    def reset_marker(self) -> None:
        self._marker_pos[:] = 0.0
        self._marker_yaw = 0.0
        self._refresh_marker_geometry()
        self._emit_marker_moved()

    def _emit_marker_moved(self) -> None:
        self.markerMoved.emit(
            float(self._marker_pos[0]),
            float(self._marker_pos[1]),
            float(self._marker_pos[2]),
            float(np.degrees(self._marker_yaw)),
        )

    def _refresh_marker_geometry(self) -> None:
        p = self._marker_pos.astype(np.float32)
        s = float(self._marker_size)
        c, si = np.cos(self._marker_yaw), np.sin(self._marker_yaw)
        # X/Y axes rotate with yaw; Z stays vertical.
        xv = np.array([c, si, 0.0], dtype=np.float32) * s
        yv = np.array([-si, c, 0.0], dtype=np.float32) * s
        zv = np.array([0.0, 0.0, 1.0], dtype=np.float32) * s
        pts = np.array(
            [
                p,
                p + xv,
                p,
                p + yv,
                p,
                p + zv,
            ],
            dtype=np.float32,
        )
        cols = np.array(
            [
                [1.0, 0.35, 0.35, 1.0],
                [1.0, 0.35, 0.35, 1.0],
                [0.35, 1.0, 0.35, 1.0],
                [0.35, 1.0, 0.35, 1.0],
                [0.45, 0.55, 1.0, 1.0],
                [0.45, 0.55, 1.0, 1.0],
            ],
            dtype=np.float32,
        )
        self._marker_axes.setData(pos=pts, color=cols, width=3, mode="lines")
        self._marker_dot.setData(pos=p[None, :])

    def _fit_marker(self, pts: np.ndarray) -> None:
        if pts.size == 0:
            self._marker_size = 1.0
        else:
            extent = float(np.linalg.norm(pts.max(axis=0) - pts.min(axis=0)))
            self._marker_size = max(extent * 0.05, 0.2)
        self._refresh_marker_geometry()

    def _cancel_marker_drag(self) -> None:
        self._marker_drag_origin = None
        self._marker_drag_start_yaw = 0.0
        self._marker_drag_button = None

    def screen_to_world_xy(self, mouse_pos: QtCore.QPoint, z_plane: float) -> tuple[float, float] | None:
        """Cast a ray from the mouse pixel through the scene, intersect with
        the horizontal plane z = `z_plane`, and return the (x, y) hit in world
        coordinates. Returns None if the ray is parallel to the plane or the
        widget has no size yet."""
        w, h = self.width(), self.height()
        if w <= 0 or h <= 0:
            return None
        try:
            inv = np.linalg.inv(self.mvp_matrix())
        except np.linalg.LinAlgError:
            return None
        nx = (float(mouse_pos.x()) / w) * 2.0 - 1.0
        ny = 1.0 - (float(mouse_pos.y()) / h) * 2.0
        near = inv @ np.array([nx, ny, -1.0, 1.0])
        far = inv @ np.array([nx, ny, 1.0, 1.0])
        if abs(near[3]) < 1e-12 or abs(far[3]) < 1e-12:
            return None
        near_w = near[:3] / near[3]
        far_w = far[:3] / far[3]
        direction = far_w - near_w
        if abs(direction[2]) < 1e-9:
            return None
        t = (z_plane - near_w[2]) / direction[2]
        # t < 0 means the plane is *behind* the near clip plane: the click was
        # above the horizon, and the ray only meets z = z_plane if extended
        # backwards through the camera. Returning that intersection puts the
        # marker at a mirrored point behind the viewer, which reads as the
        # marker teleporting to a nonsensical spot.
        if t < 0.0:
            return None
        hit = near_w + t * direction
        return float(hit[0]), float(hit[1])

    def _update_cursor(self) -> None:
        if self._crop_mode:
            self.setCursor(QtCore.Qt.CrossCursor)
        elif self._marker_mode:
            self.setCursor(QtCore.Qt.SizeAllCursor)
        elif self._eraser_mode:
            self.setCursor(QtCore.Qt.BlankCursor)  # brush ring replaces system cursor
        else:
            self.setCursor(QtCore.Qt.ArrowCursor)

    def mvp_matrix(self) -> np.ndarray:
        """Combined projection*view as a 4x4 NumPy array (row-major)."""
        viewport = self.getViewport()
        # In pyqtgraph 0.14 projectionMatrix takes (region, viewport).
        # We render the whole viewport, so region == viewport.
        m = self.projectionMatrix(viewport, viewport) * self.viewMatrix()
        # QMatrix4x4.data() returns column-major; transpose to row-major.
        return np.array(m.data(), dtype=np.float64).reshape(4, 4).T

    # -- camera / grid helpers --------------------------------------------

    def _fit_camera(self, pts: np.ndarray) -> None:
        if pts.size == 0:
            return
        lo, hi = pts.min(axis=0), pts.max(axis=0)
        center = (lo + hi) / 2.0
        extent = float(np.linalg.norm(hi - lo))
        self.opts["center"] = QtGui.QVector3D(float(center[0]), float(center[1]), float(center[2]))
        self.setCameraPosition(distance=max(extent * 0.8, 1.0), elevation=30, azimuth=-60)

    def _fit_grid(self, pts: np.ndarray) -> None:
        if pts.size == 0:
            return
        lo, hi = pts.min(axis=0), pts.max(axis=0)
        cx, cy = float((lo[0] + hi[0]) / 2), float((lo[1] + hi[1]) / 2)
        size_x = max(float(hi[0] - lo[0]), 1.0)
        size_y = max(float(hi[1] - lo[1]), 1.0)
        spacing = max(round(max(size_x, size_y) / 20, 2), 0.1)
        self._grid.resetTransform()
        self._grid.setSize(size_x * 1.2, size_y * 1.2, 1)
        self._grid.setSpacing(spacing, spacing, spacing)
        self._grid.translate(cx, cy, float(lo[2]))

    # -- mouse / overlay --------------------------------------------------

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        self._overlay.resize(self.size())
        self._position_legend()

    def _position_legend(self) -> None:
        margin = 10
        legend_w = self._legend.width()
        legend_h = max(min(self.height() - 2 * margin, 280), 140)
        self._legend.setGeometry(self.width() - legend_w - margin, margin, legend_w, legend_h)
        self._legend.raise_()

    def mousePressEvent(self, ev):
        if self._eraser_mode:
            if ev.button() == QtCore.Qt.LeftButton:
                self._eraser_path = [ev.pos()]
                self._overlay.set_stroke(self._eraser_path)
                self._overlay.set_brush(
                    ev.pos(),
                    self._eraser_brush_px,
                    self._eraser_brush_shape,
                    self._eraser_brush_aspect,
                )
                ev.accept()
                return
            # Any other button falls through to the base class so the camera can
            # still be orbited and panned while the eraser is armed. Previously
            # every button was swallowed, so the only way to re-aim the view was
            # to toggle the eraser off and back on. Crop mode already behaves
            # this way (it gates on LeftButton).
            super().mousePressEvent(ev)
            return
        if self._crop_mode and ev.button() == QtCore.Qt.LeftButton:
            self._crop_origin = ev.pos()
            self._crop_current = ev.pos()
            self._overlay.set_rect(QtCore.QRect(ev.pos(), ev.pos()))
            ev.accept()
            return
        if self._marker_mode and ev.button() in (QtCore.Qt.LeftButton, QtCore.Qt.RightButton):
            if self._marker_drag_origin is None:
                self._marker_drag_origin = ev.pos()
                self._marker_drag_start_yaw = self._marker_yaw
                self._marker_drag_button = ev.button()
                # On LMB press, also snap the marker to where you clicked so
                # the user doesn't need to "find" it first.
                if ev.button() == QtCore.Qt.LeftButton:
                    hit = self.screen_to_world_xy(ev.pos(), float(self._marker_pos[2]))
                    if hit is not None:
                        self.set_marker_position(hit[0], hit[1])
            ev.accept()
            return
        super().mousePressEvent(ev)

    def mouseMoveEvent(self, ev):
        if self._eraser_mode:
            self._overlay.set_brush(
                ev.pos(),
                self._eraser_brush_px,
                self._eraser_brush_shape,
                self._eraser_brush_aspect,
            )
            if self._eraser_path is not None:
                last = self._eraser_path[-1]
                gap = (ev.pos() - last).manhattanLength()
                if gap > 3:
                    # Interpolate along the segment. Qt only delivers a move
                    # event per repaint, so a fast drag arrives as widely spaced
                    # samples; stamping a disc at each one erased a dotted line
                    # with untouched gaps between. Step at half the brush radius
                    # so consecutive discs always overlap.
                    step = max(self._eraser_brush_px // 2, 2)
                    n_steps = int(gap // step)
                    for i in range(1, n_steps):
                        f = i / n_steps
                        self._eraser_path.append(
                            QtCore.QPoint(
                                int(round(last.x() + (ev.pos().x() - last.x()) * f)),
                                int(round(last.y() + (ev.pos().y() - last.y()) * f)),
                            )
                        )
                    self._eraser_path.append(ev.pos())
                    self._overlay.set_stroke(self._eraser_path)
            ev.accept()
            return
        if self._crop_mode and self._crop_origin is not None:
            self._crop_current = ev.pos()
            self._overlay.set_rect(QtCore.QRect(self._crop_origin, ev.pos()).normalized())
            ev.accept()
            return
        if self._marker_mode and self._marker_drag_origin is not None:
            if self._marker_drag_button == QtCore.Qt.LeftButton:
                # XY-plane move: project cursor to the marker's current Z plane.
                hit = self.screen_to_world_xy(ev.pos(), float(self._marker_pos[2]))
                if hit is not None:
                    self.set_marker_position(hit[0], hit[1])
            else:  # RightButton → yaw
                dx = ev.pos().x() - self._marker_drag_origin.x()
                # 0.5° per pixel — 720 px drag spans a full turn.
                new_yaw = self._marker_drag_start_yaw + np.radians(dx * 0.5)
                self.set_marker_yaw(new_yaw)
            ev.accept()
            return
        super().mouseMoveEvent(ev)

    def mouseReleaseEvent(self, ev):
        if self._eraser_mode and ev.button() == QtCore.Qt.LeftButton and self._eraser_path is not None:
            path = self._eraser_path
            self._eraser_path = None
            self._overlay.clear_stroke()
            ev.accept()
            if len(path) >= 1:
                self.eraserStrokeFinished.emit(path, self._eraser_brush_px)
            return
        if self._crop_mode and ev.button() == QtCore.Qt.LeftButton and self._crop_origin is not None:
            rect = QtCore.QRect(self._crop_origin, ev.pos()).normalized()
            self._crop_origin = None
            self._crop_current = None
            self._overlay.set_rect(None)
            ev.accept()
            if rect.width() > 3 and rect.height() > 3:
                self.cropRequested.emit(rect)
            return
        if self._marker_mode and self._marker_drag_origin is not None and ev.button() == self._marker_drag_button:
            self._cancel_marker_drag()
            ev.accept()
            return
        super().mouseReleaseEvent(ev)


# ---------------------------------------------------------------------------
# Projection helpers (used by main_window for crop math against the full cloud)
# ---------------------------------------------------------------------------


def project_points(points_xyz: np.ndarray, mvp: np.ndarray, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    """Project Nx3 world points to widget pixels.

    Returns (Nx2 pixel coords, mask of points in front of the camera)."""
    n = points_xyz.shape[0]
    pts4 = np.empty((n, 4), dtype=np.float64)
    pts4[:, :3] = points_xyz
    pts4[:, 3] = 1.0
    clip = pts4 @ mvp.T
    w = clip[:, 3]
    in_front = w > 1e-9
    w_safe = np.where(in_front, w, 1.0)
    ndc = clip[:, :3] / w_safe[:, None]
    px = (ndc[:, 0] + 1.0) * (width / 2.0)
    py = (1.0 - ndc[:, 1]) * (height / 2.0)
    return np.column_stack([px, py]), in_front


def mask_points_in_rect(
    points_xyz: np.ndarray,
    mvp: np.ndarray,
    width: int,
    height: int,
    rect: QtCore.QRect,
) -> np.ndarray:
    """Return a boolean mask of which points project inside `rect` and are in
    front of the camera."""
    px_py, in_front = project_points(points_xyz, mvp, width, height)
    x0, y0 = float(rect.left()), float(rect.top())
    x1, y1 = float(rect.right()), float(rect.bottom())
    return in_front & (px_py[:, 0] >= x0) & (px_py[:, 0] <= x1) & (px_py[:, 1] >= y0) & (px_py[:, 1] <= y1)


def mask_points_in_brush(
    points_xyz: np.ndarray,
    mvp: np.ndarray,
    width: int,
    height: int,
    canvas_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (mask, in_front). `canvas_mask` is a (H, W) bool image marking
    which pixels are "painted" by the eraser brush. `mask` is True for each
    cloud point whose 2D projection falls onto a True pixel of `canvas_mask`."""
    px_py, in_front = project_points(points_xyz, mvp, width, height)
    if not in_front.any():
        return np.zeros(len(points_xyz), dtype=bool), in_front
    px = np.round(px_py[:, 0]).astype(np.int64)
    py = np.round(px_py[:, 1]).astype(np.int64)
    valid = in_front & (px >= 0) & (px < width) & (py >= 0) & (py < height)
    out = np.zeros(len(points_xyz), dtype=bool)
    idx = np.where(valid)[0]
    out[idx] = canvas_mask[py[idx], px[idx]]
    return out, in_front
