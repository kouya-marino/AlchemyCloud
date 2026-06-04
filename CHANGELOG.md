# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project follows [Semantic Versioning](https://semver.org/). Version
numbers and groupings mirror the curated overview in
[version_data.md](version_data.md).

## [Unreleased]

_Nothing yet._

## [1.2.0] — 2026-06-02

_Theme: Precision & hardening._

### Added
- **Adjustable point size** — a **Point** spin box on the Edit toolbar
  (0.5–15.0 px, default 1.5) sets the rendered scatter point size live.
  Bigger points make sparse clouds far easier to see. The size persists
  across reloads / filters / axis changes / slab toggles. Backed by
  `EmbeddedViewer.set_point_size()` / `point_size()`, which update the
  scatter's own size VBO (no pos/colour re-upload).
- **Typed marker coordinates** — the Marker toolbar gains live **X**, **Y**,
  and **θ°** spin boxes. Typing a value moves the marker instantly; X/Y are
  clamped to the loaded cloud's XY bounds (the "scope", expanded to always
  include the origin), yaw wraps at ±180°. The boxes are two-way: dragging
  the marker on the canvas or hitting **Reset Marker** mirrors back into
  them. Ranges re-clamp on every cloud change (load / filter / crop /
  level-ground / apply-marker / undo / redo). All in `main_window.py` — no
  viewer API changes (reuses `set_marker_position` / `set_marker_yaw` /
  `markerMoved`).
- **Pytest test suite** in `tests/` — tests across four files
  (`test_pcd_ops.py`, `test_occupancy.py`, `test_viewer_helpers.py`,
  `test_gui_smoke.py`). `conftest.py` sets `QT_QPA_PLATFORM=offscreen`
  before PyQt import and exposes reusable fixtures (`qapp`, `cube_cloud`,
  `cluster_with_outliers`, `floor_with_chair`, `tilted_cloud`,
  `busy_wait`). Coverage spans every public op in `pcd_ops`, both
  occupancy policies (binary + tristate) over both methods (projection +
  voxel3d), PGM/YAML round-trip, the projection helpers in
  `embedded_viewer`, and GUI smoke (`MainWindow` toolbars, load → filter →
  undo → redo, `OccupancyPreviewDialog` initial render, brush-shape combo
  mirror). Run with `pytest tests/`. Field guide at
  [`tests/README.md`](tests/README.md).
- **`requirements-dev.txt`** — `pytest>=7.0`. Production deps stay in
  `requirements.txt`.

### Changed
- `matplotlib.cm.get_cmap` → `matplotlib.colormaps[name]` in
  `pcd_ops.colorize_by_axis` and `widgets.ColorbarLegend`. The old API
  is deprecated in matplotlib 3.7+; the new API is forward-compatible
  with matplotlib 4.x.
- `occupancy.GridParams.min_points_per_cell` documented as
  binary-projection-only (silently ignored by `voxel3d`).
- `MainWindow._apply_async` docstring spells out the FLAT (not nested)
  tuple convention expected from `work_fn`.

### Fixed
- **Crash when closing the window during a long worker run.**
  `MainWindow.closeEvent` now disconnects the in-flight worker's
  `success` / `error` / `finished` signals before the super call, so
  Qt won't deliver queued events into a destroyed window. The worker
  thread itself runs to completion in the background; we just stop
  listening.
- **`OccupancyPreviewDialog.Save…` could write a stale grid** if the
  user clicked Save within the 120 ms debounce window after a parameter
  change. `_on_save` now flushes the pending debounce + rebuild
  (`_on_save_when_ready` re-arms via `QTimer.singleShot` until `_busy`
  clears) before invoking `_do_save`.
- **`action_reset_camera` (`Ctrl+R`) blocked the main thread** by calling
  the synchronous `viewer.set_cloud` path. Now routed through
  `_run_async` so the loading spinner shows during the re-prepare on
  multi-million-point clouds.
- **Slab spinbox drags were laggy** — each tick re-ran
  `matplotlib.colormap` over the entire display cloud plus a fresh GL
  upload. Two fixes: (1) `EmbeddedViewer` caches the per-point
  `(pts, rgba)` buffer for the current cloud + axis and invalidates on
  cloud/axis change; slab updates become O(N) numpy slicing.
  (2) `_on_zslab_bounds_changed` is now debounced through an 80 ms
  QTimer. Median slab update drops from ~33 ms → ~26 ms on a 500 k-point
  cloud and the spin path fires at ~12 Hz max.
- **`set_marker_yaw(+π)` returned −180° from `marker_yaw_degrees()`**.
  Normalisation now clamps `-π → +π` so the label stays positive when
  the user drags just past 180°.
- **Eraser brush ring was invisible until the first drag.** `EmbeddedViewer`
  now calls `setMouseTracking(True)` so hover events fire without a held
  button, and `set_eraser_mode(True)` pre-positions the brush at the
  current cursor location so the ring appears the instant the user
  toggles Eraser on.
- **White points after LOD → Full toggle.** `pyqtgraph 0.14`'s
  `GLScatterPlotItem` reuses its colour VBO across `setData` calls; if
  the position buffer grew while the colour buffer didn't reallocate
  cleanly, the extra vertices read past the buffer and fell back to the
  shader's constant white default. `apply_prepared_cloud` now issues an
  explicit `setData(pos=empty, color=empty)` before the real upload, and
  `set_color_axis` re-uploads both `pos` *and* `color` so the two VBOs
  stay strictly in sync.
- **Misleading `px_py` name** in `_on_eraser_stroke` renamed to
  `in_brush` (the value is a brush-hit mask, not pixel coords).
- **Dead code dropped**: unused `outline_only` kwarg on
  `_CropOverlay._draw_brush_shape`; unused `_a_display_switch_widget`
  ivar in `MainWindow`.

## [1.1.0] — 2026-05-27

_Theme: Costmap cleanup wave._

### Added
- **Level Ground** action on the Edit toolbar — fits a plane via RANSAC,
  computes the rotation that aligns the plane's normal with **+Z**, and
  (by default) translates the cloud so the plane sits at `z = 0`. Status
  bar reports the corrected tilt, e.g. `Levelled 10.00°  (plane: …)`.
  Idempotent: a follow-up call on a now-flat cloud is a no-op (tilt
  below 0.05°). Backed by **`pcd_ops.level_ground(...)`**, which builds
  the SE(3) matrix in NumPy and applies it via Open3D's
  `PointCloud.transform` on a deep copy. Uses
  `scipy.spatial.transform.Rotation.align_vectors` so the identity / 180°
  edge cases are stable.
- **Radius outlier removal** (`Outlier (Radius)…` on the Edit toolbar) —
  matches PCL's `pcl_outlier_removal -method radius`. Default
  `nb_points = 30`, `radius = 0.10 m` (user-editable). Backed by
  **`pcd_ops.radius_outlier_removal(pcd, nb_points, radius)`** wrapping
  Open3D's `remove_radius_outlier`. Pairs with the existing
  `Outlier (SOR)…` (renamed from `Outlier…`; defaults bumped to
  `Neighbors = 30, Std ratio = 1.5` to match a PCL invocation).
- **Tristate occupancy grids** — `occupancy.GridParams` gains
  `height_threshold: float | None = None`. When set, the build switches
  to a three-state policy: a cell is **occupied** if any in-slab point
  has `z > threshold`, **free** if it has in-slab points but none above,
  and **unknown** otherwise. Default behaviour (no threshold) is
  unchanged — binary occupancy. Implemented for both `from_projection`
  and `from_voxel3d`.
- **Live `OccupancyPreviewDialog`** (`occupancy_preview.py`) — non-modal
  dialog with controls on the left and a live PGM preview on the right.
  Sliders/spin boxes for method, resolution, Z min/max, threshold,
  min-points/cell, padding, and a "Tristate" toggle. Each parameter
  change schedules a debounced (120 ms) rebuild on a `QThreadPool`
  worker so the UI stays responsive. **Save…** writes PGM + YAML via
  `occupancy.save`. `MainWindow.action_export_map` now opens this dialog
  instead of the old modal `_ParamDialog`.
- **Eraser brush shape dropdown** — Tools toolbar gains a `Shape` combo
  (`Circle` / `Square` / `Rectangle`) and an `Aspect` spin (only used for
  Rectangle; default 2.0). `Brush px` is the brush half-extent for all
  shapes — `Circle` = radius, `Square` = half-side, `Rectangle` =
  half-height (width = half-height × aspect). The live cursor ring, the
  drag-trail dots, and the PIL-rasterised selection mask in
  `_on_eraser_stroke` all honour the shape uniformly. New API on
  `EmbeddedViewer`: `set_eraser_shape(str)`, `set_eraser_aspect(float)`,
  `brush_shape()`, `brush_aspect()`.
- **File → Save as PLY…** (`Ctrl+Shift+S`) — one-click PCD → PLY
  conversion. Writes `current_pcd` to the chosen `.ply` path using
  Open3D's format auto-detection. Convenient when downstream tools
  (FAST-LIO prior-map ingestion, MeshLab, CloudCompare) want PLY rather
  than PCD. Goes through the standard worker thread / loading spinner.

## [1.0.0] — 2026-05-22

_Theme: First stable editing toolkit. The app is now a complete
interactive single-window PCD editor._

### Added
- **Z-Slab scope** — new **Tools** toolbar with a `Z-Slab (scope)` toggle
  and low / high spin boxes. Two translucent yellow planes are drawn at
  the z_low / z_high heights. While the scope is on, **only in-slab
  points are rendered on the canvas** (out-of-slab points are hidden
  from view but still preserved in `current_pcd`); the colorbar legend
  clamps to the visible range. **Z-Slab does not remove points by
  itself** — it's a scope filter: **Crop** and **Eraser** only affect
  in-slab points; out-of-slab points are always preserved.
- **Eraser tool** — toggleable brush on the Tools toolbar. Drag with the
  left mouse button to paint over points; on release every cloud point
  whose 2D projection falls inside the painted region is removed.
  Adjustable brush size in pixels (4–200). If **Z-Slab** is also on, the
  eraser only removes in-slab points. Brush ring + faint red trail are
  drawn on the canvas overlay; the system cursor is hidden in eraser
  mode so the ring doubles as the cursor.
- **Redo** — `Edit menu → Redo` (`Ctrl+Y`) and a toolbar button next to
  Undo. Maintains a `redo_stack` (max 10) populated by `action_undo`; any
  new edit clears it. `action_redo` returns the current state to the
  history stack so a redo can itself be undone.
- **`mask_points_in_brush(points, mvp, W, H, canvas_mask)`** in
  `embedded_viewer.py` — projects cloud points to pixels and looks them
  up in a 2D PIL-rasterised brush mask — O(N) and works for
  multi-million-point clouds.
- **`EmbeddedViewer.set_zslab_visible / set_zslab_bounds / slab_bounds`** +
  `set_eraser_mode / set_eraser_brush` + `eraserStrokeFinished(path, radius)`
  signal.

## [0.4.0] — 2026-05-20

_Theme: Marker & responsive pipeline._

### Added
- **Origin marker (X/Y/Z gizmo)** — three coloured axis lines (red X,
  green Y, blue Z) plus a centre dot. Auto-sized to ~5 % of the cloud
  extent. Resets to `(0, 0, 0)` with yaw 0 on every file load.
- **Marker yaw** — marker carries a yaw rotation about Z. Position label
  includes the angle: `Marker: (x, y, z)  θ=±NN.N°`.
- **Move-Marker toolbar** (`MarkerToolbar`) with a **Move Marker** toggle,
  **Reset Marker** button, **Apply Marker as Origin** button, and a live
  position + yaw label.
- **"Apply Marker as Origin"** action — bakes the marker pose into the
  cloud (`pcd_ops.transform_to_marker_frame`): every point is rigidly
  transformed so the marker becomes the new origin with yaw 0. Saving
  thereafter writes the new frame. Undoable.
- **Loading overlay** (`loading_overlay.LoadingOverlay`) — rotating-arc
  spinner shown during slow ops (load/save, voxel/SOR/Z-clip/RANSAC,
  crop, undo/reset, occupancy export, display-LOD toggle, apply-marker).
- **Async pipeline** in `MainWindow` — `_run_async` / `_apply_async`
  helpers route slow work to a `QThreadPool` worker (`worker.Worker`,
  `worker.run_async`); the result is applied through the history/state
  pipeline on the main thread. Toolbars are disabled while busy.
- **`EmbeddedViewer.prepare_cloud` / `apply_prepared_cloud`** — split of
  the synchronous `set_cloud` so the slow part can run off-main-thread.
- **`EmbeddedViewer.screen_to_world_xy`** — ray-casts a cursor pixel onto
  a horizontal plane in world coordinates (used for the XY marker drag).
- **Colorbar legend (RViz-style)** — a vertical viridis gradient with
  min / mid / max tick labels and an axis label (`X (m)` / `Y (m)` /
  `Z (m)`) overlaid on the top-right of the embedded viewer. Updates
  whenever the cloud or the colour axis changes. Hides when no cloud is
  loaded. Lives in `widgets.ColorbarLegend`, driven by
  `EmbeddedViewer._update_legend_range()`.

### Changed
- **Marker drag semantics** — **LMB-drag** moves the marker on the XY
  plane (cursor ray-cast onto the marker's current Z plane); **RMB-drag
  horizontal** changes the yaw (0.5 °/px, 720 px = full turn). The old
  axis-locked behaviour (LMB → ΔX, RMB → ΔY) is gone.

### Fixed
- **`worker.Worker.setAutoDelete(False)`** — without this, `QThreadPool`
  could free the C++ `QRunnable` (and the `WorkerSignals` child) before
  queued cross-thread `success`/`finished` emissions reached the main
  thread, causing `RuntimeError: WorkerSignals has been deleted` and
  leaving the loading overlay stuck. The Python reference is dropped
  explicitly in `_end_busy`.

## [0.3.0] — 2026-05-15

_Theme: Embedded in-window viewer._

### Added
- **Embedded 3D viewer** — the point cloud now renders inside the main
  window's central widget (Paint/Photoshop-style). No more pop-out window.
- **`embedded_viewer.py`** — `EmbeddedViewer(GLViewWidget)` built on
  pyqtgraph + PyOpenGL. Holds a `GLScatterPlotItem` and emits
  `cropRequested(rect)` / `markerMoved(x, y, z, yaw°)`.
- **Display-LOD** — clouds larger than 500 k points are voxel-downsampled
  for the canvas only. Filters and occupancy export still run on the full
  cloud. (`pcd_ops.display_downsample`.)
- **Display-LOD toggle** on the Edit toolbar — a modern animated switch
  labelled `Display: LOD ⟷ Full`. Off (default) = voxel-downsample any
  cloud over 500 k points for fast rendering. On = render every point.
  Switching triggers a re-prepare on a worker thread with the loading
  overlay. Helper widget lives in **`widgets.ToggleSwitch`**.
- **Qt-native screen-space crop** — checkable **Crop** action on the Edit
  toolbar and a **Keep inside / Remove inside** mode combo. Drag a
  rectangle directly on the canvas; crop math projects the *full* cloud
  through the current MVP matrix so accuracy is independent of the LOD.
- **Ground-reference grid** — a `GLGridItem` is auto-sized and placed
  under each loaded cloud for orientation.
- **View → Reset camera** (`Ctrl+R`).

### Changed
- Stats and file path moved from the central area to the permanent
  status-bar widgets, freeing the central widget for the 3D canvas.
- `pcd_ops.colorize_by_axis` is now applied to the display cloud only —
  the full `current_pcd` is never tinted, so saving preserves the
  original colours.
- `mvp_matrix()` uses pyqtgraph 0.14's `(region, viewport)` signature for
  `projectionMatrix`.

### Removed
- **`viewer.py`** (background-thread Open3D viewer + `ViewerController`)
  is gone; the embedded widget is on the Qt main thread.
- **`pcd_ops.interactive_crop`** — the Open3D `VisualizerWithEditing`
  pop-out is no longer used. The Qt-native crop replaces it.

### Dependencies
- Added `pyqtgraph>=0.13` and `PyOpenGL>=3.1` to `requirements.txt`.

## [0.2.0] — 2026-05-15

_Theme: Live 3D viewer & colour._

### Added
- **Auto-launch 3D viewer** on file open — opening a PCD (via menu or CLI
  argument) immediately pops up the Open3D visualizer.
- **`viewer.py`** — `ViewerController` runs the Open3D `Visualizer` on a
  background thread; Qt communicates with it through a thread-safe command
  queue. Camera pose is preserved across cloud swaps
  (`reset_bounding_box=False` + restored pinhole camera parameters).
- **Live updates after every filter** — applying any edit pushes the new
  cloud into the live viewer without snapping the camera.
- **Color-by-axis** — viridis colormap over X / Y / Z. Default **Z**.
  Selector lives on the Edit toolbar.
- **`pcd_ops.colorize_by_axis(pcd, axis)`** — colorizes a cloud in place
  using matplotlib's viridis cmap (matplotlib is pulled in transitively by
  Open3D).
- **View menu → Re-open 3D viewer** — relaunch the viewer if the user closed
  it manually.

### Changed
- **Toolbars instead of button panels.** Two dockable toolbars:
  - **Edit** — Voxel, Outlier, Z-Clip, Ground (RANSAC), Crop, Undo, Reset,
    Color selector.
  - **Occupancy** — Generate Map (PGM + YAML).
  Both can be dragged, docked, or torn off.
- **`action_crop`** now stops the live viewer before opening the Open3D
  editing window (avoids GLFW double-window glitches), then reopens with
  the cropped cloud.
- **`app.py`** simplified — CLI-loaded files now go through the same
  `MainWindow.load_path()` code path that auto-launches the viewer.
- **`closeEvent`** cleanly stops the viewer thread.

## [0.1.0] — 2026-05-15

_Theme: Core engine & basic GUI._

### Added
- Initial project scaffolding.
- **`pcd_env`** pyenv virtualenv (Python 3.10.6) created and pinned via
  `.python-version`.
- **`requirements.txt`** — open3d, numpy, PyQt5, PyYAML, Pillow.
- **`pcd_ops.py`** — point-cloud operations:
  - `load_pcd` / `save_pcd`
  - `voxel_downsample`
  - `statistical_outlier_removal`
  - `z_clip`
  - `remove_ground_ransac`
  - `crop_with_box` (programmatic AABB crop)
  - `interactive_crop` (Open3D editing window)
  - `stats` / `PCDStats`
- **`occupancy.py`** — 2D occupancy grid generation in ROS `map_server`
  format (PGM + YAML):
  - `from_projection` — top-down 2D projection within `[z_min, z_max]`.
  - `from_voxel3d` — full 3D voxelization, then collapse along Z.
  - `build(pcd, params)` dispatcher.
  - `save(grid, path, params)` — writes PGM (occupied=0, free=254,
    unknown=205) and YAML (image / resolution / origin / negate /
    occupied_thresh / free_thresh).
- **`main_window.py`** — PyQt5 main window with File/Edit menus, filter
  button panels, parameter dialogs, and occupancy-grid export.
- **`app.py`** — entry point; optional CLI argument loads a PCD on startup.
- **`README.md`** — installation, UI walkthrough, parameter reference,
  troubleshooting, programmatic-use snippet.

[Unreleased]: https://github.com/kouya-marino/AlchemyCloud/compare/v1.2.0...HEAD
[1.2.0]: https://github.com/kouya-marino/AlchemyCloud/releases/tag/v1.2.0
[1.1.0]: https://github.com/kouya-marino/AlchemyCloud/releases/tag/v1.1.0
[1.0.0]: https://github.com/kouya-marino/AlchemyCloud/releases/tag/v1.0.0
[0.4.0]: https://github.com/kouya-marino/AlchemyCloud/releases/tag/v0.4.0
[0.3.0]: https://github.com/kouya-marino/AlchemyCloud/releases/tag/v0.3.0
[0.2.0]: https://github.com/kouya-marino/AlchemyCloud/releases/tag/v0.2.0
[0.1.0]: https://github.com/kouya-marino/AlchemyCloud/releases/tag/v0.1.0
