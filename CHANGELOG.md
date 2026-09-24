# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project follows [Semantic Versioning](https://semver.org/). Version
numbers and groupings mirror the curated overview in
[version_data.md](version_data.md).

## [Unreleased]

### Added
- **Home-pose marker on the occupancy preview.** The preview drew the grid and
  nothing else, so there was no way to see where the robot's start pose landed
  before exporting. A cyan crosshair now marks world **(0, 0)** — the pose
  "Apply Marker as Origin" creates — and the status line reports the pixel it
  falls on and the cell value underneath, flagged red when that is UNKNOWN,
  OCCUPIED, or off the map entirely.

  Worth stating plainly because the two are easy to conflate: the YAML
  ``origin`` is the world position of the image's lower-left *corner*, not
  home. On the survey this came from, ``origin`` is (-50.081, -28.444) while
  home sits at pixel (1001, 2996) — and lands on an UNKNOWN cell, meaning a nav
  stack localising at (0, 0) would start in grey. That is exactly the kind of
  thing the overlay makes obvious and the old preview could not show at all.

  The marker is drawn *after* the pixmap is scaled to the label, at a fixed
  pixel size: painting it on the full-resolution grid first would shrink it
  along with everything else and it would vanish on a 2275 x 3566 map scaled
  into a 480 px panel. It is preview-only — the exported PGM keeps exactly its
  three map_server values, which a test asserts.

- **Marker Z, and a "Snap Z to floor" button.** `transform_to_marker_frame`
  always subtracted all three components, so applying the marker could already
  re-origin Z — there was simply no way to set it. The marker's Z was pinned at
  0 by every path: `reset_marker` zeroes it, `_fit_marker` only sizes the
  glyph, the canvas drag deliberately moves *on* the marker's current Z plane,
  and the typed-coordinate handler read only X and Y. `set_marker_position`
  even accepted a `z` that nothing ever passed. The new spin box closes that,
  clamped to the cloud's Z bounds with the same "never clamp the marker out of
  reach" guard as X/Y. Use it to put the origin at the robot's base height
  rather than on the floor; **Snap Z to floor** fills it from the constrained
  RANSAC fit for the floor-at-zero case.

- **Manual tilt trim: `Measure tilt`, `roll°`/`pitch°` and `Apply trim`** on the
  Tools toolbar, backed by `pcd_ops.apply_tilt()` and `pcd_ops.surface_tilt()`.

  Level Ground aligns the *dominant* plane and stops, which is not always
  enough. On a 180 m survey its floor and ceiling were measured 2.74° apart —
  they are not parallel, so **no single rigid rotation can level both**;
  flattening the floor necessarily tilts the ceiling. And even a 0.2° residual
  is 64 cm of height drift corner to corner on a site that size, which drags a
  costmap Z-slab across the map. **Measure tilt** reports both surfaces and
  says outright whether one rotation can fix them; the trim then lets the
  operator settle the compromise by eye.

  `apply_tilt` rotates about the cloud's **centroid**, not the world origin, so
  a survey sitting 150 m out stays where the camera is looking instead of
  swinging through space. (A rigid rotation cannot change a point set's extent
  whichever centre is used — only its position.) The spin boxes reset to zero
  after applying, so repeated clicks cannot silently compound the same
  rotation.

### Fixed
- **`surface_tilt`'s plane fit was degenerate on exactly the surfaces it
  measures.** Selecting a surface by a narrow Z-percentile window cuts a thin
  *stripe* across a tilted plane — the more tilted, the thinner — and the
  least-squares normal of a stripe is meaningless. It under-reported a known
  3.0° slab as 0.72°. The seed window is now wide, and the selection then
  iterates to follow the fitted plane rather than a horizontal slice. Caught by
  the tests written alongside it, which is also why the first measurements of
  the survey above (floor 4.66°, ceiling 0.75°) were wrong; the corrected
  figures are 1.59° and 1.93°.

## [1.3.0] — 2026-08-27

_Theme: review, remediation and true colour._

### Added
- **True-colour (RGB) view with a brightness control** — the **Color** combo
  on the Edit toolbar gains a fourth entry, **RGB**, which renders each point
  in the colour stored in the file (an `rgb` field in `.pcd`, `red`/`green`/
  `blue` in `.ply`) instead of a viridis ramp along an axis. Two companion
  controls set the exposure: **Bright** (display gamma, `0.20`–`3.00`,
  default `0.62`) and **Auto-level** (per-channel 1–99 % contrast stretch,
  default on). Raw indoor colour scans are heavily under-exposed — a mean
  channel value near `0.5` is typical, which renders close to black — so the
  defaults lift them; **Bright** `1.00` with **Auto-level** off reproduces the
  file's values exactly. Both controls are disabled unless Color is RGB, the
  colorbar legend hides in RGB mode (colour no longer encodes a scalar), and a
  cloud with no colour renders flat grey with a status-bar explanation.
  Backed by `EmbeddedViewer.set_color_axis("rgb")`, `set_rgb_gamma()`,
  `set_rgb_autolevel()` and `has_source_colors()`.

  Tone mapping is display-only and never touches the point data, so editing
  and saving a colour cloud round-trips its colours unchanged. The viewer now
  snapshots the display cloud's own RGB in `prepare_cloud` *before*
  `colorize_by_axis` overwrites it (promoted to `_source_rgb` in
  `apply_prepared_cloud`), which is what lets RGB → Z → RGB return the
  original colours rather than a re-mapped copy.

- **Background-colour choice on PGM export, as two separate controls** —
  the Occupancy Grid Preview dialog gains **"Unobserved cells (tristate)"**
  and **"Non-obstacle cells (binary)"**, each enabled only for its own
  policy. Tristate defaults to **Unknown (grey, 205)**; binary defaults to
  **Free (white, 254)**, matching `GridParams.unknown_outside_footprint`.
  Backed by `occupancy.GridParams.tristate_empty_unknown`.

  These are deliberately *not* one control. In tristate, "empty" means never
  observed and observed floor is still FREE, so grey marks genuinely unseen
  space. In binary there is no observed/unobserved distinction — every
  non-occupied cell is background — so grey there paints the navigable
  interior too and the exported map contains **zero FREE cells**. An earlier
  revision of this feature drove both flags from a single combo defaulting to
  grey, which silently flipped the binary default and produced maps a nav
  stack would treat as entirely unknown.

- **Logging and a bug-report surface** — `alchemycloud` now writes a rotating
  log (1 MB x 4) to a platform-appropriate directory (`$XDG_STATE_HOME/
  alchemycloud` on Linux, `%LOCALAPPDATA%` on Windows, `~/Library/Logs` on
  macOS) and installs a `sys.excepthook` that logs unhandled exceptions and
  shows them in a dialog. PyQt5 does not abort on an exception that escapes a
  slot — it printed a traceback to a stderr the user could not see and carried
  on with corrupt state. There was previously no logging anywhere.

- **`alchemycloud --version` / `-v` and a real argument parser.** `argv[1]`
  used to be passed straight to the loader whatever it was; a missing file now
  gives one clear message. The window title carries the version, so a
  screenshot identifies the build.

- **`__version__`** in `alchemycloud/__init__.py`, read from installed
  distribution metadata with a pyproject.toml fallback for source checkouts.
  The version was previously written independently in four places.

### Changed
- **`display_downsample` now uses uniform random sampling** instead of a
  cube-root voxel heuristic. The old formula assumed points fill a cube; a
  LiDAR floor scan is a slab, so the voxel came out one to two orders of
  magnitude too coarse — on a 2.2 M-point 24 x 36 x 1.8 m room scan it
  rendered **1,580 points, 0.32 % of the 500 k budget**, and its correction
  loop could only decimate further. Random sampling hits the budget exactly,
  is O(n), and keeps real source points rather than centroids.
- **Point clouds render depth-tested and opaque.** pyqtgraph defaults
  `GLScatterPlotItem` to `additive` (`GL_DEPTH_TEST` off,
  `glBlendFunc(SRC_ALPHA, ONE)`), so overlapping points summed instead of
  occluding: ~7 % of lit pixels blew past white on a 178 k-point room scan,
  and nothing hid anything, so the cloud read as haze rather than surfaces.
- **`min_points_per_cell` now applies to every method and policy** — raw
  points for `projection`, occupied voxels in the column for `voxel3d`, and
  returns above the threshold in tristate. It was documented as "ignored by
  voxel3d"; in reality any value above 1 produced a near-blank map, and
  tristate ignored it entirely so one stray return planted an obstacle.
- **`occupancy.save` writes `mode: trinary`** explicitly (nav2 defaults to it
  when absent; ROS 1 ignores the key) and **refuses `negate != 0`**, which
  would have inverted the map since pixels are always written in the
  `negate: 0` convention.
- `requires-python` is now `>=3.10,<3.13`. open3d 0.19 publishes cp38–cp312
  wheels and no sdist, so `pip install alchemycloud` could not resolve on
  3.13+ despite the metadata claiming support.
- `requirements-dev.txt` now matches pyproject's `dev` extra (adds
  `pytest-cov`, `ruff`). CI gates on ruff, so it belonged there.
- `publish.yml` runs lint + the test suite and checks the tag matches
  `project.version` before publishing. It previously published on any
  `v*.*.*` tag with no tests and no version check — all seven of
  `v0.1.0`…`v1.2.0` point at the same commit, so it fired seven times each
  trying to upload 1.2.0.

### Fixed
- **"Level Ground" and "Remove Ground" fitted any dominant plane, and on a
  real room scan picked a wall.** `segment_plane` returns the plane with the
  most inliers; on `ppt_15july.pcd` (2.2 M points, 24 x 36 x **1.8** m) the
  winning plane had normal ~[0.99, 0.11, 0.06] with 383 k inliers against the
  floor's 210 k. Levelling against it rotated the cloud 83–88° and turned a
  1.8 m-thick scan into a 21.6 m-thick one — then the map was exported from
  that. New `pcd_ops.fit_ground_plane` restricts candidates to the lowest
  `floor_band` (default 30 %) of the Z range *and* rejects any fit more than
  `max_tilt_deg` (default 30°) from horizontal, retrying on the remaining
  points. With no near-horizontal plane at all it raises with an explanation
  instead of applying a nonsensical rotation. Measured after the fix: tilt
  0.26°, Z extent preserved at 1.94 m. `remove_ground_ransac` also now selects
  inliers against the full cloud by point-to-plane distance.
- **A failed save reported success.** `save_pcd` discarded the `bool` from
  `o3d.io.write_point_cloud`, which returns `False` and logs to stderr rather
  than raising. Saving to a missing or unwritable directory wrote nothing,
  raised nothing, and the status bar said "Saved to <path>". It now raises,
  and rejects an extension-less filename Open3D cannot infer a format from.
- **No unsaved-changes protection.** There was no dirty tracking at all and
  `closeEvent` never prompted, so an editing session could be lost by closing
  the window. Mutating operations now set a dirty flag and closing offers
  Save / Discard / Cancel. Headless drivers can set
  `MainWindow.prompt_unsaved_on_close = False`.
- **One NaN or one glitch return broke export.** Nothing filtered non-finite
  points: a single `nan` raised "cannot convert float NaN to integer" and
  rendered the whole cloud black, and a single `x=1e9` specular return asked
  for a 1.82 TiB allocation. `load_pcd` now strips non-finite points (and
  reports how many), `colorize_by_axis` and `stats` are nan-safe, and
  `occupancy` filters non-finite rows before indexing.
- **No guard on grid size.** The resolution spin box's own minimum (0.005 m)
  on a 200 m scan asked for 1.6 G cells — 6.4 GB of counts plus a 1.6 GB
  image, i.e. swap-thrash rather than a clean error. `occupancy` now caps at
  25 M cells (both prototype scripts capped at 4 M) and, when a few stray
  points rather than the scan itself inflated the extent, says so and suggests
  outlier removal.
- **`voxel3d` punched one-cell holes through solid walls.** It converted
  integer voxel indices back to float centres `(k + 0.5) * res` and re-derived
  the cell with `floor((x - x_min) / res)`; that round trip is not exact in
  float64, so ~5 % of columns landed one cell low, merging with a neighbour and
  leaving a free gap in a wall — with the pattern changing as `padding_cells`
  changed. The grid frame is now built from the integer indices directly, exact
  by construction.
- **`padding_cells=0` crashed on a cloud with no extent in one axis** —
  `width = ceil(0/res) = 0`, then `clip(ix, 0, -1)` indexed a zero-width array
  (`IndexError: index -1 is out of bounds`). Both axes are now at least one
  cell.
- **`occupancy.save` truncated any filename containing a dot** —
  `with_suffix` turned `3F_map_v1.2` into `3F_map_v1.pgm`. The extension is
  now appended.
- **The preview dialog could write a stale grid and report "Saved".** If the
  pre-save rebuild failed, `_last_grid` still held the previous successful
  render and the pending-save path wrote that. Build failure is now recorded
  and `_do_save` refuses. The 50 ms self-rescheduling poll — which kept firing
  after the dialog closed, and could pop a file dialog from a dead window — is
  replaced by a flag the render's completion consumes.
- **A failed map export said nothing.** `occupancy.save` runs on the main
  thread, so the worker's error plumbing did not apply: the exception escaped
  the slot and the old stats stayed on screen. It now shows a dialog.
- **The preview legend contradicted the image beside it** — the swatches had
  occupied on white and unknown on dark, the inverse of the PGM's
  occupied=0 / free=254 / unknown=205.
- **Opening a second file left the first file's redo stack live.** `load_path`
  cleared `history` but not `redo_stack`, so Redo popped a cloud belonging to
  the *previous* file into `current_pcd` while `current_path` still named the
  new one — and Ctrl+S then wrote the old cloud under the new file's name.
- **Every edit destroyed the marker pose.** `apply_prepared_cloud` called
  `reset_marker()` unconditionally, so voxel, SOR, z-clip, crop, an eraser
  stroke, undo, redo and even Reset Camera zeroed a marker the user had
  positioned — after which "Apply Marker as Origin" reported "Marker is
  already at the origin". It is now opt-in via `reset_marker=True`, passed
  only on file load and after applying the marker.
- **The marker teleported when only yaw was edited.** `_refresh_state`
  clamped the X/Y spin ranges to the cloud bounding box, pulling a marker
  legitimately placed outside the scanned area back inside. The range now
  always includes the marker's current position.
- **Menus and keyboard shortcuts stayed live during long operations.** The
  loading overlay only eats mouse events on the canvas, so Ctrl+O mid-RANSAC
  could start a second job. Menu actions are now disabled while busy, and
  `action_open` has its own guard.
- **Z-Slab bounds had two sources of truth.** The viewer normalised an
  inverted low/high pair by swapping; crop and eraser read the raw spin boxes
  and became silent no-ops. Both now read `viewer.slab_bounds()`.
- **Enabling Crop or Marker left the Eraser button looking active.** The
  viewer cleared eraser state internally but the toolbar button kept its
  checked appearance, so two tools appeared active and Eraser needed two
  clicks to re-enable.
- **Marker clicks above the horizon landed behind the camera.**
  `screen_to_world_xy` accepted a negative ray parameter, returning the
  mirrored intersection instead of no hit.
- **A fast eraser drag erased a dotted line.** Qt delivers one move event per
  repaint, so widely spaced samples each stamped an isolated disc. The stroke
  is now interpolated at half the brush radius.
- **Eraser mode swallowed every mouse button**, so the camera could not be
  orbited or panned without toggling the eraser off. Only the left button is
  captured now, matching crop mode.
- **A second "Generate Map" click orphaned the first dialog**, which stayed on
  screen with a live Save button exporting a cloud that no longer matched the
  window. Previews are now tracked, the old one is closed, and the open dialog
  follows the edited cloud via a new `MainWindow.cloudChanged` signal and
  `OccupancyPreviewDialog.set_cloud`.
- **`closeEvent` dropped the only reference to a still-queued worker**, freeing
  the C++ `QRunnable` under the pool thread. The reference is now retained.
- **`_ParamDialog` accepted physically impossible values** (voxel size 0,
  negative radius, 0 RANSAC iterations) and surfaced the failure as a raw
  traceback. Fields now carry explicit bounds.
- **`level_ground` ignored `translate_to_zero` when the cloud was already
  level**, so a flat scan whose floor sat at z=5 stayed at z=5.
- **CI was red on both lint gates** — 25 `ruff check` errors and 14 of 17
  files failing `ruff format --check`, with `build` depending on `lint`, so
  nothing could ship through CI. Both gates now pass.
- The test suite grew from 44 to **109 tests**, adding the coverage the audit
  found missing: map geometry is now asserted by round-tripping an asymmetric
  obstacle through `(pixel, origin, resolution)` (hard-coding the YAML origin
  to (0,0,0) or deleting both vertical flips previously passed the whole
  suite), plus the RANSAC wall case, `voxel3d` hole-freeness, non-finite and
  oversize-grid handling, the GL blend mode, and the Qt state machine.

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

[Unreleased]: https://github.com/kouya-marino/AlchemyCloud/compare/v1.3.0...HEAD
[1.3.0]: https://github.com/kouya-marino/AlchemyCloud/releases/tag/v1.3.0
[1.2.0]: https://github.com/kouya-marino/AlchemyCloud/releases/tag/v1.2.0
[1.1.0]: https://github.com/kouya-marino/AlchemyCloud/releases/tag/v1.1.0
[1.0.0]: https://github.com/kouya-marino/AlchemyCloud/releases/tag/v1.0.0
[0.4.0]: https://github.com/kouya-marino/AlchemyCloud/releases/tag/v0.4.0
[0.3.0]: https://github.com/kouya-marino/AlchemyCloud/releases/tag/v0.3.0
[0.2.0]: https://github.com/kouya-marino/AlchemyCloud/releases/tag/v0.2.0
[0.1.0]: https://github.com/kouya-marino/AlchemyCloud/releases/tag/v0.1.0
