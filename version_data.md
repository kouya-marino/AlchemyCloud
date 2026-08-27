# Version Data

A feature-per-version summary of the LiDAR PCD Editor & Occupancy Grid
Exporter, from the first release to today. Newest version first. For the
detailed, change-by-change record see [CHANGELOG.md](CHANGELOG.md).

**Current version: 1.3.0**

---

## 1.3.0 — Review, remediation & true colour *(2026-08-27)*

- **True-colour (RGB) view** — a fourth **Color** option renders each point in
  the colour stored in the file, with **Bright** (gamma) and **Auto-level**
  controls, because raw indoor colour scans render close to black.
- **Point clouds now render opaque and depth-tested** — pyqtgraph's additive
  default made overlapping points sum to white and occlude nothing.
- **Floor detection no longer picks walls** — RANSAC returned whichever plane
  had the most inliers, which on a real room scan tilted the cloud 85°.
- **Display LOD actually fills its budget** — the old voxel heuristic drew
  1,580 of 2.2 M points; now 500,000.
- **Honest saves** — a failed write raised nothing and reported success.
- **Unsaved-changes prompt** on close; there was no dirty tracking at all.
- **Separate background controls** for the binary and tristate policies, whose
  notions of "empty" differ.
- **Robustness** — non-finite points filtered, grid size capped, exact voxel3d
  indexing, and a dozen smaller Qt state fixes.
- **Logging, `sys.excepthook`, HiDPI, `--version`** and a single-sourced
  `__version__`.
- **Test suite 43 → 109**, adding map-geometry assertions that were entirely
  absent.
- **CI green for the first time** — both ruff gates had always failed.

See [CHANGELOG.md](CHANGELOG.md) for the change-by-change detail.

---

## 1.2.0 — Precision & hardening *(2026-06-02)*

- **Typed marker coordinates** — X / Y / yaw spin boxes on the Marker
  toolbar; clamped to the map scope, applied live, two-way synced with
  mouse drags and Reset.
- **Adjustable point size** — Edit-toolbar spin box (0.5–15 px) to enlarge
  rendered points for clarity on sparse clouds.
- **Pytest test suite** — 43 tests in `tests/` (pcd_ops, occupancy,
  viewer projection helpers, GUI smoke) with a `tests/README.md` guide.
- **Codebase hardening** — fixed close-during-work crash, stale occupancy
  save, main-thread freeze on reset-camera, laggy Z-Slab drags
  (colour cache + debounce), marker yaw ±180° sign flip, and migrated off
  the deprecated matplotlib colormap API.

## 1.1.0 — Costmap cleanup wave *(2026-05)*

- **Level Ground** — RANSAC auto-straighten so the dominant plane becomes
  horizontal and lands at z = 0.
- **PCL-style outlier removal** — Statistical (SOR) and Radius methods
  with PCL-matched defaults.
- **Tristate occupancy grids** — free / occupied / unknown classification
  via a height threshold (in addition to the original binary mode).
- **Live occupancy preview** — non-modal dialog with a debounced PGM
  preview that updates as you tune resolution / Z range / threshold.
- **Eraser brush shapes** — Circle, Square, and Rectangle (with aspect).
- **Save as PLY** — one-click PCD → PLY conversion (e.g. for FAST-LIO
  prior maps).

## 1.0.0 — First stable editing toolkit *(2026-05)*

- **Z-Slab scope** — two adjustable planes defining an editable region;
  in-slab points highlighted, out-of-slab points hidden while active.
- **Eraser brush** — paint over points to delete them; respects the
  Z-Slab scope.
- **Undo / Redo** — 10-deep history in both directions.
- *Milestone: the app is now a complete interactive single-window PCD
  editor.*

## 0.4.0 — Marker & responsive pipeline *(2026-05)*

- **Origin marker gizmo** — X/Y/Z axes plus a yaw rotation, on its own
  Move-Marker toolbar.
- **Marker drag** — left-drag moves it on the XY plane, right-drag spins
  the yaw.
- **Apply Marker as Origin** — rigidly re-frames the cloud so the marker
  becomes (0, 0, 0); undoable.
- **Async pipeline + loading overlay** — heavy operations run on a worker
  thread behind a rotating-arc spinner; the UI stays responsive.
- **Colorbar legend** — RViz-style viridis scale with min / mid / max
  ticks.

## 0.3.0 — Embedded in-window viewer *(2026-05)*

- **Embedded 3D viewer** — the cloud renders inside the main window
  (pyqtgraph), no more separate pop-out window.
- **Display-LOD** — clouds over 500 k points are downsampled for display
  only, with a LOD ⟷ Full toggle.
- **Qt-native crop** — drag a rectangle on the canvas; Keep-inside /
  Remove-inside modes.
- **Ground-reference grid** and **Reset camera**.

## 0.2.0 — Live 3D viewer & colour *(2026-05)*

- **Auto-launch 3D viewer** on file open.
- **Live updates** — the viewer refreshes after every filter.
- **Colour-by-axis** — viridis over X / Y / Z (default Z).
- **Dockable toolbars** — Edit and Occupancy, replacing the old button
  panels.

## 0.1.0 — Core engine & basic GUI *(2026-05)*

- **Point-cloud operations** — load/save, voxel downsample, statistical
  outlier removal, Z-clip, RANSAC ground removal, box crop, stats.
- **Occupancy grids** — 2D projection and 3D-voxel methods exported as
  ROS `map_server` PGM + YAML.
- **PyQt main window** — File/Edit menus, filter dialogs, map export.
- **Environment** — pyenv `pcd_env` on Python 3.10.6.

---

*This file is a curated overview. The authoritative, detailed history —
including bug fixes and internal refactors — lives in
[CHANGELOG.md](CHANGELOG.md).*
