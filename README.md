# AlchemyCloud — LiDAR PCD Editor & Occupancy Grid Exporter

[![PyPI](https://img.shields.io/pypi/v/alchemycloud.svg)](https://pypi.org/project/alchemycloud/)

A desktop Python app for editing point-cloud files (`.pcd`, `.ply`, `.xyz`) and exporting **2D occupancy grids** in ROS `map_server` format (`PGM` + `YAML`).

Install it with `pip install alchemycloud` and launch with the `alchemycloud` command. **Requires Python 3.10–3.12** — open3d publishes no wheel for 3.13+.

Built on **Open3D** (point-cloud kernels), **pyqtgraph** + **PyOpenGL** (embedded 3D canvas), and **PyQt5** (GUI shell). All heavy work runs on a `QThreadPool` worker so the UI stays responsive and a loading spinner animates during long ops.

---

## Features

### Point-cloud editing
- **Voxel Downsample** — reduce point density.
- **Statistical Outlier Removal (SOR)** — denoise. Matches PCL's `pcl_outlier_removal -method statistical`; defaults `neighbours = 30, std-ratio = 1.5`.
- **Radius Outlier Removal** — drop points whose neighbourhood is too sparse. Matches PCL's `pcl_outlier_removal -method radius`; defaults `nb_points = 30, radius = 0.10 m`.
- **Z-Clip (height filter)** — keep points within a Z range (great before generating a 2D map).
- **Remove Ground (RANSAC)** — fit and strip the dominant ground plane. Only a *near-horizontal* plane is accepted: RANSAC picks the plane with the most inliers, and in an indoor scan a long wall often beats the floor, so candidates are restricted to the lowest 30 % of the Z range and any fit more than 30° from horizontal is rejected and retried.
- **Level Ground** — auto-straighten a tilted scan: RANSAC fits the dominant *floor* plane (same near-horizontal constraint as above), the cloud is rotated so its normal aligns with +Z, and translated so the plane lands at `z = 0`. Status bar reports the tilt that was corrected. If no near-horizontal plane exists the operation is refused with an explanation rather than standing the scan on its side.
- **Qt-native screen-space Crop** — toggle Crop mode, drag a rectangle on the canvas, with *Keep inside* / *Remove inside* modes. Math runs on the full cloud so it's exact regardless of any display-LOD.
- **Z-Slab scope** — two horizontal planes at adjustable z_low / z_high, with in-slab points tinted red. Acts as a *scope filter*: while on, Crop and Eraser only affect in-slab points (out-of-slab points are protected).
- **Eraser tool** — drag a circular brush over points to delete them; honours Z-Slab when active.
- **Undo / Redo** (last 10 each) and **Reset to original**.
- Save edited cloud back to `.pcd` (binary or ASCII) or `.ply`. **File → Save as PLY…** (`Ctrl+Shift+S`) is a one-click PCD → PLY converter for downstream localisers (FAST-LIO etc.).

### Viewer
- **Embedded 3D canvas** inside the main window — orbit / pan / zoom; no pop-out window.
- **Color-by-axis** (X / Y / Z, viridis colormap; default Z) — switchable from the toolbar. A **colorbar legend** in the top-right of the canvas shows the current axis, min / mid / max values (RViz-style).
- **RGB (true colour)** — a fourth **Color** option renders the cloud's own per-point colour instead of a colormap, for files that carry an `rgb` field. Two companion controls handle exposure: **Bright** (gamma, default `0.62`) and **Auto-level** (per-channel 1–99 % stretch, default on) — indoor colour scans are usually under-exposed and render near-black raw. Set **Bright** to `1.00` with **Auto-level** off for the file's unmodified values. Both controls grey out unless Color is RGB, the colorbar hides (there's no scalar to legend), and a cloud with no colour falls back to flat grey with a status-bar note.
- **Adjustable point size** — a **Point** spin box (0.5–15 px) on the Edit toolbar enlarges the rendered points for clarity on sparse clouds; persists across reloads and filters.
- **Display: LOD ⟷ Full** toggle on the toolbar — randomly subsample large clouds to 500 k points for fast rendering, or render every point. The full cloud is always what filters / crop / export operate on.
- **Origin marker (X/Y/Z gizmo)** with a yaw angle. Drag the marker on the XY plane (LMB) or change yaw (RMB-drag).
- **Apply Marker as Origin** — bakes the marker pose into the cloud so Save writes the new coordinate frame.
- **Ground-reference grid** auto-sized to each loaded cloud.
- **Loading spinner** during slow operations; toolbars are disabled while busy.

### Occupancy grid export
Two methods, picked at export time:

| Method | What it does | When to use |
|---|---|---|
| `projection` | Top-down 2D projection of points whose Z is in `[z_min, z_max]`. | Fast. Standard for ROS navigation maps. |
| `voxel3d` | Full 3D voxelization at the chosen resolution, then collapse occupied voxels along Z. | More robust to point-density variation; slower. |

Two occupancy policies (selectable in the preview dialog):

- **Binary** (no threshold) — a cell is occupied if enough points project into it; empty cells are FREE or UNKNOWN (configurable).
- **Tristate** (threshold set, default) — cell **occupied** if any in-slab point has `z > threshold`; cell **free** if it has in-slab points but none above; cell **unknown** if empty. Matches the standard ROS navigation policy (threshold ≈ robot footprint height) and gives cleaner costmaps when there's floor/ceiling noise.

Two separate selectors control the background, because the policies mean different things by "empty". **"Unobserved cells (tristate)"** (default *Unknown, grey*) colours cells the scan never observed, while observed floor stays FREE. **"Non-obstacle cells (binary)"** (default *Free, white*) colours every cell that is not an obstacle — binary mode has no observed/unobserved distinction, so choosing *Unknown* there greys out the navigable interior too and the exported map contains no FREE cells at all.

Output is the standard ROS `map_server` format:
- **PGM**: `0` = occupied, `254` = free, `205` = unknown.
- **YAML**: `image`, `resolution`, `origin`, `negate`, `occupied_thresh`, `free_thresh`.

---

## Installation

### From PyPI (recommended)

```bash
pip install alchemycloud        # Python 3.10, 3.11 or 3.12
alchemycloud                    # launch
alchemycloud path/to/scan.pcd   # launch with a file already open
alchemycloud --version
alchemycloud -v                 # debug logging to stderr
```

This installs the `alchemycloud` console command (see **Running** below).

### From source (development)

The project uses **pyenv** with a virtualenv named `pcd_env`, pinned via `.python-version`.

```bash
git clone https://github.com/kouya-marino/AlchemyCloud.git
cd AlchemyCloud

# If pcd_env does not exist yet:
pyenv virtualenv 3.10.6 pcd_env
pyenv local pcd_env

# Editable install with dev extras (pytest, ruff)
pip install -e ".[dev]"
```

Dependencies (declared in `pyproject.toml`, mirrored in `requirements.txt`):
- `open3d>=0.17` — point-cloud I/O and kernels (voxel, SOR, RANSAC, transforms)
- `numpy>=1.21`
- `PyQt5>=5.15` — GUI shell
- `pyqtgraph>=0.13` + `PyOpenGL>=3.1` — embedded 3D canvas (`GLViewWidget` + `GLScatterPlotItem`)
- `PyYAML>=6.0` — `map_server` YAML writer
- `Pillow>=9.0` — PGM writer
- `scipy>=1.10` — stable rotation alignment for **Level Ground**
- `matplotlib>=3.7` — viridis colormap for color-by-axis and the colorbar legend (not used by RGB mode, which reads the file's own colours)

---

## Running

After installing, launch the GUI with the console command:

```bash
alchemycloud                  # blank session
alchemycloud scans.pcd        # load a PCD on startup
```

From a source checkout you can also run it without installing:

```bash
python main.py                # blank session
python main.py scans.pcd      # load a PCD on startup
```

---

## UI walkthrough

```
┌──────────────────────────────────────────────────────────────────────────┐
│ File   Edit   View                                                       │
│ [ Voxel…  Outlier…  Z-Clip…  Ground… │ ☐ Crop [Keep inside ▼] │          │
│  Undo Redo Reset │ Color:[Z▼] Bright:[0.62] ☑Auto-level │ LOD[●━]Full │
│ [ ☐ Move Marker  Reset Marker  Apply Marker as Origin │                  │
│   Marker: (0.000, 0.000, 0.000) θ=+0.0° ]                    ← Marker    │
│ [ ☐ Z-Slab (scope)  low: 0.0  high: 1.0 │                                │
│   ☐ Eraser  brush px: 30 ]                                   ← Tools     │
│ [ Generate Map (PGM + YAML)… ]                               ← Occupancy │
│ ──────────────────────────────────────────────────────────────────       │
│                                                                          │
│   ┌──────────────────────────────────────────────────────────────────┐   │
│   │                                                                  │   │
│   │           Embedded 3D canvas (orbit / pan / zoom)                │   │
│   │           coloured by Z (X / Y selectable in toolbar)            │   │
│   │                                                                  │   │
│   └──────────────────────────────────────────────────────────────────┘   │
│                                                                          │
│ /path/to/scans.pcd      13,000 points | X[-2,2] Y[-2,2] Z[-0,2] ...      │
└──────────────────────────────────────────────────────────────────────────┘
```

Both toolbars are dockable: drag them to the side, the bottom, or tear them
off entirely.

### Typical workflow

1. **File → Open PCD…** (`Ctrl+O`). The cloud appears immediately in the
   embedded 3D canvas — no second window opens.
2. Points are coloured by **Z** by default. Switch via the **Color** combo
   on the Edit toolbar (X / Y / Z, viridis colormap, or **RGB** for the
   file's own colours — see below).
3. Apply filters in any order — the canvas updates **in place** (camera pose
   preserved). Each filter is undoable.
   - For 2D-map generation, a common pipeline is:
     `Voxel Downsample` → `Statistical Outlier Removal` → `Remove Ground (RANSAC)` → `Z-Clip [0.1, 2.0]`.
4. **Crop** (Edit toolbar) — toggle the checkable Crop action on, pick
   *Keep inside* or *Remove inside* from the mode combo, then **drag a
   rectangle** directly on the canvas. On release the selection is applied
   and crop mode auto-disengages.
   - The math projects the **full** cloud through the current camera matrices,
     so the crop is exact regardless of any display LOD.
5. **Generate Map…** on the Occupancy toolbar (or `Ctrl+M`) — opens the
   live **Occupancy Grid Preview** dialog (non-modal). Adjust method,
   resolution, Z min/max, height threshold (tristate), padding, etc.
   The PGM preview updates as you drag the spin boxes (debounced
   120 ms, worker-thread build). Click **Save…** when the map looks
   right — writes PGM + YAML to the chosen path.

### Origin marker

The viewer draws a marker at the world origin: a small white dot with three
short axis lines (red X, green Y, blue Z). The marker has a position **and
a yaw angle** — the X and Y axes rotate with the yaw; the Z axis stays
vertical. The marker size scales with the loaded cloud.

To move the marker:

- Toggle **Move Marker** on the Marker toolbar (cursor changes to a 4-way
  arrow). Crop is automatically disengaged so the modes don't fight.
- **Left-click + drag** → moves the marker on the **XY plane**: the
  cursor pixel is ray-cast onto the marker's current Z plane and the
  marker follows. Press anywhere — the marker snaps to that point and
  then tracks the cursor.
- **Right-click + drag horizontally** → changes the **yaw angle**
  (0.5° per pixel; ~720 px spans a full turn).
- **Type exact coordinates** — the **X**, **Y**, and **θ°** spin boxes on
  the Marker toolbar place the marker precisely. X/Y are clamped to the
  loaded cloud's bounds (the map "scope"); yaw wraps at ±180°. Editing a
  box moves the marker live, and dragging / resetting the marker updates
  the boxes — they're always in sync.
- The label updates live: `Marker: (1.250, -2.500, 0.000)  θ=+45.0°`.
- **Reset Marker** sends it back to `(0, 0, 0)` and `θ = 0°`. Loading a
  new file also resets it.
- **Apply Marker as Origin** bakes the marker pose into the cloud:
  every point is rigidly transformed so that the marker becomes the new
  origin with no yaw. Saving thereafter writes the new frame. The action
  is undoable (Ctrl+Z) and *Reset to original* always brings the loaded
  file back. By default, moving the marker is **purely visual** — only
  this button changes the underlying point coordinates.

### Level Ground

If your scan comes in tilted (the floor isn't horizontal), click
**Level Ground** on the Edit toolbar. The app fits the dominant plane
via RANSAC, rotates the cloud so that plane's normal aligns with +Z, and
translates so the plane sits at `z = 0`. The status bar reports the
corrected tilt — e.g. `Levelled 10.00° (plane: …)`. Run it once;
running it again on an already-flat cloud is a no-op. Undoable.

Do this **before** Z-Clip / Z-Slab / occupancy export so "down" is
actually `-Z` for those tools.

### Z-Slab scope

Toggle **Z-Slab (scope)** on the Tools toolbar; two translucent yellow
planes appear at `z_low` and `z_high`. Adjust the bounds with the spin
boxes — **only in-slab points are rendered on the canvas while the scope
is on** (out-of-slab points are hidden but still preserved in memory).
The colorbar legend clamps to the visible range.

### True-colour (RGB) view

Set **Color** to **RGB** to render each point in the colour stored in the
file rather than a height ramp. This needs an `rgb` field in the `.pcd`
(PCL packs it as a float whose bytes are `0x00RRGGBB`) or `red`/`green`/
`blue` properties in a `.ply`; Open3D reads both. Clouds carrying only
`x y z`, normals, or `intensity` have no colour to show and render flat
grey — the status bar says so when you switch.

| Control | Default | Meaning |
|---|---|---|
| Bright | `0.62` | Display gamma. Below `1.00` brightens, above darkens. `1.00` leaves values untouched. |
| Auto-level | on | Stretches each channel to its 1–99 % range before gamma. |

The defaults exist because raw indoor colour scans are dark — a mean channel
value near `0.5` is normal, which renders almost black. For the file's exact
values, set **Bright** to `1.00` and switch **Auto-level** off.

Tone mapping is display-only: it never touches the point data, so saving
after cropping or filtering writes the original colours through unchanged.
Switching RGB → Z → RGB also returns the original colours, not a re-mapped
copy. Under **Display: LOD**, `voxel_down_sample` averages colour per voxel,
so the view stays representative.

**Z-Slab never removes points on its own.** It's a *scope filter*: while
the toggle is on, the other editing tools only affect in-slab points:

- **Crop** — `Remove inside` removes only in-slab points that fall inside
  the rectangle. `Keep inside` removes in-slab points outside the
  rectangle. Out-of-slab points are always preserved.
- **Eraser** — only erases in-slab points under the brush stroke.

Use it for things like deleting a horizontal noise band, a ceiling
fixture, or floor scribbles without risking the rest of the scene.

### Eraser tool

Toggle **Eraser** on the Tools toolbar; the system cursor is replaced by a
red brush outline. Drag with the left mouse button — every cloud point
whose 2D projection lies inside the painted stroke is deleted on release.

The Tools toolbar also exposes:

- **Brush px** — half-extent of the brush (4 – 200 px).
- **Shape** — `Circle` (default), `Square`, or `Rectangle`.
- **Aspect** — width / height ratio, only used when shape is `Rectangle`
  (default 2.0 = 2:1 horizontal brush).

`Brush px` is interpreted as the brush's *half-extent* for all shapes:

| Shape | Drawn footprint |
|---|---|
| Circle | radius = `Brush px` |
| Square | half-side = `Brush px` (side = `2 × Brush px`) |
| Rectangle | half-height = `Brush px`; half-width = `Brush px × Aspect` |

The math projects the *full* cloud through the live MVP matrix and
rasterises the stroke into a 2D canvas mask (PIL `ellipse` / `rectangle`),
so it's O(N) and works fine on multi-million-point clouds.

If Z-Slab is also enabled, the eraser only removes points whose Z is
inside `[z_low, z_high]`.

### Display: LOD vs Full

The Edit toolbar has a modern toggle switch labelled **Display: LOD ⟷ Full**:

- **LOD** (default, switch off) — clouds larger than **500 000 points** are
  voxel-downsampled **for display only** so the canvas stays interactive.
- **Full** (switch on) — every point of the full cloud is rendered. Slower
  on multi-million-point clouds but exact.

Either way, the full cloud is what every filter, crop, occupancy export,
and Save operates on. The LOD is purely a display optimisation. Switching
the toggle re-prepares the view on a worker thread with the loading
spinner.

### Loading indicator

Slow operations (file load/save, voxel/SOR/Z-clip/RANSAC, undo/reset/crop
on huge clouds, occupancy export) run on a worker thread with a centred
**rotating-arc spinner** dimming the canvas. Toolbar actions are disabled
while busy, and the spinner clears automatically once the result is
applied. Open3D releases the GIL inside its C++ kernels, so the spinner
keeps animating smoothly.

### Keyboard shortcuts

| Action | Shortcut |
|---|---|
| Open PCD | `Ctrl+O` |
| Save PCD | `Ctrl+S` |
| Save as PLY | `Ctrl+Shift+S` |
| Export map | `Ctrl+M` |
| Undo | `Ctrl+Z` |
| Redo | `Ctrl+Y` |
| Reset camera | `Ctrl+R` |
| Quit | `Ctrl+Q` |

---

## Occupancy grid parameters

| Field | Default | Meaning |
|---|---|---|
| Method | `projection` | `projection` or `voxel3d` (see table above). |
| Resolution (m/cell) | `0.05` | Side length of one map cell in meters. |
| Z min (m) | auto | Lower bound of the slab projected to the map (drops floor). |
| Z max (m) | auto | Upper bound of the slab (drops ceiling / overheads). |
| Min points / cell | `1` | Cell becomes occupied if at least this many points fall into it. |
| Padding (cells) | `5` | Extra cells of margin around the data footprint. |
| Unobserved cells (tristate) | `Unknown (grey)` | Colour for cells the scan never observed. Tristate only — observed floor stays FREE. |
| Non-obstacle cells (binary) | `Free (white)` | Colour for every non-obstacle cell. Binary only; `Unknown` here greys the free space too. |

The exported YAML can be consumed directly by ROS `map_server` (ROS 1) or `nav2_map_server` (ROS 2):

```yaml
image: scans_map.pgm
resolution: 0.05
origin: [-2.25, -2.25, 0.0]
negate: 0
occupied_thresh: 0.65
free_thresh: 0.196
```

---

## Project layout

```
AlchemyCloud/
├── alchemycloud/            # the installable package
│   ├── __init__.py
│   ├── app.py               # entry point — wires QApplication and MainWindow (main())
│   ├── main_window.py       # PyQt5 main window, toolbars, dialogs, slots
│   ├── embedded_viewer.py   # in-window 3D viewer (pyqtgraph GLViewWidget)
│   ├── loading_overlay.py   # rotating-arc loading spinner
│   ├── worker.py            # QRunnable + run_async() helper for off-main-thread work
│   ├── widgets.py           # reusable Qt widgets (ToggleSwitch, ColorbarLegend)
│   ├── pcd_ops.py           # point-cloud operations (filters + colorize + display LOD)
│   ├── occupancy.py         # 2D occupancy grid + PGM/YAML writer (binary + tristate)
│   └── occupancy_preview.py # live PGM preview dialog (sliders + debounced worker rebuild)
├── main.py                  # dev launcher (python main.py) — imports alchemycloud.app:main
├── pyproject.toml           # packaging metadata + deps (source of truth)
├── requirements.txt         # runtime deps (mirrors pyproject)
├── requirements-dev.txt     # dev-only: pytest
├── tests/                   # pytest suite + tests/README.md guide
├── .github/workflows/       # ci.yml (lint+test) and publish.yml (PyPI on tag)
├── LICENSE                  # MIT
├── README.md
├── version_data.md          # feature-per-version summary (0.1.0 → 1.3.0)
└── CHANGELOG.md             # detailed version history (Keep a Changelog format)
```

The version is single-sourced from `pyproject.toml` and exposed as
`alchemycloud.__version__` (also shown in the window title and by
`alchemycloud --version`). See [version_data.md](version_data.md) for a
feature-per-version summary, or [CHANGELOG.md](CHANGELOG.md) for the
detailed change-by-change history.

---

## Testing

The project ships a pytest suite under `tests/`. Install the dev
dependencies once (an editable install also makes the `alchemycloud`
package importable by the tests):

```bash
pip install -e ".[dev]"
```

Then run all tests:

```bash
pytest tests/ -v
```

What's covered (109 tests, ~2 s):

| File | Scope |
|---|---|
| `tests/test_pcd_ops.py` | Every public `pcd_ops` function (I/O, voxel, SOR, radius, z-clip, RANSAC, marker-frame, level-ground, colorize, display-LOD, stats). |
| `tests/test_occupancy.py` | Binary + tristate policies for both `projection` and `voxel3d`; full PGM + YAML round-trip via PIL. |
| `tests/test_viewer_helpers.py` | `project_points` / `mask_points_in_rect` / `mask_points_in_brush` against identity-MVP projections. |
| `tests/test_gui_smoke.py` | `MainWindow` construction (toolbars present), load → filter → undo → redo pipeline, `OccupancyPreviewDialog` initial render, brush-shape combo. |
| `tests/test_occupancy_geometry.py` | Map **geometry**: an asymmetric obstacle round-tripped through `(pixel, origin, resolution)` under both methods, PGM/YAML metadata, `voxel3d` hole-freeness, non-finite and oversize-grid handling, writer edge cases. |
| `tests/test_rgb_view.py` | The RGB colour mode: source-colour passthrough, tone mapping, the RGB → Z → RGB round trip, toolbar gating. |
| `tests/test_regressions.py` | Audit fixes in the Qt layer: GL blend mode, marker survival, history/dirty state, close prompt, busy gating, preview dialog staleness. |

GUI tests run with `QT_QPA_PLATFORM=offscreen` (set in `conftest.py`) so
no X server is needed. The pyqtgraph "Requires >= OpenGL 2.1" lines
during paint are offscreen-context noise, not failures.

See [`tests/README.md`](tests/README.md) for the full guide — how to
filter / debug, what each individual test asserts, which fixtures to
reuse, and how to add new tests.

## Troubleshooting

- **"No points in the requested Z range"** when exporting the map. Widen the Z range, or skip the RANSAC ground-removal step if you over-filtered.
- **`qt.qpa.plugin: Could not load the Qt platform plugin "xcb"`** on a headless box. The app needs a display; run it locally, or forward X11 over SSH (`ssh -X`).
- **`pyqtgraph.opengl: Requires >= OpenGL 2.1`** when running smoke tests under `QT_QPA_PLATFORM=offscreen`. Cosmetic — the offscreen Qt context reports a stub version. On a real display the check passes.
- **Slow load on huge clouds.** Voxel-downsample first (e.g., `0.05` m) before applying SOR or RANSAC — those are O(N). For display, the viewer auto-LODs anything over 500 k points by uniform random subsampling (filters, crop and export always use the full cloud).
- **"No near-horizontal plane found"** from Level Ground or Remove Ground. The dominant plane in the scan is a wall or ceiling, not a floor. Crop to the floor area first, or pass a larger `max_tilt_deg` if the scan really is that steep.
- **"Grid would be N x M cells, above the 25 M limit"** on export. Either the resolution is too fine for the cloud's extent, or — if the message also mentions sensor glitches — a handful of stray returns have inflated the bounding box. Run Statistical Outlier Removal or tighten the Z range.
- **Where are the logs?** `$XDG_STATE_HOME/alchemycloud/alchemycloud.log` on Linux (`~/.local/state/alchemycloud/` by default), `%LOCALAPPDATA%\alchemycloud\` on Windows, `~/Library/Logs/alchemycloud/` on macOS. Run with `-v` for debug output on stderr.
- **The crop selection feels off.** The rectangle is screen-space, so the crop depends on the current camera angle. Orbit to a clean top-down or side view before drawing the box if you want predictable axis-aligned cuts.

---

## Programmatic use

The non-GUI modules can be used directly:

```python
import numpy as np
from alchemycloud import pcd_ops, occupancy

# Load
pcd = pcd_ops.load_pcd("scans.pcd")

# Filter pipeline
pcd = pcd_ops.voxel_downsample(pcd, 0.05)
pcd = pcd_ops.statistical_outlier_removal(pcd, nb_neighbors=20, std_ratio=2.0)
pcd, _, plane = pcd_ops.remove_ground_ransac(pcd, distance_threshold=0.05)
pcd = pcd_ops.z_clip(pcd, 0.1, 2.0)

# Optional programmatic AABB crop:
# pcd = pcd_ops.crop_with_box(pcd, (-2, -2, 0), (2, 2, 2), keep_inside=True)

# Re-frame to a chosen origin + yaw (same math as "Apply Marker as Origin"):
# pcd = pcd_ops.transform_to_marker_frame(pcd, marker_pos=(1.2, -0.4, 0), marker_yaw=np.radians(30))

# Tint by axis (used by the embedded viewer's display cloud, but works standalone):
# pcd_ops.colorize_by_axis(pcd, axis="z")

# Build & save a ROS map_server-style occupancy grid
params = occupancy.GridParams(resolution=0.05, z_min=0.1, z_max=2.0, method="projection")
grid = occupancy.build(pcd, params)
occupancy.save(grid, "map.pgm", params)

# Save the (possibly transformed) cloud
pcd_ops.save_pcd(pcd, "scans_edited.pcd")
```

Available helpers in `pcd_ops`: `load_pcd`, `save_pcd`, `voxel_downsample`, `statistical_outlier_removal`, `z_clip`, `remove_ground_ransac`, `crop_with_box`, `colorize_by_axis`, `transform_to_marker_frame`, `display_downsample`, `stats`.
